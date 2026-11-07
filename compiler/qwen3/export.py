"""Export Qwen prefill and decode as one shared-weight AOTInductor bundle."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import torch

from compiler.common.artifacts import MODEL_MANIFEST_NAME, read_json
from compiler.common.context import model_context_limit, resolve_context_length
from compiler.common.attention import cache_extent
from compiler.common.weights import compile_shared
from compiler.common.metadata import make_metadata, state_spec
from compiler.qwen3.adapters.qwen3_flash import FlashStep
from compiler.qwen3.model import MAX_PREFILL, allocate_cache, load_model
from compiler.qwen3.config import DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("artifacts/aoti"))
    parser.add_argument(
        "--max-context-length",
        type=int,
        help="Compiled KV-cache capacity; defaults to model max_position_embeddings",
    )
    args = parser.parse_args()
    if args.output.is_symlink() or (
        args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir()))
    ):
        parser.error(f"Output must be a new or empty directory: {args.output}")
    try:
        context_config = read_json(args.model_dir / "config.json")
        capacity = resolve_context_length(context_config, args.max_context_length)
        ceiling = model_context_limit(context_config)
    except ValueError as error:
        parser.error(str(error))
    print(
        json.dumps(
            dict(
                event="context",
                cache_length=capacity,
                model_max_position_embeddings=ceiling,
                max_prefill_tokens=MAX_PREFILL,
            )
        ),
        flush=True,
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from torch.fx.experimental import _config as symbolic_config

    if symbolic_config.backed_size_oblivious:
        raise ValueError("Flash must export with backed_size_oblivious disabled")
    model = load_model(args.model_dir)
    step = FlashStep(model)
    manifest = json.loads((args.model_dir / MODEL_MANIFEST_NAME).read_text())
    revision = manifest["revision"]
    checkpoint = dict(model_id=manifest["repo_id"], revision=revision, manifest=manifest)
    # Retain BF16 cast boundaries; fused arithmetic can still differ from eager.
    options = {"emulate_precision_casts": True}
    from torch._inductor.runtime.triton_helpers import get_backend_options

    def export_entry(kind, staging_directory):
        length = 1 if kind == "decode" else 16
        cache = allocate_cache(model.config, capacity)
        inputs = (
            torch.ones((1, length), dtype=torch.int64, device="cuda"),
            torch.arange(length, device="cuda"),
            cache,
            cache_extent(32),
        )
        token_shape = {}
        if kind == "prefill":
            seq = torch.export.Dim("tokens", min=2, max=MAX_PREFILL)
            token_shape = {1: seq}
        extent = torch.export.Dim("kv_extent", min=2, max=capacity + 1)
        shapes = (
            token_shape,
            {} if kind == "decode" else {0: seq},
            tuple({} for _ in cache),
            {0: extent},
        )
        start = time.perf_counter()
        exported = torch.export.export(step, inputs, dynamic_shapes=shapes, strict=False)
        exported.validate()
        targets = [n.target for n in exported.graph.nodes if n.op == "call_function"]
        sites = targets.count(torch.ops.xinfer.flash_attention.default)
        updates = targets.count(torch.ops.aten.index_copy_.default)
        assert sites == model.config.num_hidden_layers and updates == 2 * sites
        assert torch.ops.aten.scaled_dot_product_attention.default not in targets
        adapter_report = {"name": "qwen3_flash_v1", "replacements": sites, "cache_updates": updates}
        # Text dumps expose mutation/shape behavior without another copy of the weights.
        (staging_directory / f"{kind}.graph.txt").write_text(exported.graph_module.code)
        print(f"Exported {kind}; compiling...", flush=True)

        def metadata_factory(_compiled_path):
            state = [
                state_spec(layer, name, list(cache[0].shape), "bfloat16", "inplace")
                for layer in range(model.config.num_hidden_layers)
                for name in ("key", "value")
            ]
            metadata = make_metadata(
                model_type="qwen3",
                entry_point=kind,
                token_range=[1, 1] if kind == "decode" else [2, MAX_PREFILL],
                checkpoint=checkpoint,
                torch_version=torch.__version__,
                cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
                cuda_compute_capability=list(torch.cuda.get_device_capability()),
                cache_length=capacity,
                vocab_size=model.config.vocab_size,
                eos_token_ids=model.generation_config.eos_token_id,
                state=state,
                triton_enable_fp_fusion=get_backend_options()["enable_fp_fusion"],
                model_specific={},
            )
            print(
                json.dumps(
                    dict(
                        event="export",
                        model_type="qwen3",
                        entry_point=kind,
                        build=dict(
                            strict_export=False,
                            torch_cuda_version=torch.version.cuda,
                            versions={
                                name: importlib.metadata.version(name)
                                for name in (
                                    "transformers",
                                    "triton",
                                    "safetensors",
                                    "tokenizers",
                                    "huggingface-hub",
                                )
                            },
                        ),
                        diagnostics=dict(
                            graph_nodes=len(list(exported.graph.nodes)),
                            export_and_compile_seconds=time.perf_counter() - start,
                            model_adapter=adapter_report,
                        ),
                    )
                ),
                flush=True,
            )
            return metadata

        return exported, metadata_factory

    with torch.inference_mode():
        compile_shared(
            args.output,
            export_entry,
            identity=dict(model="qwen3-0.6b", revision=revision),
            inductor_configs=options,
        )
    print(f"Saved prefill/decode bundle to {args.output}", flush=True)


if __name__ == "__main__":
    main()
