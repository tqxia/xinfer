"""Export one Qwen entry point and compile it to a native AOTInductor package."""

import argparse
import json
import time
from pathlib import Path

import torch

from compiler.qwen3.adapters.qwen3_flash import FlashStep
from compiler.qwen3.custom_ops.attention import OPERATOR_CONTRACT
from compiler.qwen3.model import CACHE_LENGTH, MAX_PROMPT, allocate_cache, load_model, cache_extent
from compiler.qwen3.prepare import DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("decode", "prefill"))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("artifacts/aoti"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    package = args.output / f"{args.kind}.pt2"
    if package.exists():
        parser.error(f"Package already exists: {package}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from torch.fx.experimental import _config as symbolic_config
    if symbolic_config.backed_size_oblivious:
        raise ValueError("Flash must export with backed_size_oblivious disabled")
    export_options = {"backed_size_oblivious": symbolic_config.backed_size_oblivious}
    model = load_model(args.model_dir)
    step = FlashStep(model)
    with torch.inference_mode():
        length = 1 if args.kind == "decode" else 16
        cache = allocate_cache(model.config)
        inputs = (
            torch.ones((1, length), dtype=torch.int64, device="cuda"),
            torch.arange(length, device="cuda"),
            cache,
            cache_extent(32),
        )
        token_shape = {}
        if args.kind == "prefill":
            seq = torch.export.Dim("tokens", min=2, max=MAX_PROMPT)
            token_shape = {1: seq}
        extent = torch.export.Dim("kv_extent", min=2, max=CACHE_LENGTH + 1)
        shapes = (token_shape, {} if args.kind == "decode" else {0: seq},
                  tuple({} for _ in cache), {0: extent})
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
        (args.output / f"{args.kind}.graph.txt").write_text(exported.graph_module.code)
        print(f"Exported {args.kind}; compiling...", flush=True)
        # Retain BF16 cast boundaries; fused arithmetic can still differ from eager.
        options = {"emulate_precision_casts": True}
        torch._inductor.aoti_compile_and_package(
            exported, package_path=str(package), inductor_configs=options,
        )
        from torch._inductor.runtime.triton_helpers import get_backend_options
        metadata = {
            "entry_point": args.kind,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
            "cuda_compute_capability": list(torch.cuda.get_device_capability()),
            "model_revision": json.loads((args.model_dir / "model_manifest.json").read_text())["revision"],
            "batch_size": 1,
            "token_length": [length, length] if args.kind == "decode" else [2, MAX_PROMPT],
            "cache_length": CACHE_LENGTH,
            "layers": model.config.num_hidden_layers,
            "kv_heads": model.config.num_key_value_heads,
            "head_dim": model.config.head_dim,
            "vocab_size": model.config.vocab_size,
            "dtype": "bfloat16",
            "inductor_configs": options,
            "export_options": export_options,
            "attention_backend": "flash",
            "cache_input": "full_capacity_with_extent",
            "valid_cache_length": [1, CACHE_LENGTH],
            "kv_extent": {"device": "cpu", "dtype": "uint8", "shape": "[valid_length + 1]", "data_read": False},
            "triton_enable_fp_fusion": get_backend_options()["enable_fp_fusion"],
            "flat_input_order": "input_ids, cache_position, K0, V0, K1, V1, ..., kv_extent",
            "flat_input_count": 3 + len(cache),
            "input_alignment_bytes": 16,
            "position_contract": "contiguous absolute positions; populated prefix; final position < cache_length",
            "cache_shape": list(cache[0].shape),
            "cache_stride": list(cache[0].stride()),
            "cache_padding": 0,
            "output": "last-token logits [1, vocab_size]",
            "eos_token_ids": model.generation_config.eos_token_id,
            "export_and_compile_seconds": time.perf_counter() - start,
        }
        metadata.update(custom_attention=OPERATOR_CONTRACT, flash_implementation="adapter",
                        model_adapter=adapter_report)
        (args.output / f"{args.kind}.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Saved {package} ({package.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
