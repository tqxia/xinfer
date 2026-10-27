"""Export and AOT-compile the complete Qwen3.5 text model for one fixed step."""

import argparse
import importlib.metadata
import inspect
import json
import os
import time
from pathlib import Path

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5

from compiler.qwen35.config import ARTIFACT_DIR, CACHE_LENGTH, PREFILL, TEXT_DIR
from compiler.common.artifacts import sha256, ROOT
from compiler.qwen35.model import TextStep, configure_text, load_model, state_specs, zero_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("prefill", "decode"))
    parser.add_argument("--model-dir", type=Path, default=TEXT_DIR)
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "text")
    parser.add_argument("--eager-numerics", action="store_true",
                        help="Keep native mean reductions and disable FP contraction")
    args = parser.parse_args()
    configure_text()
    args.output.mkdir(parents=True, exist_ok=True)
    package = args.output / f"{args.kind}.pt2"
    if package.exists():
        parser.error(f"Refusing to overwrite {package}")
    if args.eager_numerics:
        os.environ["TRITON_DEFAULT_FP_FUSION"] = "0"
        from torch._inductor.decomposition import decompositions
        from torch._inductor.lowering import make_fallback
        decompositions.pop(torch.ops.aten.mean.dim, None)
        make_fallback(torch.ops.aten.mean.dim, warn=False)
    model = load_model(args.model_dir)
    config = model.config
    length = PREFILL if args.kind == "prefill" else 1
    torch.manual_seed(35)
    inputs = (torch.randint(0, config.vocab_size, (1, length), device="cuda"),
              torch.arange(length, device="cuda"), zero_state(config))
    with torch.inference_mode():
        start = time.perf_counter()
        exported = torch.export.export(TextStep(model), inputs, strict=True)
        exported.validate()
        export_seconds = time.perf_counter() - start
        assert not exported.graph_signature.buffers_to_mutate
        assert not exported.graph_signature.user_inputs_to_mutate
        (args.output / f"{args.kind}.graph.txt").write_text(exported.graph_module.code)
        print(f"Exported {args.kind}: {len(list(exported.graph.nodes))} nodes; no input mutations", flush=True)
        options = {"emulate_precision_casts": True}
        start = time.perf_counter()
        torch._inductor.aoti_compile_and_package(exported, package_path=str(package), inductor_configs=options)
        compile_seconds = time.perf_counter() - start
    metadata = dict(
        kind=args.kind, tokens=length, cache_length=CACHE_LENGTH, vocabulary=config.vocab_size,
        hidden_size=config.hidden_size, layer_types=config.layer_types,
        checkpoint=json.loads((args.model_dir / "manifest.json").read_text()),
        state=state_specs(config),
        input_order=["input_ids", "cache_position"] + [s["name"] for s in state_specs(config)],
        output_order=["last_logits"] + [s["name"] for s in state_specs(config)],
        logits=dict(shape=[1, config.vocab_size], dtype="bfloat16"), input_mutation=False,
        generation_config=model.generation_config.to_dict(),
        torch=torch.__version__, cuda=torch.version.cuda, cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
        sm=torch.cuda.get_device_capability(), tf32=False, attention_backend="efficient_sdpa",
        versions={name: importlib.metadata.version(name) for name in
                  ("transformers", "triton", "safetensors", "huggingface-hub", "tokenizers")},
        hf_source_sha256=sha256(inspect.getfile(modeling_qwen3_5)),
        source_sha256={name: sha256(ROOT / name) for name in ('compiler/qwen35/config.py', 'compiler/qwen35/model.py', 'compiler/qwen35/export.py', 'compiler/common/artifacts.py')},
        parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
        rope_frequency_dtype=str(model.model.rotary_emb.inv_freq.dtype),
        tied_embeddings=model.lm_head.weight is model.model.embed_tokens.weight,
        strict_export=True, graph_nodes=len(list(exported.graph.nodes)), inductor_configs=options,
        eager_numerics=args.eager_numerics,
        export_seconds=export_seconds, compile_seconds=compile_seconds,
        package_bytes=package.stat().st_size, package_sha256=sha256(package),
    )
    (args.output / f"{args.kind}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: metadata[k] for k in ("kind", "graph_nodes", "export_seconds", "compile_seconds", "package_bytes")}), flush=True)


if __name__ == "__main__":
    main()
