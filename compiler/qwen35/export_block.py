"""Compile fixed-shape DeltaNet prefill/decode packages for native execution."""

import argparse
import importlib.metadata
import inspect
import json
import time
from pathlib import Path

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5

from block import BlockStep, HIDDEN, MODEL_DIR, PREFILL, configure, load_block, sha256, zero_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("prefill", "decode"))
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output", type=Path, default=Path("artifacts/qwen35/block"))
    args = parser.parse_args()
    configure()
    args.output.mkdir(parents=True, exist_ok=True)
    package = args.output / f"{args.kind}.pt2"
    if package.exists():
        parser.error(f"Refusing to overwrite {package}")
    block, _ = load_block(args.model_dir)
    length = PREFILL if args.kind == "prefill" else 1
    torch.manual_seed(35)
    inputs = (torch.randn(1, length, HIDDEN, device="cuda", dtype=torch.bfloat16), *zero_state())
    with torch.inference_mode():
        start = time.perf_counter()
        exported = torch.export.export(BlockStep(block), inputs, strict=True)
        export_seconds = time.perf_counter() - start
        (args.output / f"{args.kind}.graph.txt").write_text(exported.graph_module.code)
        print(f"Exported {args.kind}: {len(list(exported.graph.nodes))} nodes", flush=True)
        assert not exported.graph_signature.buffers_to_mutate
        assert not exported.graph_signature.user_inputs_to_mutate
        options = {"emulate_precision_casts": True}
        start = time.perf_counter()
        torch._inductor.aoti_compile_and_package(
            exported, package_path=str(package), inductor_configs=options)
        compile_seconds = time.perf_counter() - start
    metadata = {
        "kind": args.kind, "tokens": length, "torch": torch.__version__,
        "cuda": torch.version.cuda, "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
        "sm": torch.cuda.get_device_capability(), "tf32": False,
        "versions": {n: importlib.metadata.version(n) for n in
                     ("transformers", "triton", "safetensors", "huggingface-hub", "tokenizers")},
        "hf_source_sha256": sha256(inspect.getfile(modeling_qwen3_5)),
        "source_sha256": {name: sha256(Path(__file__).parent / name)
                          for name in ("block.py", "export_block.py")},
        "parameter_dtypes": {n: str(p.dtype) for n, p in block.named_parameters()},
        "checkpoint": json.loads((args.model_dir / "manifest.json").read_text()),
        "inputs": [dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype)) for t in inputs],
        "outputs": ["output", "next_conv", "next_recurrent"], "input_mutation": False,
        "strict_export": True, "graph_nodes": len(list(exported.graph.nodes)),
        "inductor_configs": options, "kernel_path": "upstream PyTorch fallback",
        "export_seconds": export_seconds, "compile_seconds": compile_seconds,
        "package_bytes": package.stat().st_size, "package_sha256": sha256(package),
    }
    (args.output / f"{args.kind}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
