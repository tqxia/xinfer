"""Export and compile one T=2..512 DeltaNet prefill package with a native chunk op."""

import argparse
import inspect
import json
import time
from pathlib import Path

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5

from compiler.common.artifacts import ROOT, sha256
from compiler.qwen35.config import ARTIFACT_DIR, configure
from compiler.qwen35.adapters.native_delta import NativeDeltaPrefill, load_library
from compiler.qwen35.diagnostics.block import MODEL_DIR, input_shapes, load_block, zero_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "dynamic-block")
    args = parser.parse_args()
    configure()
    args.output.mkdir(parents=True, exist_ok=False)
    load_library(args.library)
    block, config = load_block(args.model_dir)
    torch.manual_seed(35)
    inputs = (torch.randn(input_shapes(config, 32)[0], device="cuda", dtype=torch.bfloat16),
              *zero_state(config))
    with torch.inference_mode():
        start = time.perf_counter()
        exported = torch.export.export(NativeDeltaPrefill(block), inputs, strict=True,
            dynamic_shapes=({1: torch.export.Dim("tokens", min=2, max=512)}, None, None))
        export_seconds = time.perf_counter() - start
        assert not exported.graph_signature.user_inputs_to_mutate
        assert not exported.graph_signature.buffers_to_mutate
        custom = [n for n in exported.graph.nodes if n.target == torch.ops.xinfer.delta_prefill.default]
        assert len(custom) == 1, "Expected one opaque DeltaNet operator"
        torch.export.save(exported, args.output / "prefill.exported.pt2")
        (args.output / "prefill.graph.txt").write_text(exported.graph_module.code)
        print(f"Dynamic export succeeded: {exported.range_constraints}", flush=True)
        options = {"emulate_precision_casts": True}
        start = time.perf_counter()
        torch._inductor.aoti_compile_and_package(exported,
            package_path=str(args.output / "prefill.pt2"), inductor_configs=options)
        compile_seconds = time.perf_counter() - start
    sources = ("compiler/qwen35/adapters/native_delta.py", "compiler/qwen35/diagnostics/export_dynamic_block.py",
               "compiler/qwen35/diagnostics/block.py", "compiler/qwen35/config.py", "runtime/qwen35/delta_ops.cpp")
    report = dict(torch=torch.__version__, cuda=torch.version.cuda, cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
                  sm=torch.cuda.get_device_capability(), min_tokens=2, max_tokens=512, sample_tokens=32,
                  checkpoint=json.loads((args.model_dir / "manifest.json").read_text()),
                  hf_source_sha256=sha256(inspect.getfile(modeling_qwen3_5)),
                  source_sha256={name: sha256(ROOT / name) for name in sources},
                  library_sha256=sha256(args.library), inductor_configs=options,
                  inputs=[dict(shape=list(t.shape), dtype=str(t.dtype)) for t in inputs],
                  range_constraints=str(exported.range_constraints), strict_export=True,
                  graph_nodes=len(list(exported.graph.nodes)), custom_ops=len(custom), input_mutation=False,
                  export_seconds=export_seconds, compile_seconds=compile_seconds)
    (args.output / "prefill.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
