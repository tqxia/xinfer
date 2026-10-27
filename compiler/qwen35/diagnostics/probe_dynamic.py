"""Record whether the pinned HF block supports bounded dynamic prefill export."""

import argparse
import inspect
import json
import traceback
from pathlib import Path

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5

from compiler.qwen35.config import ARTIFACT_DIR, configure
from compiler.common.artifacts import sha256, ROOT
from compiler.qwen35.diagnostics.block import MODEL_DIR, BlockStep, input_shapes, load_block, zero_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "dynamic-prefill")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure()
    block, config = load_block(args.model_dir)
    torch.manual_seed(35)
    results = []
    with torch.inference_mode():
        for maximum in (512, 64):
            inputs = (torch.randn(input_shapes(config, 32)[0], device="cuda", dtype=torch.bfloat16),
                      *zero_state(config))
            try:
                exported = torch.export.export(BlockStep(block), inputs, strict=True,
                    dynamic_shapes=({1: torch.export.Dim("tokens", min=2, max=maximum)}, None, None))
                (args.output / f"max-{maximum}.graph.txt").write_text(exported.graph_module.code)
                result = dict(maximum=maximum, exported=True, nodes=len(list(exported.graph.nodes)),
                              constraints=str(exported.range_constraints))
            except Exception as error:
                (args.output / f"max-{maximum}.log").write_text(traceback.format_exc())
                result = dict(maximum=maximum, exported=False, error_type=type(error).__name__, error=str(error))
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "error"}), flush=True)
    report = dict(torch=torch.__version__, sample_tokens=32, results=results,
                  checkpoint=json.loads((args.model_dir / "manifest.json").read_text()),
                  hf_source_sha256=sha256(inspect.getfile(modeling_qwen3_5)),
                  source_sha256={name: sha256(ROOT / name) for name in ('compiler/qwen35/config.py', 'compiler/qwen35/diagnostics/block.py', 'compiler/qwen35/diagnostics/probe_dynamic.py', 'compiler/common/artifacts.py')})
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
