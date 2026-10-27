"""Reproduce Q35-NUM-001 at the layer-0 DeltaNet gated-normalization input.

This diagnostic exposes intermediate graph outputs; it does not change the
production exporter. Exit zero means the issue reproduced, not that it is fixed.
"""

import argparse
import inspect
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5

from compiler.qwen35.config import ARTIFACT_DIR, TEXT_DIR
from compiler.common.artifacts import sha256
from compiler.qwen35.diagnostics.block import BlockStep
from compiler.qwen35.model import configure_text, load_model, zero_state


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=TEXT_DIR)
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "rounding-probe")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_text()
    os.environ["TRITON_DEFAULT_FP_FUSION"] = "0"
    from torch._inductor.decomposition import decompositions
    from torch._inductor.lowering import make_fallback
    # Remove reduction-order differences to isolate the cast boundary.
    for op in (torch.ops.aten.mean.dim, torch.ops.aten.sum.dim_IntList):
        decompositions.pop(op, None)
        make_fallback(op, warn=False)
    model = load_model(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    ids = torch.tensor([tokenizer.encode("The", add_special_tokens=False)], device="cuda")
    hidden = model.model.layers[0].input_layernorm(model.model.embed_tokens(ids))
    inputs = (hidden, *zero_state(model.config)[:2])
    wrapper = BlockStep(model.model.layers[0].linear_attn)
    graph = torch.export.export(wrapper, inputs, strict=True).module()
    nodes = [n for n in graph.graph.nodes if n.op == "call_function" and isinstance(n.meta.get("val"), torch.Tensor)]
    boundary = [n for n in nodes if n.target == torch.ops.aten.to.dtype and
                n.meta["val"].dtype == torch.float32 and n.args[0].meta["val"].dtype == torch.bfloat16 and
                "hidden_states = hidden_states.to(torch.float32)" in n.meta.get("stack_trace", "")]
    assert len(boundary) == 1, "Pinned HF cast location changed; inspect the graph"
    after = nodes.index(boundary[0])
    before = nodes.index(boundary[0].args[0])
    next(n for n in graph.graph.nodes if n.op == "output").args = (tuple(nodes),)
    graph.graph._codegen = torch.fx.graph.CodeGen()
    graph.recompile()
    (args.output / "graph.py").write_text(graph.code)
    expected = graph(*inputs)
    exported = torch.export.export(graph, inputs, strict=True)
    package = args.output / "probe.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package),
                                            inductor_configs={"emulate_precision_casts": True})
    actual = torch._inductor.aoti_load_package(str(package))(*inputs)
    a, b = actual[after], expected[after]
    report = dict(issue="Q35-NUM-001", torch=torch.__version__, cuda=torch.version.cuda,
                  hf_source_sha256=sha256(inspect.getfile(modeling_qwen3_5)), source_sha256=sha256(__file__),
                  checkpoint=json.loads((args.model_dir / "manifest.json").read_text()),
                  package_sha256=sha256(package), before_node=nodes[before].name, after_node=nodes[after].name,
                  bf16_equal=torch.equal(actual[before], expected[before]), fp32_equal=torch.equal(a, b),
                  mismatched_elements=(a != b).sum().item(), elements=a.numel(),
                  max_abs=(a - b).abs().max().item(), recast_equal=torch.equal(a.bfloat16().float(), b))
    report["reproduced"] = report["bf16_equal"] and not report["fp32_equal"] and report["recast_equal"]
    torch.save(dict(eager_bf16=expected[before].cpu(), aoti_bf16=actual[before].cpu(),
                    eager_fp32=b.cpu(), aoti_fp32=a.cpu()), args.output / "boundary.pt")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["reproduced"]:
        raise SystemExit("Original signature did not reproduce; inspect the retained evidence")


if __name__ == "__main__":
    main()
