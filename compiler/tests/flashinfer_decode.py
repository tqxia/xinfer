"""Validate the experimental native decode op, layouts, and AOTI mutation order."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ["AOTI_RUNTIME_CHECK_INPUTS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from custom_attention import AttentionStep, inputs
from custom_ops.flashinfer_decode import SOURCE, load_flashinfer_ops
from model import cache_extent
from passes.attention import rewrite_attention
from validate_aoti import compare


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    load_flashinfer_ops()
    op = torch.ops.xinfer.flashinfer_decode.default
    flash = torch.ops.xinfer.flash_attention.default
    scale = 128**-0.5
    lengths = (1, 2, 25, 26, 128, 129, 256, 257, 512, 513, 1024)
    report = dict(source=SOURCE, operator_cases=[], layouts=[], compiled_cases=[], rejections=[])
    for length in lengths:
        q, _, _, _, (k, v), extent = inputs(1, length)
        before = [x.clone() for x in (q, k, v)]
        actual = op(q, k, v, extent, scale)
        with sdpa_kernel(SDPBackend.MATH):
            expected = torch.nn.functional.scaled_dot_product_attention(
                q.float(), k[:, :, :length].float(), v[:, :, :length].float(), enable_gqa=True)
        metrics = compare(actual.flatten(1).float(), expected.flatten(1))
        flash_metrics = compare(actual.flatten(1).float(), flash(q, k, v, extent, scale).flatten(1).float())
        assert actual.stride() == (2048, 128, 128, 1)
        assert actual.data_ptr() not in [x.data_ptr() for x in (q, k, v)]
        assert all(torch.equal(a, b) for a, b in zip((q, k, v), before))
        k[:, :, length:].fill_(float("nan"))
        v[:, :, length:].fill_(float("nan"))
        extent.fill_(255)
        assert torch.equal(actual, op(q, k, v, extent, scale))
        # Keep NaNs out of opcheck's generic equality machinery.
        k[:, :, length:].zero_()
        v[:, :, length:].zero_()
        check = torch.library.opcheck(op, (q, k, v, extent, scale))
        report["operator_cases"].append(dict(length=length, fp32=metrics, flash=flash_metrics,
            opcheck=check, read_only=True, future_poison_exact=True, extent_data_ignored=True))
        print(f"L={length}: FP32 L2={metrics['max_relative_l2']:.6f}, opcheck passed", flush=True)
    # Exercise layouts the flexible_layout tag allows Inductor to hand us.
    for name in ("padded_heads", "feature_stride", "mismatched_kv", "unaligned_offset", "expanded_heads"):
        q, _, _, _, (k, v), extent = inputs(1, 513)
        if name == "padded_heads":
            k = torch.randn(1, 8, 1025, 128, device="cuda", dtype=torch.bfloat16)[:, :, :1024]
            v = torch.randn(1, 8, 1025, 128, device="cuda", dtype=torch.bfloat16)[:, :, :1024]
        elif name == "feature_stride":
            q = torch.randn(1, 16, 1, 256, device="cuda", dtype=torch.bfloat16)[..., ::2]
            k = torch.randn(1, 8, 1024, 256, device="cuda", dtype=torch.bfloat16)[..., ::2]
        elif name == "mismatched_kv":
            v = torch.randn(1, 1024, 8, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        elif name == "unaligned_offset":
            k = torch.randn(k.numel()+1, device="cuda", dtype=torch.bfloat16)[1:].view_as(k)
            q = torch.randn(q.numel()+1, device="cuda", dtype=torch.bfloat16)[1:].view_as(q)
        else:
            q = q[:, :1].expand_as(q)
            k = k[:, :1].expand_as(k)
            v = v[:, :1].expand_as(v)
        expected = op(*(x.clone(memory_format=torch.contiguous_format) for x in (q, k, v)), extent, scale)
        assert torch.equal(op(q, k, v, extent, scale), expected), name
        report["layouts"].append(dict(name=name, contiguous_reference_exact=True))
    # Ensure work is enqueued on the caller's stream, including local scratch.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        q, _, _, _, (k, v), extent = inputs(1, 1024)
        streamed = op(q, k, v, extent, scale)
    stream.synchronize()
    assert torch.equal(streamed, op(q, k, v, extent, scale))
    report["nondefault_stream_exact"] = True
    for name, q, extent in (("zero_length", q, cache_extent(0)),
                            ("overflow", q, cache_extent(1025)),
                            ("prefill", q.expand(1, 16, 2, 128), cache_extent(2))):
        try:
            op(q, k, v, extent, scale)
        except RuntimeError:
            report["rejections"].append(name)
        else:
            raise AssertionError(name)
    extent_dim = torch.export.Dim("extent", min=2, max=1025)
    shapes = ({}, {}, {}, {}, ({}, {}), {0: extent_dim})
    example = inputs(1, 32)
    ep = torch.export.export(AttentionStep(), example, dynamic_shapes=shapes, strict=False)
    gm, stats = rewrite_attention(ep, 1, "flashinfer-decode")
    rewritten = torch.export.export(gm, example, dynamic_shapes=shapes, strict=False)
    rewritten.validate()
    targets = [n.target for n in rewritten.graph.nodes if n.op == "call_function"]
    assert targets.count(op) == 1 and targets.count(torch.ops.aten.index_copy_.default) == 2
    assert torch.ops.aten.scaled_dot_product_attention.default not in targets
    (args.output/"decode.graph.txt").write_text(rewritten.graph_module.code)
    package = args.output/"decode.pt2"
    torch._inductor.aoti_compile_and_package(rewritten, package_path=str(package))
    fn = torch._inductor.aoti_load_package(str(package))
    for length in lengths:
        q, nk, nv, pos, kv, extent = inputs(1, length)
        expected_kv = [x.clone() for x in kv]
        expected_kv[0].index_copy_(2, pos, nk)
        expected_kv[1].index_copy_(2, pos, nv)
        expected = op(q, *expected_kv, extent, scale)
        pointers = [x.data_ptr() for x in kv]
        actual = fn(q, nk, nv, pos, kv, extent)
        assert torch.equal(actual, expected)
        assert pointers == [x.data_ptr() for x in kv]
        assert all(torch.equal(a, b) for a, b in zip(kv, expected_kv))
        report["compiled_cases"].append(dict(length=length, output_exact=True,
            full_cache_exact=True, stable_storage=True, **stats))
    for name in ("unknown_mask", "prefill"):
        ep = torch.export.export(AttentionStep(), inputs(2 if name == "prefill" else 1, 32), strict=False)
        if name == "unknown_mask":
            site = next(n for n in ep.graph.nodes if n.target == torch.ops.aten.scaled_dot_product_attention.default)
            site.args = (*site.args[:3], None)
        try:
            rewrite_attention(ep, 1, "flashinfer-decode")
        except ValueError:
            report["rejections"].append("rewrite_"+name)
        else:
            raise AssertionError(name)
    from torch.fx.experimental import _config as symbolic_config
    assert not symbolic_config.backed_size_oblivious
    report.update(passed=True, cache_padding=0, backed_size_oblivious=False)
    (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print("PASS: operator, layouts, nondefault stream, compiled writes and boundaries", flush=True)


if __name__ == "__main__":
    main()
