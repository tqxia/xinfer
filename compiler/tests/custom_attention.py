"""Validate the read-only native op and post-export rewrite on GB10."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ["AOTI_RUNTIME_CHECK_INPUTS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from custom_ops.attention import load_attention_ops
from passes.attention import rewrite_attention
from model import cache_extent
from validate_aoti import compare


class AttentionStep(torch.nn.Module):
    def forward(self, q, new_k, new_v, cache_position, kv, kv_extent):
        k, v = kv
        k.index_copy_(2, cache_position, new_k)
        v.index_copy_(2, cache_position, new_v)
        future = torch.arange(1024, device=q.device).unsqueeze(0) > cache_position.unsqueeze(1)
        mask = torch.zeros_like(future, dtype=torch.bfloat16)
        mask = mask.masked_fill(future, torch.finfo(torch.bfloat16).min)[None, None]
        k = k.unsqueeze(2).expand(1, 8, 2, 1024, 128).reshape(1, 16, 1024, 128)
        v = v.unsqueeze(2).expand(1, 8, 2, 1024, 128).reshape(1, 16, 1024, 128)
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, mask, scale=128**-0.5)


def cache():
    return tuple(torch.randn(1, 8, 1024, 128, device="cuda", dtype=torch.bfloat16)
                 for _ in range(2))


def inputs(tokens, length):
    return (torch.randn(1, 16, tokens, 128, device="cuda", dtype=torch.bfloat16),
            torch.randn(1, 8, tokens, 128, device="cuda", dtype=torch.bfloat16),
            torch.randn(1, 8, tokens, 128, device="cuda", dtype=torch.bfloat16),
            torch.arange(length-tokens, length, device="cuda"), cache(), cache_extent(length))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    load_attention_ops()
    op = torch.ops.xinfer.flash_attention.default
    from torch.fx.experimental import _config as symbolic_config
    assert not symbolic_config.backed_size_oblivious, "Test the default size-one specialization behavior"
    report = {"operator_cases": [], "compiled_cases": [], "rejections": [],
              "strict_runtime_checks": True, "cache_padding": 0, "backed_size_oblivious": False}
    for tokens, length in ((1,1), (1,2), (1,32), (1,1024), (2,2), (2,3), (16,32), (512,512), (510,1022)):
        q, _, _, _, (k,v), extent = inputs(tokens, length)
        before = [t.clone() for t in (q,k,v)]
        pointers = [t.data_ptr() for t in (q,k,v)]
        actual = op(q,k,v,extent,128**-0.5)
        reference = torch.ops.aten._scaled_dot_product_flash_attention.default(
            q,k[:,:,:length],v[:,:,:length],0.,True,scale=128**-0.5)[0]
        assert torch.equal(actual,reference)
        assert actual.stride() == (16*tokens*128,tokens*128,128,1)
        assert actual.data_ptr() not in pointers
        assert all(torch.equal(a,b) for a,b in zip((q,k,v),before))
        positions = torch.arange(tokens,device="cuda") + length-tokens
        mask = torch.arange(length,device="cuda")[None,:] <= positions[:,None]
        with sdpa_kernel(SDPBackend.MATH):
            expected = torch.nn.functional.scaled_dot_product_attention(
                q.float(),k[:,:,:length].float(),v[:,:,:length].float(),attn_mask=mask,enable_gqa=True)
        metrics = compare(actual.flatten(1).float(),expected.flatten(1))
        k[:,:,length:].fill_(100)
        v[:,:,length:].fill_(-100)
        extent.fill_(255)
        assert torch.equal(actual,op(q,k,v,extent,128**-0.5))
        check = torch.library.opcheck(op,(q,k,v,extent,128**-0.5))
        report["operator_cases"].append(dict(tokens=tokens,valid_keys=length,read_only=True,
            native_flash_exact=True,future_poison_exact=True,extent_data_ignored=True,opcheck=check,**metrics))
        print(f"Operator T={tokens} L={length}: exact native Flash, read-only, opcheck passed",flush=True)
    # The layout tag promises support beyond the usual unit feature stride.
    q = torch.randn(1,16,2,256,device="cuda",dtype=torch.bfloat16)[...,::2]
    k = torch.randn(1,8,1024,256,device="cuda",dtype=torch.bfloat16)[...,::2]
    v = torch.randn_like(k)
    expected = torch.ops.aten._scaled_dot_product_flash_attention.default(
        q.contiguous(),k[:,:,:3].contiguous(),v[:,:,:3].contiguous(),0.,True,scale=128**-0.5)[0]
    assert torch.equal(op(q,k,v,cache_extent(3),128**-0.5),expected)
    report["strided_inputs_opcheck"] = torch.library.opcheck(op,(q,k,v,cache_extent(3),128**-0.5))
    q, _, _, _, (k,v), _ = inputs(2,2)
    for name,extent in (("zero_length",cache_extent(0)),("shorter_than_queries",cache_extent(1)),
                        ("overflow",cache_extent(1025))):
        try:
            op(q,k,v,extent,128**-0.5)
        except RuntimeError as exc:
            assert "Invalid valid-prefix length" in str(exc),str(exc)
            report["rejections"].append(name)
        else:
            raise AssertionError(name)
    for kind in ("decode","prefill"):
        t = torch.export.Dim("tokens", min=2, max=512)
        e = torch.export.Dim("extent", min=2, max=1025)
        shapes = ({}, {}, {}, {}, ({},{}), {0:e}) if kind == "decode" else (
            {2:t},{2:t},{2:t},{0:t},({},{}),{0:e})
        example = inputs(1 if kind=="decode" else 16,32)
        assert all(kv.is_contiguous() and kv.untyped_storage().nbytes() == kv.numel()*kv.element_size()
                   for kv in example[4])
        ep = torch.export.export(AttentionStep(),example,dynamic_shapes=shapes,strict=False)
        # A non-causal/unknown mask must fail closed rather than silently disappear.
        bad = torch.export.export(AttentionStep(),example,dynamic_shapes=shapes,strict=False)
        site = next(n for n in bad.graph.nodes if n.target == torch.ops.aten.scaled_dot_product_attention.default)
        site.args = (*site.args[:3],None)
        try:
            rewrite_attention(bad,1)
        except ValueError:
            report["rejections"].append(kind+"_unknown_mask")
        else:
            raise AssertionError("Unknown mask accepted")
        gm, stats = rewrite_attention(ep,1)
        rewritten = torch.export.export(gm,example,dynamic_shapes=shapes,strict=False)
        rewritten.validate()
        targets = [n.target for n in rewritten.graph.nodes if n.op=="call_function"]
        assert targets.count(op)==1 and torch.ops.aten.scaled_dot_product_attention.default not in targets
        assert torch.ops.aten._scaled_dot_product_flash_attention.default not in targets
        (args.output/f"{kind}.graph.txt").write_text(rewritten.graph_module.code)
        package = args.output/f"{kind}.pt2"
        torch._inductor.aoti_compile_and_package(rewritten,package_path=str(package))
        fn = torch._inductor.aoti_load_package(str(package))
        cases = ((1,1),(1,2),(1,1024)) if kind=="decode" else (
            (2,2),(2,3),(16,32),(510,1022),(512,512),(512,1024))
        for tokens,length in cases:
            q,nk,nv,pos,kv,extent=inputs(tokens,length)
            expected_kv=[x.clone() for x in kv]
            expected_kv[0].index_copy_(2,pos,nk)
            expected_kv[1].index_copy_(2,pos,nv)
            expected=op(q,*expected_kv,extent,128**-0.5)
            pointers=[x.data_ptr() for x in kv]
            actual=fn(q,nk,nv,pos,kv,extent)
            assert torch.equal(actual,expected)
            assert pointers==[x.data_ptr() for x in kv]
            assert all(torch.equal(a,b) for a,b in zip(kv,expected_kv))
            report["compiled_cases"].append(dict(kind=kind,tokens=tokens,valid_keys=length,
                output_exact=True,full_cache_exact=True,stable_storage=True,**stats))
            print(f"Compiled {kind} T={tokens} L={length}: output and full caches exact",flush=True)
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
