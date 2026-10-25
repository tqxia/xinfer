"""Check frozen projection packing, dynamic shapes and AOTI mutation behavior."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ["AOTI_RUNTIME_CHECK_INPUTS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from passes.projections import pack_projections
from validate_aoti import compare


class Projections(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.ModuleDict({
            name: torch.nn.Linear(k, n, bias=False)
            for name, k, n in (("q_proj",1024,2048), ("k_proj",1024,1024),
                               ("v_proj",1024,1024), ("o_proj",2048,1024))})
        layer.mlp = torch.nn.ModuleDict({
            name: torch.nn.Linear(k, n, bias=False)
            for name, k, n in (("gate_proj",1024,3072), ("up_proj",1024,3072),
                               ("down_proj",3072,1024))})
        self.layers = torch.nn.ModuleList([layer])

    def forward(self, x, cache, positions):
        layer = self.layers[0]
        q = layer.self_attn.q_proj(x)
        torch.ops.aten._assert_tensor_metadata.default(
            q, None, None, torch.bfloat16, device=x.device, layout=torch.strided)
        k = layer.self_attn.k_proj(x)
        v = layer.self_attn.v_proj(x)
        h = layer.self_attn.o_proj(q) + x
        gate = layer.mlp.gate_proj(h)
        up = layer.mlp.up_proj(h)
        out = layer.mlp.down_proj(torch.nn.functional.silu(gate) * up)
        # Deliberately unused result: dead-code elimination must keep the write.
        cache.index_copy_(1, positions, x)
        return q, k, v, h, gate, up, out


def inputs(tokens):
    return (torch.randn(1,tokens,1024,device="cuda",dtype=torch.bfloat16),
            torch.randn(1,1024,1024,device="cuda",dtype=torch.bfloat16),
            torch.arange(1024-tokens,1024,device="cuda"))


def clone_inputs(args):
    return tuple(x.clone() for x in args)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Projections().eval().to(device="cuda",dtype=torch.bfloat16)
    original = {name: value.clone() for name,value in model.state_dict().items()}
    report = dict(cases=[], rejections=[], strict_runtime_checks=True)
    for mode in ("qkv", "gate-up", "all"):
        for kind in ("decode", "prefill"):
            tokens = torch.export.Dim("tokens",min=2,max=512)
            shapes = None if kind == "decode" else ({1:tokens},{},{0:tokens})
            example = inputs(1 if kind == "decode" else 16)
            ep = torch.export.export(model,example,dynamic_shapes=shapes,strict=False)
            packed, stats = pack_projections(ep,1,mode)
            assert all(torch.equal(model.state_dict()[name],value) for name,value in original.items())
            for name,value in packed.named_buffers():
                if not name.startswith("packed_layer_"):
                    continue
                names = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj") if "qkv" in name else (
                    "mlp.gate_proj", "mlp.up_proj")
                expected = torch.cat([original[f"layers.0.{n}.weight"] for n in names])
                assert value.is_contiguous() and not value.requires_grad and torch.equal(value,expected)
            rewritten = torch.export.export(packed,example,dynamic_shapes=shapes,strict=False)
            rewritten.validate()
            targets = [n.target for n in rewritten.graph.nodes if n.op == "call_function"]
            assert torch.ops.aten.cat.default not in targets
            assert targets.count(torch.ops.aten.linear.default) == stats["linear_calls_after"]
            assert targets.count(torch.ops.aten.index_copy_.default) == 1
            assert all(name not in rewritten.state_dict for name in stats["removed_weight_attributes"])
            prefix = f"{mode}-{kind}"
            (args.output/(prefix+".graph.txt")).write_text(rewritten.graph_module.code)
            package = args.output/(prefix+".pt2")
            torch._inductor.aoti_compile_and_package(rewritten,package_path=str(package),
                                                     inductor_configs={"emulate_precision_casts":True})
            fn = torch._inductor.aoti_load_package(str(package))
            for length in ((1,) if kind == "decode" else (2,25,128,512)):
                actual_inputs = inputs(length)
                expected_inputs = clone_inputs(actual_inputs)
                packed_inputs = clone_inputs(actual_inputs)
                before_x = actual_inputs[0].clone()
                before_pos = actual_inputs[2].clone()
                pointer = actual_inputs[1].data_ptr()
                expected = model(*expected_inputs)
                packed_expected = packed(*packed_inputs)
                actual = fn(*actual_inputs)
                errors = []
                for a,b,c in zip(actual,expected,packed_expected):
                    errors.append(dict(unpacked=compare(a.flatten(1).float(),b.flatten(1).float()),
                                       packed_eager=compare(a.flatten(1).float(),c.flatten(1).float())))
                assert actual_inputs[1].data_ptr() == pointer
                assert torch.equal(actual_inputs[0],before_x) and torch.equal(actual_inputs[2],before_pos)
                assert torch.equal(actual_inputs[1],expected_inputs[1])
                assert torch.equal(actual_inputs[1],packed_inputs[1])
                report["cases"].append(dict(kind=kind,tokens=length,errors=errors,
                    cache_exact=True,stable_storage=True,source_weights_unchanged=True,**stats))
                print(prefix,length,"passed; linear calls",stats["linear_calls_after"],flush=True)
    for name in ("missing_projection", "different_input", "bias", "shared_weight", "wrong_layers", "side_effect"):
        ep = torch.export.export(model,inputs(1),strict=False)
        linear = [n for n in ep.graph.nodes if n.target == torch.ops.aten.linear.default]
        if name == "missing_projection":
            linear[1].target = torch.ops.aten.alias.default
            linear[1].args = (linear[1].args[0],)
        elif name == "different_input":
            linear[1].args = (linear[0], *linear[1].args[1:])
        elif name == "bias":
            linear[0].args = (*linear[0].args, linear[0].args[0])
        elif name == "shared_weight":
            linear[2].args = (linear[2].args[0], linear[1].args[1])
        elif name == "side_effect":
            cache = next(n for n in ep.graph.nodes if n.op == "placeholder" and n.target == "cache")
            with ep.graph.inserting_before(linear[1]):
                ep.graph.call_function(torch.ops.aten.zero_.default, (cache,))
        try:
            pack_projections(ep,2 if name == "wrong_layers" else 1,"all")
        except ValueError:
            report["rejections"].append(name)
        else:
            raise AssertionError(name)
    report["passed"] = True
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print("PASS: export-time packing, all modes, dynamic prefill, compiled cache writes",flush=True)


if __name__ == "__main__":
    main()
