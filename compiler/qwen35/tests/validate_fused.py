"""Check fused preparation, carried states and Python-free C++ AOTI."""

import argparse
import importlib.metadata
import inspect
import os
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    causal_conv1d_fn, causal_conv1d_update, torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

from compiler.common.artifacts import ROOT, read_json, sha256, write_json
from compiler.common.validation import bits_equal, compare_file, cpu_copy, raw_bytes, row_errors
from compiler.qwen35.config import ARTIFACT_DIR, configure
from compiler.qwen35.diagnostics.block import MODEL_DIR, load_block
from compiler.qwen35.kernels.chunked import CHUNK_SIZE
from compiler.qwen35.kernels.recurrent import fused

LENGTHS = (1, 2, 3, 4, 5, 31, 32, 33, 63, 64, 65, 127, 128, 129, 511, 512)
PARTITIONS = (1, 2, 3, 4, 5, 31, 32, 33, 64, 65, 272)
# Predeclared normalized error limits; FP32 reference uses the same input values.
LIMITS = {"bf16": 0.01, "fp32": 0.0001}
# Nearly orthogonal Q/K can make a whole output row tiny. Retain normalized
# metrics, but permit FP32 absolute roundoff below 1e-7 (see experiment record).
ABS_LIMITS = {"bf16": 0.0, "fp32": 1e-7}


class Fused(torch.nn.Module):
    def forward(self, qkv, b, a, conv, state, weight, A_log, dt_bias):
        return fused(qkv, b, a, conv, state, weight, A_log, dt_bias)


def reference(qkv, b, a, conv, state, weight, A_log, dt_bias, *, chunked=False):
    """HF preparation with either recurrent or chunked prefill arithmetic."""
    tokens = qkv.shape[1]
    if tokens == 1:
        next_conv = conv.clone()
        mixed = causal_conv1d_update(qkv.transpose(1, 2), next_conv, weight, None, "silu")
    else:
        combined = torch.cat((conv, qkv.transpose(1, 2)), dim=-1)
        next_conv = combined[..., -4:].contiguous()
        mixed = causal_conv1d_fn(combined, weight, None, activation="silu")[..., -tokens:]
    q, k, v = (x.reshape(1, tokens, 16, 128) for x in mixed.transpose(1, 2).split(2048, dim=-1))
    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    implementation = torch_chunk_gated_delta_rule if chunked and tokens > 1 else torch_recurrent_gated_delta_rule
    options = dict(chunk_size=CHUNK_SIZE) if chunked and tokens > 1 else {}
    core, final = implementation(q, k, v, g, beta, initial_state=state,
                                 output_final_state=True, use_qk_l2norm_in_kernel=True, **options)
    return core, next_conv, final


def make_cases(block_dir):
    cases = []
    for dtype_name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        def case(name, lengths=(1,), nonzero=True):
            return dict(name=f"{dtype_name}-{name}", dtype=dtype_name,
                        steps=[(torch.randn(1, t, 6144).to(dtype),
                                torch.randn(1, t, 16).to(dtype),
                                torch.randn(1, t, 16).to(dtype)) for t in lengths],
                        conv=(torch.randn(1, 6144, 4) * 0.3).to(dtype) if nonzero else torch.zeros(1, 6144, 4, dtype=dtype),
                        state=torch.randn(1, 16, 128, 128) * 0.1 if nonzero else torch.zeros(1, 16, 128, 128),
                        weight=(torch.randn(6144, 4) * 0.2).to(dtype),
                        A_log=torch.full((16,), -2.0, dtype=dtype),
                        dt_bias=(torch.randn(16) * 0.1).to(dtype))

        for tokens in LENGTHS:
            for nonzero in (False, True):
                cases.append(case(f"t{tokens}-{'state' if nonzero else 'zero'}", (tokens,), nonzero))
        whole = case("whole", (512,))
        stream = [x.split(PARTITIONS, dim=1) for x in whole["steps"][0]]
        split = dict(whole, name=f"{dtype_name}-split", steps=list(zip(*stream)))
        cases.extend((whole, split, dict(split, name=f"{dtype_name}-split-repeat")))
        carry = case("carry", (128,) + (1,) * 64)
        cases.extend((carry, dict(carry, name=f"{dtype_name}-repeat")))
        gates = case("gate-extremes", (1,) * 8)
        for i, (_, b, a) in enumerate(gates["steps"]):
            b.copy_(torch.linspace(-80, 80, 16))
            a.fill_((-80, -20, -1, 0, 19, 20, 21, 80)[i])
        cases.append(gates)
        # Reuse whole's seeded data so adding this case leaves old fixtures unchanged.
        prefill_gates = dict(whole, name=f"{dtype_name}-prefill-gate-extremes",
                             steps=[tuple(x[:, start:end].clone() for x in whole["steps"][0])
                                    for start, end in ((0, 65), (65, 66))])
        extremes = torch.tensor([-80, -20, -1, 0, 19, 20, 21, 80], dtype=dtype)
        for _, b, a in prefill_gates["steps"]:
            b.copy_(torch.linspace(-80, 80, 16))
            a.copy_(extremes[torch.arange(a.shape[1]) % len(extremes)].view(1, -1, 1))
        cases.append(prefill_gates)
        zero = case("zero-qkv")
        zero["conv"].zero_()
        zero["steps"][0][0].zero_()
        cases.append(zero)
        # Cancellation and non-exact products expose missing convolution rounding.
        cancellation = case("conv-cancellation", (5,))
        cancellation["conv"].copy_(torch.tensor([0.99609375, -1.0078125, 0.50390625, -0.49609375]))
        cancellation["weight"].copy_(torch.tensor([0.333984375, 0.333984375, -0.66796875, -0.66796875]))
        cancellation["steps"][0][0].fill_(0.498046875)
        cases.append(cancellation)

    if block_dir.exists():
        block, _ = load_block(block_dir)
        conv = torch.randn(1, 6144, 4, device="cuda", dtype=torch.bfloat16) * 0.1
        state = torch.randn(1, 16, 128, 128, device="cuda") * 0.1
        steps = []
        for tokens in (1, 3, 4, 5, 32, 1):
            hidden = torch.randn(1, tokens, block.in_proj_qkv.in_features, device="cuda", dtype=torch.bfloat16)
            steps.append(cpu_copy((block.in_proj_qkv(hidden), block.in_proj_b(hidden), block.in_proj_a(hidden))))
        cases.append(dict(name="bf16-checkpoint", dtype="bf16", steps=steps,
                          conv=conv.cpu(), state=state.cpu(),
                          weight=block.conv1d.weight.squeeze(1).detach().cpu(),
                          A_log=block.A_log.detach().cpu(), dt_bias=block.dt_bias.detach().cpu()))
        del block
        torch.cuda.empty_cache()
    return cases


def write_cases(directory, cases):
    for case in cases:
        initial = (case[k] for k in ("conv", "state", "weight", "A_log", "dt_bias"))
        (directory / f"{case['name']}-initial.bin").write_bytes(b"".join(raw_bytes(x) for x in initial))
        for i, inputs in enumerate(case["steps"]):
            (directory / f"{case['name']}-{i}.bin").write_bytes(b"".join(raw_bytes(x) for x in inputs))
    write_json(directory / "cases.json", dict(cases=[dict(name=c["name"], dtype=c["dtype"],
                                                        lengths=[x[0].shape[1] for x in c["steps"]]) for c in cases]))


def run_cases(call, cases, fp32=False):
    results = {}
    for case in cases:
        convert = lambda x: x.to(device="cuda", dtype=torch.float32 if fp32 else x.dtype).contiguous()
        conv, state, weight, A_log, dt_bias = (convert(case[k]) for k in ("conv", "state", "weight", "A_log", "dt_bias"))
        for step, data in enumerate(case["steps"]):
            inputs = (*map(convert, data), conv, state, weight, A_log, dt_bias)
            before = [x.clone() for x in inputs]
            values = call(*inputs)
            assert len(values) == 3 and values[0].shape == (1, inputs[0].shape[1], 16, 128)
            assert values[0].dtype == inputs[0].dtype and values[1].dtype == conv.dtype
            assert values[1].shape == conv.shape and values[2].shape == state.shape
            assert values[2].dtype == torch.float32
            assert all(bits_equal(a, b) for a, b in zip(inputs, before)), "Input mutation"
            assert not any(torch._C._is_alias_of(x, y) for x in values for y in inputs), "Output aliases input"
            results[case["name"], step] = cpu_copy(values)
            conv, state = values[1:]
    return results


def compare(report, label, actual, expected, exact_tensors=()):
    for key, values in actual.items():
        dtype = key[0].split("-")[0]
        for i, (a, b) in enumerate(zip(values, expected[key])):
            width = (128, 4, 128 * 128)[i]
            error = row_errors(a.float().reshape(-1, width), b.float().reshape(-1, width))
            row = dict(comparison=label, case=key[0], step=key[1],
                       tensor=("core", "conv", "recurrent")[i], bitwise=bits_equal(a, b), **error)
            report["comparisons"].append(row)
            failed = max(error["relative_l2"], error["peak_relative"]) > LIMITS[dtype] and error["max_abs"] > ABS_LIMITS[dtype]
            if failed or (i in exact_tensors and not row["bitwise"]) or (i == 1 and not torch.equal(a.float(), b.float())):
                report["failures"].append(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("work/qwen35/build/qwen35_fused"))
    parser.add_argument("--block-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "fused-validation")
    args = parser.parse_args()
    configure()
    torch.manual_seed(35)
    args.output.mkdir(parents=True, exist_ok=False)
    with torch.inference_mode():
        cases = make_cases(args.block_dir)
        write_cases(args.output, cases)
        report = dict(seed=35, torch=torch.__version__, triton=importlib.metadata.version("triton"),
                      chunk_size=CHUNK_SIZE,
                      partition_acceptance=dict(core="numerical", conv="bitwise", recurrent="numerical"),
                      limits=LIMITS, absolute_limits=ABS_LIMITS, comparisons=[], failures=[], packages=[],
                      checkpoint_case=any(c["name"] == "bf16-checkpoint" for c in cases),
                      binary_sha256=sha256(args.binary),
                      hf_source_sha256=sha256(inspect.getfile(causal_conv1d_update)),
                      source_sha256={name: sha256(ROOT / name) for name in (
                          "compiler/qwen35/tests/validate_fused.py", "compiler/qwen35/kernels/recurrent.py",
                          "compiler/qwen35/kernels/chunked.py",
                          "runtime/qwen35/tests/fused.cpp", "runtime/qwen35/CMakeLists.txt")})
        packages = {}
        destination = args.output / "packages"
        destination.mkdir()
        for dtype in ("bf16", "fp32"):
            for mode, tokens in (("decode", 1), ("prefill", 32)):
                case = next(c for c in cases if c["name"] == f"{dtype}-t{tokens}-state")
                sample = tuple(x.cuda().contiguous() for x in (*case["steps"][0], *(case[k] for k in ("conv", "state", "weight", "A_log", "dt_bias"))))
                ops = [torch.ops.xinfer.delta_cond_short.default]
                if mode == "prefill":
                    ops.append(torch.ops.xinfer.delta_cond_long.default)
                for op in ops:
                    report.setdefault("opcheck", {})[f"{dtype}-{mode}-{op}"] = torch.library.opcheck(
                        op, sample, test_utils=("test_schema", "test_faketensor"))
                dynamic = None
                if mode == "prefill":
                    length = torch.export.Dim("tokens", min=2, max=512)
                    dynamic = tuple([{1: length}] * 3 + [None] * 5)
                exported = torch.export.export(Fused(), sample, dynamic_shapes=dynamic, strict=True)
                exported.validate()
                assert not exported.graph_signature.user_inputs_to_mutate
                nodes = [n for module in exported.graph_module.modules()
                         if isinstance(module, torch.fx.GraphModule) for n in module.graph.nodes]
                assert sum(n.target == torch.ops.higher_order.cond for n in nodes) == int(mode == "prefill")
                assert sum(n.target == torch.ops.xinfer.delta_cond_short.default for n in nodes) == 1
                assert sum(n.target == torch.ops.xinfer.delta_cond_long.default for n in nodes) == int(mode == "prefill")
                path = destination / f"{dtype}-{mode}.pt2"
                torch._inductor.aoti_compile_and_package(exported, package_path=str(path),
                                                        inductor_configs={"emulate_precision_casts": True})
                packages[dtype, mode] = torch._inductor.aoti_load_package(str(path))
                report["packages"].append(dict(dtype=dtype, mode=mode, sha256=sha256(path),
                                               range_constraints=str(exported.range_constraints)))

        results = {}
        calls = dict(fused=fused, aoti=lambda *x: packages["bf16" if x[0].dtype == torch.bfloat16 else "fp32",
                                                        "decode" if x[0].shape[1] == 1 else "prefill"](*x),
                     hf=reference, hf_chunked=lambda *x: reference(*x, chunked=True), fp32=reference)
        for name, call in calls.items():
            print(f"Running {name}: {len(cases)} cases", flush=True)
            results[name] = run_cases(call, cases, fp32=name == "fp32")
        for dtype in ("bf16", "fp32"):
            for engine, values in results.items():
                for original, repeat in (("carry", "repeat"), ("split", "split-repeat")):
                    length = next(len(c["steps"]) for c in cases if c["name"] == f"{dtype}-{original}")
                    for step in range(length):
                        assert all(bits_equal(a, b) for a, b in zip(values[f"{dtype}-{original}", step], values[f"{dtype}-{repeat}", step]))
        expected = cpu_copy(fused(*sample))
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            streamed = fused(*(x.clone() for x in sample))
        stream.synchronize()
        assert all(bits_equal(a, b) for a, b in zip(cpu_copy(streamed), expected))
        report["nondefault_stream"] = True
        strided = list(sample)
        backing = torch.empty((*sample[0].shape[:-1], sample[0].shape[-1] * 2), device="cuda", dtype=sample[0].dtype)
        backing[..., ::2].copy_(sample[0])
        strided[0] = backing[..., ::2]
        strided[5] = sample[5].t().contiguous().t()
        assert not strided[0].is_contiguous() and not strided[5].is_contiguous()
        assert all(bits_equal(a, b) for a, b in zip(cpu_copy(fused(*strided)), expected))
        report["noncontiguous_inputs"] = True

        bad = dict(empty=[*(x[:, :0] for x in sample[:3]), *sample[3:]],
                   too_long=[*(x.repeat(1, 17, 1)[:, :513] for x in sample[:3]), *sample[3:]],
                   gates=[sample[0], sample[1][..., :1], *sample[2:]],
                   conv_width=[*sample[:3], sample[3][..., :3], *sample[4:]],
                   state_dtype=[*sample[:4], sample[4].bfloat16(), *sample[5:]],
                   weight_dtype=[*sample[:5], sample[5].half(), *sample[6:]],
                   cpu=[x.cpu() for x in sample])
        report["rejections"] = []
        for name, inputs in bad.items():
            try:
                fused(*inputs)
            except (ValueError, RuntimeError):
                report["rejections"].append(name)
            else:
                raise AssertionError(f"Accepted invalid {name}")

    pairs = (("fused", "aoti"), ("fused", "hf"), ("fused", "hf_chunked"),
             ("fused", "fp32"), ("hf", "fp32"))
    for actual_engine, expected_engine in pairs:
        compare(report, f"{actual_engine}_vs_{expected_engine}", results[actual_engine],
                results[expected_engine], exact_tensors=(0, 1, 2) if expected_engine == "aoti" else ())
    for engine, values in results.items():
        for dtype in ("bf16", "fp32"):
            parts = [values[f"{dtype}-split", i] for i in range(len(PARTITIONS))]
            key = (f"{dtype}-partition", 0)
            combined = {key: (torch.cat([p[0] for p in parts], dim=1), *parts[-1][1:])}
            # Chunking reorders FP32 core/state arithmetic across partitions.
            # Keep the existing numerical limits and require exact conv history.
            compare(report, f"{engine}_partition", combined, {key: values[f"{dtype}-whole", 0]},
                    exact_tensors=(1,))
    env = {k: os.environ[k] for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH") if k in os.environ}
    env.update(PATH="/usr/bin:/bin", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHOME="/nonexistent-python-home")
    output = args.output / "cpp-aoti"
    child = subprocess.run([str(args.binary.resolve()), str(args.output.resolve()), str(output.resolve()),
                            str(destination.resolve())], env=env, text=True, capture_output=True, timeout=1200)
    (args.output / "cpp-aoti.log").write_text(child.stdout + child.stderr)
    child.check_returncode()
    for (name, step), tensors in results["aoti"].items():
        compare_file(output / f"{name}-{step}.bin", tensors)
    report["cpp"] = dict(**read_json(output / "native.json"), bitwise_tensors=3 * len(results["aoti"]))
    report["checks"] = dict(inputs_unchanged=True, outputs_do_not_alias_inputs=True, reset_repeat_bitwise=True,
                           finite_shapes_dtypes=True,
                           convolution_state_exact=all(row["max_abs"] == 0 for row in report["comparisons"]
                                                       if row["tensor"] == "conv"))
    # Separate the fusion regression from the pre-existing BF16-to-FP32 gap.
    fusion_comparisons = {"fused_vs_aoti", "fused_vs_hf", "fused_vs_hf_chunked",
                          "fused_partition", "aoti_partition"}
    report["fusion_passed"] = not any(
        row["comparison"] in fusion_comparisons or
        (row["comparison"] == "fused_vs_fp32" and row["case"].startswith("fp32-"))
        for row in report["failures"])
    report["fp32_failure_counts"] = {
        engine: sum(row["comparison"] == f"{engine}_vs_fp32" for row in report["failures"])
        for engine in ("fused", "hf")}
    report["fp32_acceptance"] = not any(report["fp32_failure_counts"].values())
    report["passed"] = not report["failures"]
    write_json(args.output / "summary.json", report)
    print({key: report[key] for key in ("passed", "fusion_passed", "fp32_acceptance",
                                      "fp32_failure_counts", "cpp", "checkpoint_case")})
    print(f"Numerical failures: {len(report['failures'])}; report: {args.output / 'summary.json'}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
