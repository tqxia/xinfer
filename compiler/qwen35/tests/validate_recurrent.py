"""Validate the recurrent Triton baseline, carried state and Python-free C++/AOTI."""

import argparse
import importlib.metadata
import inspect
import os
from pathlib import Path
import subprocess

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule

from compiler.common.artifacts import ROOT, read_json, sha256, write_json
from compiler.common.validation import bits_equal, compare_file, cpu_copy, raw_bytes, row_errors
from compiler.qwen35.config import ARTIFACT_DIR, configure
from compiler.qwen35.kernels.recurrent import recurrent, run as triton_run

LENGTHS = (1, 2, 3, 22, 31, 32, 33, 63, 64, 65, 127, 128, 129, 511, 512)
# Predeclared normalized error limits; FP32 reference uses the same input values.
LIMITS = {"bf16": 0.01, "fp32": 0.0001}
# Nearly orthogonal Q/K can make a whole output row tiny. Retain normalized
# metrics, but permit FP32 absolute roundoff below 1e-7 (see experiment record).
ABS_LIMITS = {"bf16": 0.0, "fp32": 1e-7}


class Recurrent(torch.nn.Module):
    def forward(self, q, k, v, g, beta, state):
        return recurrent(q, k, v, g, beta, state)


def make_cases(directory):
    cases = []
    for dtype_name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        def inputs(tokens):
            qkv = [torch.randn(1, tokens, 16, 128).to(dtype) for _ in range(3)]
            return (*qkv, -torch.rand(1, tokens, 16) * 0.1,
                    torch.rand(1, tokens, 16).to(dtype))

        state = torch.randn(1, 16, 128, 128) * 0.1
        for tokens in LENGTHS:
            data = inputs(tokens)
            for nonzero in (False, True):
                name = f"{dtype_name}-t{tokens}-{'state' if nonzero else 'zero'}"
                cases.append((name, dtype_name, [data], state if nonzero else torch.zeros_like(state)))
        stream = inputs(512)
        for name, lengths in (("whole", [512]), ("split", [129] + [1] * 32 + [351]),
                              ("repeat", [129] + [1] * 32 + [351])):
            parts = [x.split(lengths, dim=1) for x in stream]
            cases.append((f"{dtype_name}-{name}", dtype_name, list(zip(*parts)), state))
        for name in ("identity", "forget", "zero_qk"):
            data = list(inputs(65))
            if name == "identity":
                data[3].zero_()
                data[4].zero_()
            elif name == "forget":
                data[3].fill_(-80)
                data[4].fill_(1)
            else:
                data[0].zero_()
                data[1].zero_()
            cases.append((f"{dtype_name}-{name}", dtype_name, [tuple(data)], state))
    manifest = []
    for name, dtype, steps, state in cases:
        (directory / f"{name}-state.bin").write_bytes(raw_bytes(state))
        for i, data in enumerate(steps):
            (directory / f"{name}-{i}.bin").write_bytes(b"".join(raw_bytes(x.contiguous()) for x in data))
        manifest.append(dict(name=name, dtype=dtype, lengths=[x[0].shape[1] for x in steps]))
    write_json(directory / "cases.json", dict(cases=manifest))
    return cases


def run_cases(call, cases, fp32=False):
    results = {}
    for name, _, steps, initial in cases:
        state = initial.cuda()
        for step, data in enumerate(steps):
            inputs = tuple(x.to("cuda", torch.float32 if fp32 else x.dtype).contiguous() for x in data) + (state,)
            before = [x.clone() for x in inputs]
            output, final = call(*inputs)
            assert all(bits_equal(a, b) for a, b in zip(inputs, before)), "Input mutation"
            assert output.shape == inputs[2].shape and output.dtype == inputs[2].dtype
            assert final.shape == state.shape and final.dtype == torch.float32
            assert not any(torch._C._is_alias_of(x, y) for x in (output, final) for y in inputs)
            results[name, step] = cpu_copy((output, final))
            state = final
    return results


def hf(function):
    def call(q, k, v, g, beta, state):
        return function(q, k, v, g, beta, initial_state=state,
                        output_final_state=True, use_qk_l2norm_in_kernel=True)
    return call


def export_packages(directory, cases):
    directory.mkdir()
    packages, rows = {}, []
    for dtype in ("bf16", "fp32"):
        for mode, tokens in (("decode", 1), ("prefill", 32)):
            case = next(c for c in cases if c[0] == f"{dtype}-t{tokens}-state")
            inputs = tuple(x.cuda().contiguous() for x in (*case[2][0], case[3]))
            dynamic = None
            if mode == "prefill":
                length = torch.export.Dim("tokens", min=2, max=512)
                dynamic = tuple([{1: length}] * 5 + [None])
            exported = torch.export.export(Recurrent(), inputs, dynamic_shapes=dynamic, strict=True)
            exported.validate()
            assert not exported.graph_signature.user_inputs_to_mutate
            ops = [n for n in exported.graph.nodes if n.target == torch.ops.xinfer.delta_recurrent.default]
            assert len(ops) == 1
            path = directory / f"{dtype}-{mode}.pt2"
            torch._inductor.aoti_compile_and_package(exported, package_path=str(path))
            packages[dtype, mode] = torch._inductor.aoti_load_package(str(path))
            rows.append(dict(dtype=dtype, mode=mode, sha256=sha256(path),
                             range_constraints=str(exported.range_constraints), custom_ops=len(ops)))
    return packages, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("work/qwen35/build/qwen35_recurrent"))
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "recurrent-validation")
    args = parser.parse_args()
    configure()
    torch.manual_seed(35)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(seed=35, torch=torch.__version__, triton=importlib.metadata.version("triton"),
                  limits=LIMITS, absolute_limits=ABS_LIMITS, comparisons=[], failures=[],
                  binary_sha256=sha256(args.binary),
                  hf_source_sha256=sha256(inspect.getfile(torch_recurrent_gated_delta_rule)),
                  source_sha256={name: sha256(ROOT / name) for name in (
                      "compiler/qwen35/tests/validate_recurrent.py", "compiler/qwen35/kernels/recurrent.py",
                      "runtime/qwen35/tests/recurrent.cpp", "runtime/qwen35/CMakeLists.txt")})
    cases = make_cases(args.output)
    op = torch.ops.xinfer.delta_recurrent
    with torch.inference_mode():
        sample = tuple(x.cuda().contiguous() for x in (*cases[3][2][0], cases[3][3]))
        report["opcheck"] = torch.library.opcheck(op.default, sample,
            test_utils=("test_schema", "test_faketensor"))
        packages, report["packages"] = export_packages(args.output / "packages", cases)
        def aoti(*inputs):
            dtype = "bf16" if inputs[0].dtype == torch.bfloat16 else "fp32"
            return packages[dtype, "decode" if inputs[0].shape[1] == 1 else "prefill"](*inputs)
        results = {}
        for name, call, fp32 in (("triton", triton_run, False), ("triton_op", recurrent, False),
                                 ("aoti", aoti, False), ("hf", hf(torch_recurrent_gated_delta_rule), False),
                                 ("fp32", hf(torch_recurrent_gated_delta_rule), True)):
            print(f"Running {name}: {len(cases)} cases", flush=True)
            results[name] = run_cases(call, cases, fp32)

        # Wrapped Triton launches must respect PyTorch's current stream.
        expected = cpu_copy(op(*sample))
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            streamed = op(*(x.clone() for x in sample))
        stream.synchronize()
        assert all(bits_equal(a, b) for a, b in zip(cpu_copy(streamed), expected))
        report["nondefault_stream"] = True

    def compare(label, actual, expected, exact=False):
        for (name, step), values in actual.items():
            dtype = name.split("-")[0]
            for i, (a, b) in enumerate(zip(values, expected[name, step])):
                width = 128 if i == 0 else 128 * 128
                error = row_errors(a.float().reshape(-1, width), b.float().reshape(-1, width))
                row = dict(comparison=label, case=name, step=step, tensor=("output", "state")[i],
                           bitwise=bits_equal(a, b), **error)
                report["comparisons"].append(row)
                numerical_failure = (max(error["relative_l2"], error["peak_relative"]) > LIMITS[dtype]
                                     and error["max_abs"] > ABS_LIMITS[dtype])
                if (exact and not row["bitwise"]) or numerical_failure:
                    report["failures"].append(row)
    for name in ("triton", "aoti", "hf", "fp32"):
        compare(f"triton_op_vs_{name}", results["triton_op"], results[name], name in ("triton", "aoti"))
    for engine, values in results.items():
        for dtype in ("bf16", "fp32"):
            combined = {}
            for suffix in ("whole", "split", "repeat"):
                name = f"{dtype}-{suffix}"
                steps = [x for (case, _), x in values.items() if case == name]
                combined[name, 0] = (torch.cat([x[0] for x in steps], dim=1), steps[-1][1])
            compare(f"{engine}_partition", combined,
                    {key: combined[f"{dtype}-whole", 0] for key in combined}, engine in ("triton", "triton_op", "aoti"))
            assert all(bits_equal(a, b) for a, b in zip(combined[f"{dtype}-split", 0], combined[f"{dtype}-repeat", 0]))
        for name, _, _, initial in cases:
            if name.endswith("identity"):
                assert bits_equal(values[name, 0][1], initial)
            if name.endswith("zero_qk"):
                assert torch.count_nonzero(values[name, 0][0]) == 0

    # Contract rejection must happen before launching a kernel.
    bad = {"empty": [x[:, :0].contiguous() if i < 5 else x for i, x in enumerate(sample)],
           "too_long": [x.repeat(1, 513, *([1] * (x.ndim - 2)))[:, :513].contiguous() if i < 5 else x
                        for i, x in enumerate(sample)],
           "state_dtype": [*sample[:5], sample[5].bfloat16()],
           "gate_shape": [*sample[:3], sample[3][..., :1].contiguous(), *sample[4:]],
           "cpu": [x.cpu() for x in sample]}
    report["rejections"] = []
    for name, inputs in bad.items():
        try:
            op(*inputs)
        except (ValueError, RuntimeError):
            report["rejections"].append(name)
        else:
            raise AssertionError(f"Accepted invalid {name}")
    # The wrapper accepts noncontiguous tensors and establishes the kernel layout.
    strided = [sample[0].transpose(2, 3).contiguous().transpose(2, 3), *sample[1:]]
    assert not strided[0].is_contiguous()
    assert all(bits_equal(a, b) for a, b in zip(cpu_copy(op(*strided)), expected))
    report["noncontiguous_inputs"] = True
    env = {k: os.environ[k] for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH") if k in os.environ}
    env.update(PATH="/usr/bin:/bin", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHOME="/nonexistent-python-home")
    destination = args.output / "cpp-aoti"
    command = [str(args.binary.resolve()), str(args.output.resolve()), str(destination.resolve()),
               str((args.output / "packages").resolve())]
    print("Running standalone C++ AOTI", flush=True)
    child = subprocess.run(command, env=env, text=True, capture_output=True, timeout=1200)
    (args.output / "cpp-aoti.log").write_text(child.stdout + child.stderr)
    child.check_returncode()
    for (name, step), tensors in results["aoti"].items():
        compare_file(destination / f"{name}-{step}.bin", tensors)
    report["cpp"] = dict(**read_json(destination / "native.json"),
                         bitwise_tensors=2 * len(results["aoti"]))
    report["passed"] = not report["failures"]
    write_json(args.output / "summary.json", report)
    print({key: report[key] for key in ("passed", "cpp")})
    print(f"Numerical failures: {len(report['failures'])}; report: {args.output / 'summary.json'}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
