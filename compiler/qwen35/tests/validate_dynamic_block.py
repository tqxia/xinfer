"""Validate one dynamic DeltaNet package against HF, FP32 and native C++."""

import argparse
import json
import os
from pathlib import Path
import subprocess

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

from compiler.common.artifacts import ROOT, sha256
from compiler.common.validation import bits_equal, compare_file, cpu_copy, raw_bytes
from compiler.qwen35.adapters.native_delta import NativeDeltaPrefill, load_library
from compiler.qwen35.config import ARTIFACT_DIR, configure
from compiler.qwen35.diagnostics.block import BlockStep, MODEL_DIR, input_shapes, load_block
from compiler.qwen35.tests.common import LIMIT, metrics

LENGTHS = (2, 3, 22, 31, 32, 33, 63, 64, 65, 127, 128, 129, 511, 512)
NAMES = ("output", "conv", "recurrent")


def check_operator():
    """Separate errors in the C++ translation from errors introduced by compilation."""
    rows = []
    for dtype in (torch.bfloat16, torch.float32):
        for tokens in (2, 65, 129):
            q, k, v = (torch.randn(1, tokens, 2, 16, device="cuda", dtype=dtype) for _ in range(3))
            g = -torch.rand(1, tokens, 2, device="cuda")
            beta = torch.rand_like(g).to(dtype)
            state = torch.randn(1, 2, 16, 16, device="cuda")
            inputs = (q, k, v, g, beta, state)
            expected = torch_chunk_gated_delta_rule(q, k, v, g, beta, initial_state=state,
                output_final_state=True, use_qk_l2norm_in_kernel=True)
            actual = torch.ops.xinfer.delta_prefill(*inputs)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
            rows.append(dict(dtype=str(dtype), tokens=tokens,
                             bitwise=[bits_equal(a, b) for a, b in zip(actual, expected)]))
    checks = torch.library.opcheck(torch.ops.xinfer.delta_prefill.default, inputs,
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"))
    return dict(comparisons=rows, opcheck=checks)


def make_fixtures(config, directory):
    """Zero/nonzero states, chunk boundaries, partition equivalence and reset/repeat."""
    _, conv_shape, state_shape = input_shapes(config, 2)
    random_state = (torch.randn(conv_shape, dtype=torch.bfloat16) * 0.1,
                    torch.randn(state_shape) * 0.1)
    cases = []
    for tokens in LENGTHS:
        for nonzero in (False, True):
            state = random_state if nonzero else tuple(torch.zeros_like(x) for x in random_state)
            hidden = torch.randn(input_shapes(config, tokens)[0], dtype=torch.bfloat16)
            cases.append((f"t{tokens}-{'state' if nonzero else 'zero'}", [hidden], state))
    stream = torch.randn(input_shapes(config, 129)[0], dtype=torch.bfloat16)
    for name, lengths in (("whole", (129,)), ("split", (64, 65)),
                          ("uneven", (2, 63, 64)), ("repeat", (2, 63, 64))):
        cases.append((name, list(stream.split(lengths, dim=1)), random_state))
    manifest = []
    for name, steps, state in cases:
        (directory / f"{name}-state.bin").write_bytes(b"".join(raw_bytes(t) for t in state))
        for i, hidden in enumerate(steps):
            (directory / f"{name}-{i}.bin").write_bytes(raw_bytes(hidden))
        manifest.append(dict(name=name, lengths=[t.shape[1] for t in steps]))
    (directory / "cases.json").write_text(json.dumps(dict(cases=manifest), indent=2) + "\n")
    return cases


def run_cases(call, cases, dtype=torch.bfloat16):
    results = {}
    for name, steps, initial in cases:
        conv, state = (initial[0].to("cuda", dtype), initial[1].to("cuda"))
        for step, hidden in enumerate(steps):
            inputs = (hidden.to("cuda", dtype), conv, state)
            before = tuple(t.clone() for t in inputs)
            output = call(*inputs)
            assert all(bits_equal(a, b) for a, b in zip(inputs, before)), "Input mutation"
            assert all(a.shape == b.shape and a.dtype == b.dtype for a, b in zip(output, inputs))
            results[name, step] = cpu_copy(output)
            _, conv, state = output
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--packages", type=Path, default=ARTIFACT_DIR / "dynamic-block")
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "dynamic-block-validation")
    args = parser.parse_args()
    configure()
    load_library(args.library)
    metadata = json.loads((args.packages / "prefill.json").read_text())
    assert sha256(args.library) == metadata["library_sha256"]
    for name, expected in metadata["source_sha256"].items():
        assert sha256(ROOT / name) == expected, f"Package source changed: {name}"
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(35)
    report = dict(limit=LIMIT, comparisons=[], failures=[], metadata=metadata,
                  seed=35, fixture="Unit-scale Gaussian hidden inputs; zero or Gaussian states (scale 0.1)",
                  lengths=LENGTHS, binary_sha256=sha256(args.binary),
                  package_sha256=sha256(args.packages / "prefill.pt2"),
                  source_sha256={name: sha256(ROOT / name) for name in
                      ("compiler/qwen35/tests/validate_dynamic_block.py", "runtime/qwen35/tests/dynamic_block.cpp")})
    with torch.inference_mode():
        report["operator"] = check_operator()
        block, config = load_block(args.model_dir)
        cases = make_fixtures(config, args.output)
        eager = NativeDeltaPrefill(block)
        compiled = torch._inductor.aoti_load_package(str(args.packages / "prefill.pt2"))
        results = {}
        for name, call in (("hf", BlockStep(block)), ("adapter", eager), ("aoti", compiled)):
            print(f"Running {name}", flush=True)
            results[name] = run_cases(call, cases)
        fp32, _ = load_block(args.model_dir, dtype=torch.float32)
        print("Running FP32 control", flush=True)
        results["fp32"] = run_cases(BlockStep(fp32), cases, torch.float32)

    def compare(name, actual, expected, enforce=True):
        for key in actual:
            for i, (a, b) in enumerate(zip(actual[key], expected[key])):
                error = metrics(a, b, i)
                row = dict(comparison=name, case=key[0], step=key[1], tensor=NAMES[i],
                           bitwise_equal=bits_equal(a, b), **error)
                report["comparisons"].append(row)
                if enforce and max(error["relative_l2"], error["peak_relative"]) > LIMIT:
                    report["failures"].append(row)

    for actual, expected in (("adapter", "hf"), ("aoti", "adapter"), ("aoti", "hf"),
                             ("aoti", "fp32"), ("hf", "fp32")):
        compare(f"{actual}_vs_{expected}", results[actual], results[expected], expected != "fp32")
    for engine, values in results.items():
        final = {}
        for name in ("whole", "split", "uneven", "repeat"):
            steps = [v for (case, _), v in values.items() if case == name]
            final[name, 0] = (torch.cat([v[0] for v in steps], dim=1), *steps[-1][1:])
        compare(f"{engine}_partition", final, {key: final["whole", 0] for key in final})
        assert all(bits_equal(a, b) for a, b in zip(final["uneven", 0], final["repeat", 0]))

    env = {k: os.environ[k] for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH") if k in os.environ}
    env.update(PATH="/usr/bin:/bin", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [str(args.binary.resolve()), str(args.packages.resolve()), str(args.output.resolve())]
    print("Running standalone C++", flush=True)
    child = subprocess.run([*command, str((args.output / "native").resolve())], env=env,
                           text=True, capture_output=True)
    (args.output / "native.log").write_text(child.stdout + child.stderr)
    child.check_returncode()
    for (name, step), tensors in results["aoti"].items():
        compare_file(args.output / "native" / f"{name}-{step}.bin", tensors)
    report["native"] = json.loads((args.output / "native/native.json").read_text())
    report["cpp_aoti_bitwise_tensors"] = 3 * len(results["aoti"])
    report["reset_repeat_bitwise"] = True
    # Test the C++ range guard; invalid lengths must never reach the package.
    report["native_rejections"] = []
    original = json.loads((args.output / "cases.json").read_text())
    try:
        for tokens in (1, 513):
            bad = dict(cases=[dict(name=original["cases"][0]["name"], lengths=[tokens])])
            (args.output / "cases.json").write_text(json.dumps(bad))
            child = subprocess.run([*command, str((args.output / f"reject-{tokens}").resolve())], env=env,
                                   text=True, capture_output=True)
            assert child.returncode != 0 and "Token length outside exported range" in child.stderr
            report["native_rejections"].append(tokens)
    finally:
        (args.output / "cases.json").write_text(json.dumps(original, indent=2) + "\n")
    report["passed"] = not report["failures"]
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "cpp_aoti_bitwise_tensors", "failures")}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
