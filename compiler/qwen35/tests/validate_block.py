"""Compare HF, the state adapter, AOTI and a Python-free C++ block runner."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch
from transformers import DynamicCache

from compiler.qwen35.config import ARTIFACT_DIR, PREFILL, configure
from compiler.common.artifacts import sha256, ROOT
from compiler.qwen35.diagnostics.block import BlockStep, MODEL_DIR, input_shapes, load_block, zero_state
from compiler.common.validation import bits_equal
from compiler.qwen35.tests.common import metrics
from compiler.qwen35.tests.sequence import chunks, run_sequence


LIMIT = 0.05
NAMES = ("output", "convolution", "recurrent")


def fixtures(directory, config):
    generator = torch.Generator().manual_seed(35)
    inputs = {"prefill": torch.randn(input_shapes(config, PREFILL)[0], generator=generator).bfloat16()}
    inputs.update({f"decode_{i}": torch.randn(input_shapes(config, 1)[0], generator=generator).bfloat16()
                   for i in range(8)})
    for name, value in inputs.items():
        (directory / f"{name}.bin").write_bytes(value.view(torch.uint8).numpy().tobytes())

    def step(name):
        return dict(kind="prefill" if name == "prefill" else "decode", file=f"{name}.bin")

    chain = [step("prefill")] + [step(f"decode_{i}") for i in range(8)]
    cases = [dict(name="prefill_decode", steps=chain),
             dict(name="decode_zero", steps=[step(f"decode_{i}") for i in range(4)]),
             dict(name="prefill_continue", steps=[step("prefill"), step("prefill"), step("decode_0")]),
             dict(name="repeat", steps=chain)]
    (directory / "cases.json").write_text(json.dumps(dict(cases=cases), indent=2) + "\n")
    return cases, inputs


def hf_trace(block, config, cases, inputs, dtype, partitioned=False):
    trace = {}
    for case in cases:
        cache = DynamicCache(config=config)
        for i, step in enumerate(case["steps"]):
            x = inputs[Path(step["file"]).stem].to(device="cuda", dtype=dtype)
            pieces = list(chunks(x)) if partitioned else [x]
            outputs = [block(piece, cache_params=cache) for piece in pieces]
            output = torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]
            layer = cache.layers[0]
            trace[(case["name"], i)] = tuple(t.cpu().clone() for t in
                (output, layer.conv_states[0], layer.recurrent_states[0]))
    return trace


def explicit_trace(functions, config, cases, inputs):
    trace = {}
    for case in cases:
        conv, state = zero_state(config)
        for i, step in enumerate(case["steps"]):
            x = inputs[Path(step["file"]).stem].cuda()
            args = (x, conv, state)
            before = tuple(t.clone() for t in args)
            result = run_sequence(functions, *args)
            assert len(result) == 3
            for a, b, c in zip(args, before, result):
                assert bits_equal(a, b), "Input mutation"
                assert c.shape == a.shape and c.dtype == a.dtype
            trace[(case["name"], i)] = tuple(t.cpu().clone() for t in result)
            conv, state = result[1:]
    return trace


def check_native_rejections(args, env):
    """Invalid tensor contracts must fail before loading a package or running CUDA."""
    cases = (("hidden", 0, "Package hidden dimensions differ"),
             ("state", 1, "Package state shapes differ"),
             ("dtype", 2, "Unexpected input dtype"),
             ("stride", 0, "Expected positive contiguous input shape"))
    for name, index, expected in cases:
        directory = args.output / "rejections" / name
        directory.mkdir(parents=True)
        for kind in ("prefill", "decode"):
            meta = json.loads((args.packages / f"{kind}.json").read_text())
            if kind == "decode":
                item = meta["inputs"][index]
                if name in ("hidden", "state"):
                    item["shape"][-1] += 1
                    stride = 1
                    for d in reversed(range(len(item["shape"]))):
                        item["stride"][d] = stride
                        stride *= item["shape"][d]
                elif name == "dtype":
                    item["dtype"] = "torch.bfloat16"
                else:
                    item["stride"][-1] = 2
            (directory / f"{kind}.json").write_text(json.dumps(meta) + "\n")
        command = [str(args.binary.resolve()), str(directory.resolve()), str(args.output.resolve()),
                   str((directory / "native").resolve())]
        child = subprocess.run(command, env=env, text=True, capture_output=True)
        (directory / "native.log").write_text(child.stdout + child.stderr)
        assert child.returncode != 0 and expected in child.stderr, f"Missing native rejection: {name}"
    rejected = [name for name, _, _ in cases]
    for name, kind, tokens, expected in (
        ("empty", "prefill", 0, "Expected 1 <= tokens <= 512"),
        ("oversized", "prefill", 513, "Expected 1 <= tokens <= 512"),
        ("decode_length", "decode", 2, "Decode requires one token"),
    ):
        directory = args.output / "rejections" / name
        directory.mkdir(parents=True)
        manifest = dict(cases=[dict(name=name, steps=[dict(kind=kind, tokens=tokens, file="unused.bin")])])
        (directory / "cases.json").write_text(json.dumps(manifest) + "\n")
        child = subprocess.run([str(args.binary.resolve()), str(args.packages.resolve()),
                                str(directory.resolve()), str((directory / "native").resolve())],
                               env=env, text=True, capture_output=True)
        (directory / "native.log").write_text(child.stdout + child.stderr)
        assert child.returncode != 0 and expected in child.stderr, f"Missing native rejection: {name}"
        rejected.append(name)
    return rejected


def compare_partitions(traces, cases, report):
    """Compare the same token stream split at different request boundaries."""
    by_name = {case["name"]: case for case in cases}
    rows = []
    for case in cases:
        if "compare_to" not in case:
            continue
        reference = by_name[case["compare_to"]]
        for path, trace in traces.items():
            a = [trace[(case["name"], i)] for i in range(len(case["steps"]))]
            b = [trace[(reference["name"], i)] for i in range(len(reference["steps"]))]
            comparisons = [("all_outputs", torch.cat([x[0] for x in a], dim=1),
                            torch.cat([x[0] for x in b], dim=1), 0)]
            for point, ai, bi in (("prefix", case["prefix_steps"] - 1, reference["prefix_steps"] - 1),
                                  ("final", len(a) - 1, len(b) - 1)):
                comparisons.extend((f"{point}_{NAMES[i]}", a[ai][i], b[bi][i], i) for i in (1, 2))
            for label, actual, expected, axis in comparisons:
                error = metrics(actual, expected, axis)
                row = dict(case=case["name"], reference=reference["name"], path=path,
                           tensor=label, bitwise_equal=bits_equal(actual, expected), **error)
                rows.append(row)
                if path != "fp32" and max(error["relative_l2"], error["peak_relative"]) > LIMIT:
                    report["failures"].append(dict(comparison="partition", **row))
    report["partition_comparisons"] = rows


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--packages", type=Path, default=ARTIFACT_DIR / "block")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--sequence-tests", action="store_true", help="Boundary, partition and 512-step decode tests")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.output = args.output or ARTIFACT_DIR / ("sequence-validation" if args.sequence_tests else "validation")
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    configure()
    args.output.mkdir(parents=True)
    block, config = load_block(args.model_dir)
    if args.sequence_tests:
        from compiler.qwen35.tests.sequence_cases import fixtures as sequence_fixtures
        cases, inputs = sequence_fixtures(args.output, config)
    else:
        cases, inputs = fixtures(args.output, config)
    for kind in ("prefill", "decode"):
        meta = json.loads((args.packages / f"{kind}.json").read_text())
        assert meta["torch"] == torch.__version__
        assert meta["package_sha256"] == sha256(args.packages / f"{kind}.pt2")
        assert meta["checkpoint"] == json.loads((args.model_dir / "manifest.json").read_text())
        expected_shapes = input_shapes(config, PREFILL if kind == "prefill" else 1)
        assert [item["shape"] for item in meta["inputs"]] == [list(shape) for shape in expected_shapes]
        assert [item["dtype"] for item in meta["inputs"]] == ["torch.bfloat16", "torch.bfloat16", "torch.float32"]
    print(f"Validating {len(cases)} cases / {sum(len(case['steps']) for case in cases)} calls", flush=True)
    print("HF eager reference", flush=True)
    reference = hf_trace(block, config, cases, inputs, torch.bfloat16)
    print("HF with matching chunk partitions", flush=True)
    matched_reference = (hf_trace(block, config, cases, inputs, torch.bfloat16, partitioned=True)
                         if args.sequence_tests else reference)
    wrapper = BlockStep(block)
    print("Explicit state adapter", flush=True)
    adapted = explicit_trace(dict(prefill=wrapper, decode=wrapper), config, cases, inputs)
    fp32_block, _ = load_block(args.model_dir, torch.float32)
    print("FP32 reference", flush=True)
    fp32 = hf_trace(fp32_block, config, cases, inputs, torch.float32)
    functions = {kind: torch._inductor.aoti_load_package(str(args.packages / f"{kind}.pt2"))
                 for kind in ("prefill", "decode")}
    print("Python AOTI", flush=True)
    compiled = explicit_trace(functions, config, cases, inputs)

    # The child inherits CUDA visibility/driver paths, but no Python environment.
    env = {k: os.environ[k] for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH") if k in os.environ}
    env.update(PATH="/usr/bin:/bin", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [str(args.binary.resolve()), str(args.packages.resolve()), str(args.output.resolve()),
               str((args.output / "native").resolve())]
    print("Native C++", flush=True)
    child = subprocess.run(command, env=env, text=True, capture_output=True)
    (args.output / "native.log").write_text(child.stdout + child.stderr)
    child.check_returncode()
    ldd = subprocess.check_output(["ldd", str(args.binary.resolve())], text=True)
    (args.output / "ldd.txt").write_text(ldd)
    assert "libpython" not in ldd and "libtorch_python" not in ldd
    report = dict(limit=LIMIT, fixture="seeded hidden states; real layer-0 weights", seed=35,
                  checkpoint=json.loads((args.model_dir / "manifest.json").read_text()),
                  source_sha256={name: sha256(ROOT / name) for name in ('compiler/qwen35/config.py', 'compiler/qwen35/diagnostics/block.py', 'compiler/qwen35/tests/validate_block.py', 'compiler/qwen35/tests/sequence.py', 'compiler/qwen35/tests/sequence_cases.py', 'compiler/qwen35/tests/common.py', 'compiler/common/validation.py', 'compiler/common/artifacts.py')},
                  cases=cases, comparisons=[], failures=[], binary_sha256=sha256(args.binary),
                  native=json.loads((args.output / "native/native.json").read_text()),
                  native_rejections=check_native_rejections(args, env),
                  input_mutation=False, fp32_control="FP32 computation using the same checkpoint weights",
                  metric_axes=dict(output="per token", convolution="per channel", recurrent="per head"))
    for key, tensors in compiled.items():
        name, step = key
        row = dict(case=name, step=step, tensors={})
        for i, (label, value) in enumerate(zip(NAMES, tensors)):
            path = args.output / "native" / f"{name}-{step}-{i}.bin"
            native = torch.frombuffer(bytearray(path.read_bytes()), dtype=value.dtype).reshape(value.shape)
            assert bits_equal(native, value), f"C++/Python AOTI differ: {path}"
            entry = dict(cpp_aoti_bitwise=True)
            for comparison, actual, expected in (
                ("adapter_vs_hf", adapted[key][i], matched_reference[key][i]),
                ("chunked_hf_vs_hf", matched_reference[key][i], reference[key][i]),
                ("aoti_vs_hf", value, reference[key][i]),
                ("aoti_vs_adapter", value, adapted[key][i]),
                ("aoti_vs_fp32", value, fp32[key][i]),
                ("hf_vs_fp32", reference[key][i], fp32[key][i]),
            ):
                error = metrics(actual, expected, i)
                error["bitwise_equal"] = bits_equal(actual, expected)
                entry[comparison] = error
                if comparison in ("adapter_vs_hf", "chunked_hf_vs_hf", "aoti_vs_hf", "aoti_vs_adapter"):
                    if max(error["relative_l2"], error["peak_relative"]) > LIMIT:
                        report["failures"].append(dict(case=name, step=step, tensor=label,
                                                       comparison=comparison, **error))
            row["tensors"][label] = entry
        report["comparisons"].append(row)
    for case in cases:
        original = case.get("repeat_of", "prefill_decode" if case["name"] == "repeat" else None)
        if original:
            for i in range(len(case["steps"])):
                assert all(bits_equal(a, b) for a, b in zip(compiled[(original, i)], compiled[(case["name"], i)]))
    compare_partitions(dict(hf=reference, adapter=adapted, aoti=compiled, fp32=fp32), cases, report)
    report["reset_repeat_bitwise"] = True
    report["passed"] = not report["failures"]
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "reset_repeat_bitwise", "failures")}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
