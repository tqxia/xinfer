"""Verify an ordinary Qwen3 package through Python and native AOTI loaders.

This checks compiled-profile routing, caller-owned KV and exact deployment parity.
It does not inspect the tensor pointers received inside the operator.
It does not replace live HF numerical validation or make performance claims.
"""

import argparse
import zipfile
from pathlib import Path

import torch

from compiler.common.artifacts import read_json, require, sha256, write_json
from compiler.common.metadata import read_package_metadata
from compiler.common.tests.tensors import bits_equal
from compiler.qwen3.packages import decode_backend
from tools.common import run
from tools.validation.common.native import native_environment
from tools.validation.common.runner import load_aoti_runtime
from tools.validation.megaqwen_op import (
    cache_file_digests,
    cases_for_capacity,
    inspect_generated_copies,
    inspect_megaqwen_dispatch,
    require_no_unexpected_copies,
    tensor_digest,
)


def expected_counts(lengths):
    """Count graph calls and consumed tokens from the input partitions."""
    return {
        "position": sum(lengths),
        "rows": len(lengths),
        "prefill_calls": sum(length > 1 for length in lengths),
        "decode_calls": lengths.count(1),
    }


def inspect_package(package, output, backend):
    sources = {}
    with zipfile.ZipFile(package) as archive:
        for name in archive.namelist():
            if name.endswith((".cpp", ".h")):
                text = archive.read(name).decode()
                destination = output / "generated" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(text)
                sources[name] = text
    require(bool(sources), "Decode package contains no generated source to inspect")
    result = inspect_generated_copies(sources)
    if backend == "megaqwen":
        require_no_unexpected_copies(result)
        result.update(inspect_megaqwen_dispatch(sources))
    write_json(output / "source-inspection.json", result)
    return result


def python_trace(runtime, cases, output):
    results = []
    runtime.reset()
    persistent = [tensor.data_ptr() for tensor in runtime.state]
    for case in cases:
        torch.cuda.synchronize()
        stream = (
            torch.cuda.Stream() if case.get("nondefault_stream") else torch.cuda.current_stream()
        )
        retained = []
        path = output / f"{case['name']}.python.f32"
        with torch.cuda.stream(stream), path.open("wb") as file:
            runtime.reset(case.get("poison_kv", 0))
            for tokens in case["steps"]:
                score = runtime.step(
                    tokens
                )  # Shared runtime checks full write regions and finiteness.
                require(
                    [tensor.data_ptr() for tensor in runtime.state] == persistent,
                    "Reset or execution replaced caller-owned KV",
                )
                for previous, expected in retained:
                    require(
                        previous.data_ptr() != score.data_ptr()
                        and bits_equal(previous.cpu(), expected),
                        "A later package call reused or changed retained logits",
                    )
                cpu = score.cpu().clone()
                retained.append((score, cpu))
                file.write(cpu.float().numpy().tobytes())
            hashes = [tensor_digest(tensor[:, :, : runtime.position]) for tensor in runtime.state]
            if case.get("check_overflow"):
                require(runtime.position == runtime.capacity, "Overflow case did not fill context")
                rejected = False
                try:
                    runtime.step([13])
                except ValueError:
                    rejected = True
                require(rejected, "Context overflow was accepted")
                require(
                    hashes == [tensor_digest(tensor) for tensor in runtime.state],
                    "Rejected overflow changed caller KV",
                )
        row = {
            "name": case["name"],
            **expected_counts(list(map(len, case["steps"]))),
            "logits_file": str(path),
            "valid_kv_sha256": hashes,
            "dispatch_counts_source": "package_profile_and_input_lengths",
            "caller_owned_storage_preserved": True,
            "fresh_output_storage": True,
            "passed": True,
        }
        results.append(row)
    by_name = {row["name"]: row for row in results}
    for case in cases:
        if "same_as" in case:
            actual, expected = by_name[case["name"]], by_name[case["same_as"]]
            require(
                sha256(actual["logits_file"]) == sha256(expected["logits_file"])
                and actual["valid_kv_sha256"] == expected["valid_kv_sha256"],
                "Poison/reset changed Python package logits or valid KV",
            )
    return results


def check_native(cases, python, native, destination):
    require(
        native.get("passed") is True and native.get("no_python_library") is True,
        "Native package correctness or Python-free linkage failed",
    )
    require(native.get("async_destruction_lifetime") is True, "Native lifetime probe is missing")
    require(
        [row["name"] for row in native["cases"]] == [case["name"] for case in cases],
        "Native package case coverage differs",
    )
    for case, expected, actual in zip(cases, python, native["cases"]):
        require(actual.get("passed") is True, "Native package case did not pass")
        for key in ("position", "rows", "prefill_calls", "decode_calls"):
            require(actual.get(key) == expected[key], f"Native package {key} differs")
        require(
            actual.get("caller_owned_storage_preserved") is True,
            "Native caller-owned storage check is missing",
        )
        if case.get("check_overflow"):
            require(
                actual.get("overflow_rejected_without_mutation") is True,
                "Native overflow contract is missing",
            )
        require(
            sha256(destination / f"{case['name']}.f32") == sha256(expected["logits_file"]),
            f"Native/Python package logits differ bitwise: {case['name']}",
        )
        if case.get("dump_cache"):
            require(
                cache_file_digests(destination / f"{case['name']}.kv.bf16", expected["position"])
                == expected["valid_kv_sha256"],
                f"Native/Python package KV differs bitwise: {case['name']}",
            )


def verify(args, report):
    pair = read_package_metadata(args.packages, model_type="qwen3")
    backend = decode_backend(pair["decode"])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    runtime = load_aoti_runtime(args.packages, model_type="qwen3", specs=pair["decode"]["state"])
    rope = (
        [
            runtime.owner.tensors[runtime.owner.manifest["sources"][f"megaqwen_{key}"]]
            for key in ("cos", "sin")
        ]
        if backend == "megaqwen"
        else []
    )
    rope_before = [tensor_digest(tensor) for tensor in rope]
    cases = cases_for_capacity(runtime.capacity, runtime.max_prefill)
    write_json(args.output / "cases.json", {"cases": cases})
    report.update(
        compiled_decode_backend=backend,
        capacity=runtime.capacity,
        source_inspection=inspect_package(args.packages / "decode.pt2", args.output, backend),
    )
    python = python_trace(runtime, cases, args.output)
    require(rope_before == [tensor_digest(tensor) for tensor in rope], "Canonical RoPE was mutated")
    report["python"] = python
    del rope
    runtime.close()
    torch.cuda.empty_cache()
    destination = args.output / "native"
    run(
        [
            args.native.resolve(),
            args.packages.resolve(),
            (args.output / "cases.json").resolve(),
            destination.resolve(),
        ],
        native_environment(),
        args.output / "native.log",
    )
    native = read_json(destination / "summary.json")
    require(native.get("compiled_decode_backend") == backend, "Native package profile differs")
    check_native(cases, python, native, destination)
    report.update(
        native=native,
        native_python_logits_bitwise=True,
        native_python_selected_kv_bitwise=True,
        passed=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Choose an absent output directory")
    args.output.mkdir(parents=True)
    report = {"passed": False, "benchmark": False, "numerical_acceptance": False}
    try:
        with torch.inference_mode():
            verify(args, report)
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        write_json(args.output / "summary.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
