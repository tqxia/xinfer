"""Validate current-kernel AOTI/native parity without loading an HF model bank."""

import argparse
import gc
import os
from pathlib import Path
import re
import traceback

import torch

from compiler.common.artifacts import (
    MODEL_MANIFEST_NAME,
    json_sha256,
    read_json,
    require,
    sha256,
    write_json,
)
from compiler.common.metadata import read_package_metadata
from compiler.qwen36.config import TEXT_DIR
from tools.common import native_attention_library
from tools.validation.common.chain import Policy, compare_traces, new_report
from tools.validation.common.native import run_native
from tools.validation.common.runner import Trace, greedy, load_aoti_runtime, teacher_trace
from tools.validation.qwen36.eager import (
    ObservedRuntime,
    case_trace,
    load_trace,
    memory,
    request_guards,
    save_trace,
)


def reference_info(reference, stage, pair, model_dir):
    """Bind a completed stage's CPU traces to the same checkpoint and package ABI."""
    summary = read_json(reference / "summary.json")
    protocol = read_json(reference / "protocol.json")
    expected = "adapted" if stage == "python" else "python_aoti"
    require(
        summary.get("model_type") == "qwen36"
        and summary.get("stage") == expected
        and summary.get("execution_passed") is True
        and not summary.get("error"),
        f"Expected a completed {expected} reference",
    )
    decode = pair["decode"]
    manifest_path = model_dir / MODEL_MANIFEST_NAME
    require(
        sha256(manifest_path) == protocol["identity"]["manifest_sha256"]
        and read_json(manifest_path) == decode["checkpoint"]["manifest"],
        "Reference and package checkpoint manifests differ",
    )
    require(
        protocol["identity"]["layers"] == 40
        and protocol["identity"]["capacity"] == decode["cache_length"]
        and protocol["state"] == decode["state"]
        and protocol["eos_token_ids"] == decode["eos_token_ids"],
        "Reference state or execution bounds differ from package",
    )
    require(
        pair["prefill"]["token_range"] == [2, 512] and decode["token_range"] == [1, 1],
        "Unexpected package entry bounds",
    )
    names = [case["name"] for group in protocol["groups"] for case in group] + [
        case["name"] for case in protocol["chats"]
    ]
    require(
        len(names) == len(set(names))
        and names
        and all(re.fullmatch(r"[A-Za-z0-9_]+", name) for name in names),
        "Invalid or duplicate trace case names",
    )
    require(
        protocol["input_sha256"]
        == json_sha256(dict(groups=protocol["groups"], chats=protocol["chats"])),
        "Reference input checksum differs",
    )
    require(
        protocol["identity"]["profile"] == "matrix"
        and {
            "length_2",
            "length_96",
            "length_98",
            "length_512",
            "nonzero_prefill",
            "multicall_513",
            "repeat_98",
            "poison_98",
        }
        <= set(names),
        "A complete current matrix reference is required",
    )
    cases = [case for group in protocol["groups"] for case in group]
    require(
        summary.get("teacher_steps") == sum(len(case["steps"]) for case in cases)
        and summary.get("state_checkpoints")
        == sum(len({case["prompt_steps"] - 1, len(case["steps"]) - 1}) for case in cases)
        and summary.get("greedy_cases") == len(protocol["chats"]),
        "Reference execution coverage differs from protocol",
    )
    require(
        all((reference / "traces" / f"{name}.pt").is_file() for name in names),
        "Missing reference trace",
    )
    hashes = {kind: item["package_sha256"] for kind, item in pair.items()}
    if stage == "native":
        require(summary.get("package_sha256") == hashes, "Python reference packages differ")
        hf = summary["hf_numerical_reference"]
    else:
        hf = dict(
            passed=summary.get("comparison_passed") is True,
            failures=len(summary.get("failures", [])),
            source=str((reference / "summary.json").resolve()),
            summary_sha256=sha256(reference / "summary.json"),
        )
    return protocol, summary, hf


def compare_output(report, actual, expected, specs, *, exact):
    """Compiler checks retain the existing limits; native checks are always exact."""
    policy = Policy(state_required=True, tokens_required=True)
    matching = {
        name
        for name, value in actual.greedy.items()
        if value["ids"] == expected.greedy[name]["ids"]
        and value["position"] == expected.greedy[name]["position"]
    }
    compare_traces(
        report,
        "cpp_aoti" if exact else "python_aoti",
        "python_aoti" if exact else "adapted_eager",
        actual,
        expected,
        specs,
        policy,
        exact=exact,
        greedy_state_names=matching,
    )
    if not exact:
        for name, value in actual.greedy.items():
            other = expected.greedy[name]
            count = 0
            for token, reference_token in zip(value["ids"], other["ids"]):
                count += 1
                if token != reference_token:
                    break
            left = Trace(logits={(name, i): row for i, row in enumerate(value["rows"][:count])})
            right = Trace(logits={(name, i): row for i, row in enumerate(other["rows"][:count])})
            compare_traces(
                report,
                "python_aoti",
                "adapted_eager_greedy_prefix",
                left,
                right,
                specs,
                policy,
                state_keys=set(),
                greedy_names=set(),
            )


def python_stage(args, protocol, pair, report):
    from compiler.qwen36.config import configure

    configure()
    require(torch.cuda.is_available(), "CUDA is required")
    decode = pair["decode"]
    require(str(torch.__version__) == decode["torch_version"], "Torch version differs from package")
    require(
        list(torch.cuda.get_device_capability()) == decode["cuda_compute_capability"],
        "GPU architecture differs from package",
    )
    if args.attention_library:
        os.environ["XINFER_ATTENTION_OPS_LIBRARY"] = str(args.attention_library.resolve())
    torch.cuda.reset_peak_memory_stats()
    report["memory"].append(memory("before_package_load"))
    runtime = load_aoti_runtime(args.packages, model_type="qwen36", specs=protocol["state"])
    report["memory"].append(memory("after_package_load"))
    observed = ObservedRuntime(runtime, None, [], report)
    args.output.joinpath("traces").mkdir()
    specs = protocol["state"]
    try:
        for group in protocol["groups"]:
            observed.case_names = iter(case["name"] for case in group)
            trace = teacher_trace(
                observed,
                group,
                specs,
                lambda case, step: step in {case["prompt_steps"] - 1, len(case["steps"]) - 1},
            )
            for case in group:
                name = case["name"]
                selected = case_trace(trace, name)
                expected, _ = load_trace(args.reference / "traces" / f"{name}.pt")
                compare_output(report, selected, expected, specs, exact=False)
                save_trace(args.output / "traces" / f"{name}.pt", selected, {})
                report["teacher_steps"] += len(selected.logits)
                report["state_checkpoints"] += len(selected.states)
                print(f"Python AOTI: {name}, failures={len(report['failures'])}", flush=True)
            del trace, selected, expected
        observed.case_names = None
        for chat in protocol["chats"]:
            name = chat["name"]
            observed.case = name
            result = greedy(observed, chat["prompt"], chat["limit"], protocol["eos_token_ids"])
            trace = Trace(greedy={name: result}, compact_states=runtime.capacity > 1024)
            expected, _ = load_trace(args.reference / "traces" / f"{name}.pt")
            compare_output(report, trace, expected, specs, exact=False)
            save_trace(args.output / "traces" / f"{name}.pt", trace, {})
            report["greedy_cases"] += 1
        report["request_guards"] = request_guards(runtime)
        report["execution_passed"] = True
        report["memory"].append(memory("after_requests"))
    finally:
        observed.close()
        gc.collect()
        torch.cuda.empty_cache()


def native_stage(args, protocol, pair, report):
    require(args.native and args.native.is_file(), "Build xinfer_validate first")
    library = native_attention_library(args.native)
    require(library is not None, "Native worker requires its linked attention library")
    report["attention_library_sha256"] = sha256(library)
    report["native_sha256"] = sha256(args.native)
    cases = [case for group in protocol["groups"] for case in group]
    state_keys = {
        (case["name"], step)
        for case in cases
        for step in {case["prompt_steps"] - 1, len(case["steps"]) - 1}
    }
    trace, execution = run_native(
        args.native,
        args.packages,
        args.output,
        cases,
        protocol["chats"],
        pair,
        state_keys=state_keys,
    )
    for case in cases:
        name = case["name"]
        actual = case_trace(trace, name)
        expected, _ = load_trace(args.reference / "traces" / f"{name}.pt")
        compare_output(report, actual, expected, protocol["state"], exact=True)
        report["teacher_steps"] += len(actual.logits)
        report["state_checkpoints"] += len(actual.states)
    for chat in protocol["chats"]:
        name = chat["name"]
        actual = Trace(greedy={name: trace.greedy[name]}, compact_states=trace.compact_states)
        expected, _ = load_trace(args.reference / "traces" / f"{name}.pt")
        compare_output(report, actual, expected, protocol["state"], exact=True)
        report["greedy_cases"] += 1
    report["native_execution"] = execution
    report["native_bitwise"] = not report["failures"]
    report["execution_passed"] = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("python", "native"), required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=TEXT_DIR,
        help="Only model_manifest.json is read; no HF model is loaded",
    )
    parser.add_argument("--native", type=Path)
    parser.add_argument(
        "--attention-library", type=Path, default=os.environ.get("XINFER_ATTENTION_OPS_LIBRARY")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    report = new_report("qwen36")
    report.update(
        stage="python_aoti" if args.stage == "python" else "cpp_aoti",
        scope="Current-kernel functional pipeline; HF numerical acceptance remains separate",
        execution_passed=False,
        functional_passed=False,
        model_accepted=False,
        native_bitwise=None,
        memory=[],
        teacher_steps=0,
        state_checkpoints=0,
        greedy_cases=0,
    )
    try:
        from compiler.qwen36.packages import validate_model_metadata

        pair = read_package_metadata(args.packages, model_type="qwen36")
        for metadata in pair.values():
            validate_model_metadata(metadata)
        protocol, reference, hf = reference_info(args.reference, args.stage, pair, args.model_dir)
        write_json(args.output / "protocol.json", protocol)
        report.update(
            package_sha256={kind: item["package_sha256"] for kind, item in pair.items()},
            shared_weights=pair["decode"]["shared_weights"],
            hf_numerical_reference=hf,
            hf_numerical_passed=hf["passed"],
            reference_summary_sha256=sha256(args.reference / "summary.json"),
            compiler_numerical_policy=dict(
                relative_l2=0.05, peak_relative=0.05, tokens="required", native="bitwise"
            ),
            omitted_from_package_abi="Decoder hooks are not package outputs; their required HF/eager results remain in the reference summary",
        )
        if args.stage == "native":
            report["compiler_eager_parity_passed"] = reference.get("functional_passed")
        with torch.inference_mode():
            if args.stage == "python":
                python_stage(args, protocol, pair, report)
            else:
                native_stage(args, protocol, pair, report)
        measured = [row for row in report["comparisons"] if "bitwise_equal" in row]
        report["bitwise_comparisons"] = dict(
            equal=sum(row["bitwise_equal"] for row in measured), total=len(measured)
        )
        report["functional_passed"] = report["execution_passed"] and not report["failures"]
        report["passed"] = report["functional_passed"]
        # Two bounded comparisons are not transitive. Direct compiled/HF
        # acceptance is separate work even if both intermediate checks pass.
        report["model_acceptance_policy"] = (
            "Not evaluated here; direct compiled-versus-HF numerical acceptance remains required"
        )
    except Exception as error:
        report.update(passed=False, error=dict(type=type(error).__name__, message=str(error)))
        traceback.print_exc()
    write_json(args.output / "summary.json", report)
    print(
        f"Functional pass: {report['functional_passed']}; HF numerical pass: {report.get('hf_numerical_passed')}; report: {args.output / 'summary.json'}",
        flush=True,
    )
    return 0 if report["functional_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
