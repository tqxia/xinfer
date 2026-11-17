"""Validate current-kernel AOTI/native parity without loading an HF model bank."""

import argparse
import gc
from pathlib import Path
import traceback

import torch

from compiler.common.artifacts import (
    require,
    sha256,
    write_json,
)
from compiler.common.metadata import read_package_metadata
from compiler.qwen36.config import TEXT_DIR
from tools.validation.common.chain import new_report
from tools.validation.common.native import run_native
from tools.validation.common.runner import Trace, greedy, load_aoti_runtime, teacher_trace
from tools.validation.qwen36.common import (
    compare_output,
    reference_info,
    ObservedRuntime,
    case_trace,
    load_trace,
    memory,
    request_guards,
    save_trace,
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
        from tools.packages import validate_model_metadata

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
