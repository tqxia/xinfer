"""Validate AOTI prefill → MegaQwen decode with independent HF/FP32 controls.

This is a correctness diagnostic. It runs both native backends on identical
teacher-forced manifests and also checks the shared-buffer handoff in a native
contract probe. It intentionally synchronizes and dumps logits/KV; its elapsed
time is not a decode performance measurement.
"""

import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback

from compiler.common.artifacts import default_build_dir, read_json, require, sha256, write_json
from compiler.common.metadata import read_package_metadata
from tools.validation.common import require_assertions


def dispatch_counts(lengths, backend):
    """The first nonempty call always uses AOTI, including a one-token prompt."""
    require(backend in ("aoti", "megaqwen"), "Unknown validation backend")
    require(
        bool(lengths) and all(type(length) is int and 1 <= length <= 512 for length in lengths),
        "Invalid validation step lengths",
    )
    return dict(
        position=sum(lengths),
        prefill_calls=sum(length > 1 for length in lengths),
        decode_calls=lengths.count(1),
        megaqwen_calls=sum(index > 0 and length == 1 for index, length in enumerate(lengths))
        if backend == "megaqwen"
        else 0,
    )


def check_dispatch(row, lengths, backend):
    expected = dispatch_counts(lengths, backend)
    for name, value in expected.items():
        require(row.get(name) == value, f"Incorrect {backend} {name}: {row.get(name)} != {value}")


def native_environment():
    """Keep the same Python-free subprocess boundary as the common validator."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH")
    }
    env.update(PATH="/usr/bin:/bin", PYTHONHOME="/nonexistent-python-home")
    return env


def extra_cases(cases, seed):
    """Add the missing prompt-25 and AOTI-after-native handoff partitions."""
    require(bool(seed), "Hybrid validation needs a nonempty prompt seed")
    names = {case["name"] for case in cases}

    def prompt(length):
        return (seed * (length // len(seed) + 1))[:length]

    additions = [
        dict(name="hybrid_prompt_25", steps=[prompt(25), [0], [151645], [151935]]),
        dict(
            name="hybrid_back_to_aoti",
            steps=[prompt(2), [13], prompt(13), [0], prompt(2), [151643]],
        ),
    ]
    for order, case in enumerate(additions, start=100):
        require(case["name"] not in names, "Duplicate hybrid fixture")
        case.update(order=order, dump_cache=True)
        names.add(case["name"])
    clean = additions[1]
    additions.append(
        dict(
            name=clean["name"] + "_poison",
            steps=clean["steps"],
            same_as=clean["name"],
            poison_kv=3,
            dump_cache=True,
            order=102,
        )
    )
    require(additions[-1]["name"] not in names, "Duplicate hybrid poison fixture")
    return [*cases, *additions]


def _execute(command, log, timeout):
    with log.open("w") as stream:
        result = subprocess.run(
            [str(value) for value in command],
            env=native_environment(),
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    require(result.returncode == 0, f"Native validation failed ({result.returncode}); see {log}")


def run_native(binary, packages, output, cases, chats, metadata, backend, timeout):
    """Reuse the native validator's existing trace format, with explicit backend selection."""
    import torch

    from compiler.common.tests.tensors import bits_equal
    from tools.validation.common.native import _map, _states
    from tools.validation.common.runner import Trace

    output.mkdir()
    manifest = sorted([*cases, *chats], key=lambda case: case.get("order", 1_000_000))
    require(len({case["name"] for case in manifest}) == len(manifest), "Duplicate native case")
    manifest_path = output / "cases.json"
    write_json(manifest_path, dict(cases=manifest))
    destination = output / "native"
    _execute(
        [
            binary.resolve(),
            packages.resolve(),
            manifest_path.resolve(),
            destination.resolve(),
            "--decode-backend",
            backend,
        ],
        output / "native.log",
        timeout,
    )
    native = read_json(destination / "native.json")
    require(
        native.get("loader_pairs") == 1
        and native.get("no_python_library")
        and native.get("attention_library"),
        "Native validator did not prove its Python-free execution environment",
    )
    info = native.get("backend_info", native.get("decode_info"))
    require(isinstance(info, dict) and info.get("backend") == backend, "Wrong native backend")
    if backend == "megaqwen":
        require(
            info.get("weights_alias") and info.get("cache_alias"),
            "Native backend copied state/weights",
        )
    require(
        [row["name"] for row in native["cases"]] == [case["name"] for case in manifest],
        "Native case coverage differs",
    )
    rows = {row["name"]: row for row in native["cases"]}
    specs = metadata["decode"]["state"]
    vocab = metadata["decode"]["vocab_size"]
    eos = metadata["decode"]["eos_token_ids"]
    trace = Trace()
    for case in cases:
        name, row = case["name"], rows[case["name"]]
        lengths = [len(tokens) for tokens in case["steps"]]
        check_dispatch(row, lengths, backend)
        require(row["rows"] == len(lengths), "Incorrect native logit row count")
        require(
            all(
                row.get(key)
                for key in (
                    "stable_kv_storage",
                    "write_region_checks",
                    "reset_check",
                    "kv_write_regions_preserved",
                    "non_kv_inputs_unchanged",
                )
            ),
            "Native state contract failed",
        )
        if case.get("check_overflow"):
            require(row.get("overflow_rejected_without_mutation"), "Overflow mutated native state")
        logits = _map(destination / f"{name}.f32", torch.float32, len(lengths) * vocab)
        for step, value in enumerate(logits.reshape(-1, 1, vocab)):
            trace.logits[name, step] = value
        if case.get("dump_cache"):
            trace.states[name, len(lengths) - 1] = _states(destination / f"{name}.kv.bf16", specs)
        if "same_as" in case:
            for step in range(len(lengths)):
                require(
                    bits_equal(trace.logits[name, step], trace.logits[case["same_as"], step]),
                    "Native reset/future poison changed logits",
                )
            key = name, len(lengths) - 1
            if key in trace.states:
                original = trace.states[case["same_as"], key[1]]
                for actual, expected in zip(trace.states[key], original):
                    require(
                        bits_equal(actual[:, :, : sum(lengths)], expected[:, :, : sum(lengths)]),
                        "Native future poison changed valid cache",
                    )
    for case in chats:
        name, row = case["name"], rows[case["name"]]
        ids = row["ids"]
        require(0 < len(ids) <= case["limit"], "Invalid native generation length")
        stop_tokens = case.get("eos", eos)
        require(not any(token in stop_tokens for token in ids[:-1]), "Native generation passed EOS")
        stop = "eos" if ids[-1] in stop_tokens else "limit"
        require(
            row["stop"] == stop and (stop == "eos" or len(ids) == case["limit"]),
            "Invalid native stop",
        )
        check_dispatch(row, [len(case["prompt"])] + [1] * (len(ids) - 1), backend)
        logits = _map(destination / f"{name}.f32", torch.float32, len(ids) * vocab)
        value = dict(
            ids=ids, rows=list(logits.reshape(-1, 1, vocab)), position=row["position"], state=()
        )
        if "same_as" in case:
            previous = trace.greedy[case["same_as"]]
            require(value["ids"] == previous["ids"], "Native reset/reuse changed generation")
            require(
                all(bits_equal(a, b) for a, b in zip(value["rows"], previous["rows"])),
                "Native reset/reuse changed generation logits",
            )
        trace.greedy[name] = value
    return trace, native


def _greedy_controls(suite, trace, label, specs, policy, chats, report):
    """HF must receive the candidate's actual continuation if its greedy path differs."""
    import torch

    from tools.validation.common.chain import compare_traces
    from tools.validation.common.runner import Trace, teacher_trace

    checked = 0
    with suite.reference(torch.bfloat16) as runtime:
        for case in chats:
            name = case["name"]
            candidate = trace.greedy[name]
            replay = dict(name=name, steps=[case["prompt"]] + [[t] for t in candidate["ids"][:-1]])
            expected = teacher_trace(runtime, [replay], specs, lambda case, step: False)
            actual = Trace(logits={(name, step): row for step, row in enumerate(candidate["rows"])})
            compare_traces(
                report,
                label,
                "hf_matched_greedy",
                actual,
                expected,
                specs,
                policy,
                state_keys=set(),
                greedy_names=set(),
            )
            checked += len(actual.logits)
    return checked


def validate(args, report):
    import torch

    from compiler.common.attention import load_attention_ops, native_attention_library
    from tools.validation.common.chain import Policy, compare_traces, new_report
    from tools.validation.common.runner import load_aoti_runtime, run_stage, teacher_trace
    from tools.validation.qwen3.model import Suite, default_model_dir

    pair = read_package_metadata(args.packages, model_type="qwen3")
    native = (args.native or default_build_dir("qwen3") / "xinfer_validate").resolve()
    contract = (args.contract_native or native.parent / "qwen_hybrid_contract").resolve()
    require(
        native.is_file() and contract.is_file(),
        "Build xinfer_validate and qwen_hybrid_contract first",
    )
    library = native_attention_library(native)
    require(library is not None, "Missing native attention library")
    os.environ["XINFER_ATTENTION_OPS_LIBRARY"] = str(library)
    load_attention_ops(library, model="qwen3")
    linked = subprocess.check_output(["ldd", str(native)], env=native_environment(), text=True)
    mega_library = None
    for line in linked.splitlines():
        name, separator, target = line.strip().partition(" => ")
        if separator and name == "libxinfer_megaqwen.so":
            mega_library = Path(target.rsplit(" (", 1)[0]).resolve()
    require(mega_library is not None and mega_library.is_file(), "Missing native MegaQwen library")
    suite = Suite(args.model_dir or default_model_dir(pair), pair, args.reference)
    report.update(new_report("qwen3", pair))
    report.update(
        status="running",
        benchmark=False,
        native_sha256=sha256(native),
        contract_native_sha256=sha256(contract),
        attention_library_sha256=sha256(library),
        megaqwen_library=dict(path=str(mega_library), sha256=sha256(mega_library)),
        reference_protocol="HF StaticCache; matched fed tokens and partitions",
        native_bitwise_scope="native AOTI versus Python AOTI; hybrid first-call handoff checked separately",
        hybrid_python_parity="not checked: this integration has no separate Python MegaQwen backend",
        numerical_policy=dict(
            hf_logit_relative_l2=0.05,
            hf_logit_peak_relative=0.05,
            hf_cache_and_token_differences="diagnostic",
            fp32_controls="diagnostic",
            initial_aoti_handoff="bitwise",
            cpp_aoti_vs_python_aoti="bitwise",
            hybrid_vs_aoti="diagnostic after the first call",
        ),
    )
    specs = suite.specs
    policy = Policy(state_required=False, tokens_required=False, state_rows=suite.state_rows)
    with torch.inference_mode():
        print("Stage: independent HF static cache", flush=True)
        with suite.reference(torch.bfloat16) as runtime:
            cases, chats = suite.fixtures(runtime)
            cases = extra_cases(cases, suite.references["capital"][0])
            hf = run_stage(runtime, cases, chats, specs, suite.capture_states, suite.eos)
            for case in chats:
                if "expected_ids" in case:
                    require(
                        hf.greedy[case["name"]]["ids"] == case["expected_ids"],
                        f"Independent HF fixture changed: {case['name']}",
                    )
        report["cases"], report["generation_cases"] = cases, chats
        report["teacher_steps"] = len(hf.logits)
        print("Stage: Python AOTI control", flush=True)
        runtime = load_aoti_runtime(args.packages, model_type="qwen3", specs=specs)
        try:
            compiled = run_stage(runtime, cases, chats, specs, suite.capture_states, suite.eos)
        finally:
            runtime.close()
            del runtime
            torch.cuda.empty_cache()
        native_traces = {}
        for backend in ("aoti", "megaqwen"):
            print(f"Stage: native {backend}", flush=True)
            actual, execution = run_native(
                native,
                args.packages,
                args.output / backend,
                cases,
                chats,
                pair,
                backend,
                args.timeout,
            )
            report.setdefault("native_execution", {})[backend] = execution
            native_traces[backend] = actual
            compare_traces(
                report,
                f"native_{backend}",
                "hf",
                actual,
                hf,
                specs,
                policy,
                state_keys=set(actual.states),
            )
            if backend == "aoti":
                start = len(report["failures"])
                compare_traces(
                    report,
                    "native_aoti",
                    "python_aoti",
                    actual,
                    compiled,
                    specs,
                    policy,
                    exact=True,
                    state_keys=set(actual.states),
                )
                report["native_bitwise"] = len(report["failures"]) == start
        baseline, hybrid = native_traces["aoti"], native_traces["megaqwen"]
        compare_traces(
            report,
            "native_megaqwen",
            "native_aoti",
            hybrid,
            baseline,
            specs,
            policy,
            required=False,
            state_keys=set(hybrid.states),
        )
        handoff = {(case["name"], 0) for case in cases}
        # Both backends must run the first step with AOTI. This includes P=1.
        compare_traces(
            report,
            "hybrid_first_call",
            "aoti_first_call",
            hybrid,
            baseline,
            specs,
            policy,
            exact=True,
            logit_keys=handoff,
            state_keys=set(),
            greedy_names=set(),
        )
        print("Stage: native shared-state and dispatch contracts", flush=True)
        _execute(
            [contract, args.packages.resolve(), (args.output / "contracts.json").resolve()],
            args.output / "contracts.log",
            args.timeout,
        )
        contracts = read_json(args.output / "contracts.json")
        require(
            contracts.get("passed") and contracts.get("no_python_library"),
            "Native hybrid contracts failed",
        )
        report["hybrid_contracts"] = contracts
        matched = {}
        for backend, trace in native_traces.items():
            print(f"Control: HF matched {backend} greedy trajectories", flush=True)
            matched[backend] = _greedy_controls(suite, trace, backend, specs, policy, chats, report)
        print("Control: independent FP32 HF", flush=True)
        checked = 0
        fp32_cases = suite.fp32_cases(cases)
        require(fp32_cases, "Missing FP32 controls")
        with suite.reference(torch.float32) as runtime:
            for case in fp32_cases:
                clean = {name: value for name, value in case.items() if name != "same_as"}
                control = teacher_trace(runtime, [clean], specs, suite.capture_states)
                for label, actual in (("hf_bf16", hf), ("aoti", baseline), ("megaqwen", hybrid)):
                    compare_traces(
                        report,
                        label,
                        "hf_fp32",
                        actual,
                        control,
                        specs,
                        policy,
                        required=False,
                        logit_keys=set(control.logits),
                        state_keys=set(control.states) & set(actual.states),
                        greedy_names=set(),
                    )
                checked += len(control.logits)
                del control
        report["controls"] = dict(hf_fp32_steps=checked, matched_greedy_steps=matched)
        report["passed"] = (
            not report["failures"] and report["native_bitwise"] and contracts["passed"]
        )
        report["status"] = "passed" if report["passed"] else "failed"


def main():
    require_assertions()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--contract-native", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=3600, help="Per native subprocess timeout")
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["AOTI_RUNTIME_CHECK_INPUTS"] = "1"
    report = dict(passed=False, status="running", benchmark=False)
    started = time.monotonic()
    exit_code = 2
    try:
        validate(args, report)
        exit_code = 0 if report["passed"] else 1
    except Exception as error:
        report.update(
            passed=False,
            status="incomplete",
            error=dict(type=type(error).__name__, message=str(error)),
        )
        traceback.print_exc()
    report["elapsed_seconds"] = time.monotonic() - started
    write_json(args.output / "summary.json", report)
    print(f"Passed: {report['passed']}; report: {args.output / 'summary.json'}", flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
