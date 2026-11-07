"""Execute one HF → eager → Python AOTI → C++ AOTI validation chain."""

from contextlib import closing
from dataclasses import replace

import torch

from compiler.common.artifacts import require, sha256
from compiler.common.tests.tensors import bits_equal, cpu_copy
from tools.validation.common.chain import Policy, compare_traces
from tools.validation.common.native import run_native
from tools.validation.common.runner import Trace, load_aoti_runtime, run_stage, teacher_trace
from tools.common import prompt_chunks


def _close(runtime):
    if hasattr(runtime, "close"):
        runtime.close()


def _stage_summary(name, trace, *, control=False):
    return dict(
        name=name,
        control=control,
        teacher_steps=len(trace.logits),
        state_checkpoints=len(trace.states),
        greedy_cases=len(trace.greedy),
    )


def _contract_probes(runtime, specs, seed, lengths):
    for length in lengths:
        tokens = (seed * (length // len(seed) + 1))[:length]
        runtime.reset()
        clean = cpu_copy((runtime.step(tokens), *runtime.state))
        runtime.reset({"key": 7, "value": -3})
        logits = runtime.step(tokens)
        require(bits_equal(logits.cpu(), clean[0]), "Future cache affects logits")
        for actual, expected, spec in zip(runtime.state, clean[1:], specs):
            actual = actual.cpu()
            if spec["update"] == "inplace":
                require(
                    bits_equal(actual[:, :, :length], expected[:, :, :length]),
                    "Poison changes valid KV",
                )
                fill = 7 if spec["kind"] == "key" else -3
                require(bool((actual[:, :, length:] == fill).all()), "Future KV overwritten")
            else:
                require(bits_equal(actual, expected), "Poison changes functional state")
        runtime.step([tokens[0]])
    return dict(lengths=list(lengths), poisoned_future_ignored=True, write_regions_preserved=True)


def _greedy_anchors(suite, report, hf, compiled, specs, policy, chats):
    """HF logits must see the same continuation as the compiled greedy run."""
    mismatches = [
        case
        for case in chats
        if hf.greedy[case["name"]]["ids"] != compiled.greedy[case["name"]]["ids"]
    ]
    replay = {}
    if mismatches:
        with suite.reference(torch.bfloat16) as runtime:
            for case in mismatches:
                name = case["name"]
                ids = compiled.greedy[name]["ids"]
                chunks = prompt_chunks(case["prompt"], getattr(runtime, "max_prefill", 512))
                teacher = dict(name=name, steps=chunks + [[token] for token in ids[:-1]])
                result = teacher_trace(runtime, [teacher], specs, lambda case, step: False)
                replay[name] = [
                    result.logits[name, len(chunks) - 1 + step] for step in range(len(ids))
                ]
    for case in chats:
        name = case["name"]
        expected = replay[name] if name in replay else hf.greedy[name]["rows"]
        actual = compiled.greedy[name]["rows"]
        require(len(actual) == len(expected), "Missing greedy teacher logits")
        left = Trace(logits={(name, step): value for step, value in enumerate(actual)})
        right = Trace(logits={(name, step): value for step, value in enumerate(expected)})
        compare_traces(
            report,
            "python_aoti",
            "hf_greedy_teacher",
            left,
            right,
            specs,
            replace(policy, tokens_required=False),
            state_keys=set(),
            greedy_names=set(),
        )


@torch.inference_mode()
def validate_chain(suite, packages, native, output, metadata, report):
    specs = suite.specs
    policy = Policy(
        state_required=suite.state_required,
        tokens_required=suite.tokens_required,
        state_rows=getattr(suite, "state_rows", None),
    )
    numerical_policy = replace(policy, tokens_required=False)
    report["reference_protocol"] = dict(
        cache="HF StaticCache", teacher_inputs="matched tokens and partitions"
    )
    report["state_snapshot_policy"] = dict(
        kv="populated prefix; unused future fill checked independently on capture"
        if suite.capacity > 1024
        else "full cache",
        functional_states="complete",
        long_context="completed prompt and final request checkpoints; every logit retained",
        native_write_and_ownership_checks="every call",
    )
    report["numerical_policy"] = dict(
        limits=dict(relative_l2=policy.limit, peak_relative=policy.limit),
        hf_state_errors="required" if policy.state_required else "reported",
        hf_token_differences="required for final AOTI on end-to-end cases"
        if policy.tokens_required
        else "reported",
        native_python_aoti="bitwise",
        fp32_controls="reported",
    )
    report["native_sha256"] = sha256(native)
    hf = eager = None
    cases = chats = None
    seen = set()
    with closing(suite.stages()) as stages:
        for stage in stages:
            name, runtime = stage.name, stage.runtime
            require(name not in seen, "Duplicate validation stage")
            seen.add(name)
            print(f"Stage: {name}", flush=True)
            try:
                if cases is None:
                    require(name == "hf", "The validation chain must start with HF")
                    cases, chats = suite.fixtures(runtime)
                    report["cases"] = cases
                    report["generation_cases"] = chats
                selected = suite.stage_cases(name, cases)
                control = name.startswith("hf_")
                result = run_stage(
                    runtime,
                    selected,
                    [] if control else chats,
                    specs,
                    suite.capture_states,
                    suite.eos,
                )
                if name == "hf":
                    for case in chats:
                        if "expected_ids" in case:
                            require(
                                result.greedy[case["name"]]["ids"] == case["expected_ids"],
                                f"HF reference fixture changed: {case['name']}",
                            )
                    hf = result
                elif control:
                    compare_traces(
                        report,
                        name,
                        "hf",
                        result,
                        hf,
                        specs,
                        numerical_policy,
                        logit_keys=set(result.logits),
                        state_keys=set(result.states),
                        greedy_names=set(),
                    )
                else:
                    require(name == "eager", "Unknown eager stage")
                    eager = result
                    compare_traces(report, "eager", "hf", eager, hf, specs, numerical_policy)
                report["stages"].append(_stage_summary(name, result, control=control))
                del result
            finally:
                _close(runtime)
                del runtime, stage
    require(hf is not None and eager is not None, "Missing HF/eager validation stage")
    require(cases and chats, "Empty validation fixtures")

    print("Stage: python_aoti", flush=True)
    runtime = load_aoti_runtime(packages, model_type=suite.model_type, specs=specs)
    try:
        compiled = run_stage(runtime, cases, chats, specs, suite.capture_states, suite.eos)
        lengths = (
            (2, 22, 64, 65, 96, 97, 98, 511, 512)
            if suite.model_type == "qwen35"
            else (1, 2, 7, 128, 511, 512)
        )
        seed_case = next(
            (case for case in cases if case["name"] == "model_prefill_decode"), cases[0]
        )
        report["state_contracts"] = _contract_probes(runtime, specs, seed_case["steps"][0], lengths)
    finally:
        runtime.close()
        del runtime
        torch.cuda.empty_cache()
    compare_traces(report, "python_aoti", "eager", compiled, eager, specs, numerical_policy)
    del eager
    token_keys = {
        (case["name"], step)
        for case in cases
        if case.get("origin", {}).get("suite") != "model"
        for step in range(len(case["steps"]))
    }
    anchor_policy = replace(policy, token_keys=token_keys)
    compare_traces(report, "python_aoti", "hf", compiled, hf, specs, anchor_policy)
    report["stages"].append(_stage_summary("python_aoti", compiled))

    print("Stage: cpp_aoti", flush=True)
    native_trace, native_report = run_native(
        native, packages, output, cases, chats, metadata, state_keys=suite.native_state_keys(cases)
    )
    require(
        set(native_trace.states) == suite.native_state_keys(cases), "Native state coverage differs"
    )
    failure_start = len(report["failures"])
    compare_traces(
        report,
        "cpp_aoti",
        "python_aoti",
        native_trace,
        compiled,
        specs,
        policy,
        exact=True,
        state_keys=suite.native_state_keys(cases),
        greedy_state_names={case["name"] for case in chats}
        if suite.model_type == "qwen35"
        else set(),
    )
    report["native_bitwise"] = len(report["failures"]) == failure_start
    report["native_execution"] = native_report
    report["stages"].append(_stage_summary("cpp_aoti", native_trace))
    report["state_coverage"] = dict(
        python_checkpoints=len(compiled.states), native_checkpoints=len(native_trace.states)
    )
    del native_trace

    print("Control: HF logits on compiled greedy trajectories", flush=True)
    _greedy_anchors(suite, report, hf, compiled, specs, policy, chats)
    print("Control: independent FP32 HF", flush=True)
    checked = 0
    fp32_cases = suite.fp32_cases(cases)
    require(fp32_cases, "Missing independent FP32 controls")
    require(len({case["name"] for case in fp32_cases}) == len(fp32_cases), "Duplicate FP32 control")
    require(
        all(case in cases for case in fp32_cases), "FP32 control inputs differ from the main chain"
    )
    with suite.reference(torch.float32) as runtime:
        for case in fp32_cases:
            # The control is compared one request at a time; completed control
            # snapshots can be released after their metrics have been recorded.
            control_case = {key: value for key, value in case.items() if key != "same_as"}
            control = teacher_trace(runtime, [control_case], specs, suite.capture_states)
            for label, actual in (("hf", hf), ("python_aoti", compiled)):
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
                    state_keys=set(control.states),
                    greedy_names=set(),
                )
            checked += len(control.logits)
            del control
    report["controls"] = dict(hf_fp32_steps=checked, matched_greedy_logits=True)
    report["passed"] = not report["failures"] and report["native_bitwise"]
