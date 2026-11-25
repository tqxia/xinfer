"""Execute one HF → eager → Python AOTI → C++ AOTI validation chain."""

from contextlib import closing
from pathlib import Path

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
                    bits_equal(actual[:1, :, :length], expected[:1, :, :length]),
                    "Poison changes valid KV",
                )
                fill = 7 if spec["kind"] == "key" else -3
                require(bool((actual[:, :, length:] == fill).all()), "Future KV overwritten")
                require(bool((actual[1:, :, :length] == fill).all()), "Unused KV row overwritten")
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
        with suite.reference() as runtime:
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
            policy,
            state_keys=set(),
            greedy_names=set(),
        )


def _describe_validation(suite, policy, native, report):
    report["reference_protocol"] = dict(
        dtype="bfloat16", cache="HF StaticCache", teacher_inputs="matched tokens and partitions"
    )
    report["reference_protocol"].update(getattr(suite, "reference_protocol", {}))
    report["state_snapshot_policy"] = dict(
        kv="populated prefix; unused future fill checked independently on capture"
        if suite.capacity > 1024
        else "full cache",
        functional_states="complete",
        long_context="completed prompt and final request checkpoints; every logit retained",
        native_write_and_ownership_checks="every call",
    )
    report["numerical_policy"] = dict(
        name="shared_bf16",
        version=1,
        limits=dict(relative_l2=policy.limit, peak_relative=policy.limit),
        logit_errors="required",
        state_errors="diagnostic",
        token_differences="diagnostic",
        greedy_differences="diagnostic",
        native_python_aoti="bitwise",
        runtime_contracts="required",
    )
    report["native_sha256"] = sha256(native)


def _run_eager_stages(suite, specs, policy, report, output):
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
                    state_directory=output / name,
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
                        policy,
                        logit_keys=set(result.logits),
                        state_keys=set(result.states),
                        greedy_names=set(),
                    )
                else:
                    require(name == "eager", "Unknown eager stage")
                    eager = result
                    compare_traces(report, "eager", "hf", eager, hf, specs, policy)
                report["stages"].append(_stage_summary(name, result, control=control))
                del result
            finally:
                _close(runtime)
                del runtime, stage
    require(hf is not None and eager is not None, "Missing HF/eager validation stage")
    require(cases and chats, "Empty validation fixtures")
    return hf, eager, cases, chats


def _run_python_aoti(suite, packages, cases, chats, specs, report, output):
    print("Stage: python_aoti", flush=True)
    runtime = load_aoti_runtime(packages, model_type=suite.model_type, specs=specs)
    try:
        compiled = run_stage(
            runtime,
            cases,
            chats,
            specs,
            suite.capture_states,
            suite.eos,
            state_directory=output / "python_aoti",
        )
        lengths = (
            (2, 22, 64, 65, 96, 97, 98, 511, 512)
            if suite.model_type in ("qwen35", "qwen38")
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
    return compiled


def _compare_native(
    suite, packages, native, output, metadata, cases, chats, compiled, specs, policy, report
):
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
        if suite.model_type in ("qwen35", "qwen38")
        else set(),
    )
    report["native_bitwise"] = len(report["failures"]) == failure_start
    report["native_execution"] = native_report
    report["stages"].append(_stage_summary("cpp_aoti", native_trace))
    report["state_coverage"] = dict(
        python_checkpoints=len(compiled.states), native_checkpoints=len(native_trace.states)
    )
    del native_trace


@torch.inference_mode()
def validate_chain(suite, packages, native, output, metadata, report):
    output = Path(output)
    specs = suite.specs
    policy = Policy(state_rows=getattr(suite, "state_rows", None))
    _describe_validation(suite, policy, native, report)
    hf, eager, cases, chats = _run_eager_stages(suite, specs, policy, report, output)
    # Repeated-generation state checks finish inside run_stage. Subsequent
    # HF/eager comparisons use their greedy logits and tokens, not these caches.
    for generation in (*hf.greedy.values(), *eager.greedy.values()):
        del generation["state"]
    compiled = _run_python_aoti(suite, packages, cases, chats, specs, report, output)
    compare_traces(report, "python_aoti", "eager", compiled, eager, specs, policy)
    del eager
    compare_traces(report, "python_aoti", "hf", compiled, hf, specs, policy)
    report["stages"].append(_stage_summary("python_aoti", compiled))

    _compare_native(
        suite, packages, native, output, metadata, cases, chats, compiled, specs, policy, report
    )

    print("Control: HF logits on compiled greedy trajectories", flush=True)
    _greedy_anchors(suite, report, hf, compiled, specs, policy, chats)
    report["controls"] = dict(matched_greedy_logits=True)
    report["passed"] = not report["failures"] and report["native_bitwise"]
