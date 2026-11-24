"""Shared dynamic-batch fixtures, numerical comparisons and native parity checks."""

import gc
import json
import shutil
import subprocess

import torch

from compiler.common.artifacts import read_json, require, write_json
from tools.prepare.common import encode_prompt
from tools.validation.common.runner import capture_state
from tools.validation.common.validation import (
    bits_equal,
    disk_snapshot,
    tensor_metrics,
    record_failure,
    write_trace,
)
from tools.common import native_environment


def case_groups(tokenizer, capacity):
    texts = [
        "What is the capital of France?",
        "Explain how photosynthesis helps plants grow, in one short sentence.",
        "Name the largest planet.",
        "How does a bicycle stay balanced while moving?",
        "List two colors found in a rainbow.",
        "Describe why water freezes when it becomes cold.",
        "What is seven times eight?",
        "Give one practical way to reduce household electricity consumption.",
    ]
    prompts = [encode_prompt(tokenizer, text) for text in texts]
    require(len(prompts[0]) != len(prompts[1]), "Unexpected prompt lengths")
    continuation = [
        tokenizer.encode(text, add_special_tokens=False)[:3]
        for text in ["Paris is beautiful.", "Rome is beautiful."] * 4
    ]
    require(all(len(row) == 3 for row in continuation), "Expected three teacher tokens")
    groups = {}
    for batch_size in range(1, 9):
        steps = [prompts[:batch_size]] + [
            [[row[i]] for row in continuation[:batch_size]] for i in range(3)
        ]
        groups[f"b{batch_size}"] = (
            texts[:batch_size],
            common_cases(steps, capacity) + multi_row_cases(steps, capacity),
        )
    groups["transitions"] = (texts, batch_transition_cases(prompts, continuation))
    for _, cases in groups.values():
        for case in cases:
            case["state_steps"] = [len(case["batch_steps"]) - 1]
    return groups


def common_cases(steps, capacity):
    prompts = steps[0]
    cases = [
        # Baseline: real prompts followed by three teacher-forced decode steps.
        dict(name="prompts", batch_steps=steps),
        # Repeat after reset; logits and populated KV must match the baseline exactly.
        dict(name="reset", batch_steps=steps),
        # Fill KV with 17; unwritten slots must not affect logits or populated KV.
        dict(name="poison", batch_steps=steps, fill=17),
    ]
    # 1/2: decode/minimum prefill; 513: cross a chunk boundary with a one-token tail.
    # Full capacity: reject the next call without changing cache or positions.
    for length in sorted({1, 2, min(513, capacity), capacity}):
        rows = [(row * (length // len(row) + 1))[:length] for row in prompts]
        chunks = [[row[start : start + 512] for row in rows] for start in range(0, length, 512)]
        if length < capacity:
            chunks += steps[1 : 1 + min(3, capacity - length)]
        cases.append(
            dict(name=f"length_{length}", batch_steps=chunks, check_overflow=length == capacity)
        )
    # Padding token IDs must not affect valid logits or populated KV.
    cases.append(dict(name="padding", batch_steps=steps, padding_token=17))
    return cases


def multi_row_cases(steps, capacity):
    prompts = steps[0]
    batch_size = len(prompts)
    if batch_size == 1:
        return []
    cases = [
        # Change row zero only; all other rows' logits and KV must stay identical.
        dict(name="perturb", batch_steps=[[list(reversed(prompts[0])), *prompts[1:]]] + steps[1:])
    ]
    for pattern in (
        (1, 2),  # One real token in a padded two-token prefill.
        (511, 512),  # Unequal lengths within one full prefill call.
        (512, 513),  # Only the longer row consumes a second chunk.
        (19, 513),  # Short row waits through a second chunk, then resumes decode.
        (513, 19),  # Repeat with the long row first.
        (19, min(900, capacity - 3)),  # Large length gap with room for three decode steps.
        (min(900, capacity - 3), 19),  # Repeat with the long row first.
        (capacity, capacity - 7),  # A full row rejects overflow despite space in another row.
        (capacity - 7, capacity),  # Repeat with the second row full.
    ):
        lengths = tuple(pattern[row % len(pattern)] for row in range(batch_size))
        if max(lengths) > capacity:
            continue
        rows = [(row * (length // len(row) + 1))[:length] for row, length in zip(prompts, lengths)]
        chunks = [
            [row[start : start + 512] for row in rows] for start in range(0, max(lengths), 512)
        ]
        if max(lengths) + 3 <= capacity:
            chunks += steps[1:]
        cases.append(
            dict(
                name=f"lengths_{lengths[0]}_{lengths[1]}",
                batch_steps=chunks,
                check_overflow=max(lengths) == capacity,
            )
        )
    if batch_size > 2:
        # Advance one row at a time from distinct positions; idle rows retain their state.
        rows = [
            (prompt * (600 // len(prompt) + 1))[: min(19 + 73 * row, capacity - 1)]
            for row, prompt in enumerate(prompts)
        ]
        chunks = [
            [row[start : start + 512] for row in rows]
            for start in range(0, max(map(len, rows)), 512)
        ]
        chunks += [
            [steps[1][row] if row == active else [] for row in range(batch_size)]
            for active in range(batch_size)
        ]
        cases.append(dict(name="staggered_resume", batch_steps=chunks))
    return cases


def batch_transition_cases(prompts, continuation):
    return [
        # 8->1->3->5->8: omitted rows preserve their cache and resume when restored.
        dict(
            name="changing_batch",
            batch_steps=[
                prompts,
                [[continuation[0][0]]],
                [row[:2] for row in continuation[:3]],
                [[row[0]] for row in continuation[:5]],
                [[row[1]] for row in continuation],
            ],
        ),
        # 1->3->8: new rows start fresh while existing rows continue from their caches.
        dict(
            name="growing_batch",
            batch_steps=[
                prompts[:1],
                [[continuation[0][0]], *prompts[1:3]],
                [[row[1]] for row in continuation[:3]] + prompts[3:],
            ],
        ),
    ]


def collect_explicit(runtime, case):
    runtime.reset(case.get("fill", 0))
    logits = [
        runtime.step_batch(rows, padding_token=case.get("padding_token", 0)).cpu()
        for rows in case["batch_steps"]
    ]
    state = [
        (
            tensor[row : row + 1, :, :position]
            if spec["update"] == "inplace"
            else tensor[row : row + 1]
        ).cpu()
        for tensor, spec in zip(runtime.state, runtime.specs)
        for row, position in enumerate(runtime.positions[: max(map(len, case["batch_steps"]))])
    ]
    return logits, state


def compare_stage(report, name, stage, actual, reference, *, logits):
    require(len(actual) == len(reference), "Stage comparison coverage differs")
    for index, (a, b) in enumerate(zip(actual, reference)):
        metric = tensor_metrics(
            a,
            b,
            logits=logits,
            rows=(a.float(), b.float()) if logits else (a.float().flatten(1), b.float().flatten(1)),
        )
        context = dict(case=name, stage=stage, kind="logits" if logits else "state", index=index)
        report["comparisons"].append(dict(**context, required=logits, **metric))
        record_failure(report["failures"], metric, required=logits, **context)


def _release_stage(runtime):
    if hasattr(runtime, "close"):
        runtime.close()
    del runtime
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _snapshot_paths(staging, stage, group, name):
    return staging / f"{stage}_{group}_{name}.logits", staging / f"{stage}_{group}_{name}.state"


def validate_python_stages(
    args,
    groups,
    meta,
    *,
    load_reference,
    collect_reference,
    load_eager,
    load_aoti,
):
    """Run HF, eager and Python AOTI in separate stages so only one GPU model stays resident."""
    staging = args.output / "_staging"
    staging.mkdir(parents=True, exist_ok=True)
    expected_dir = args.output / "expected"
    reports = {}
    for group, (texts, cases) in groups.items():
        output = args.output / group
        output.mkdir()
        write_json(output / "cases.json", dict(trace_format=2, cases=cases))
        reports[group] = dict(
            batch_size=len(texts),
            prompts=texts,
            prompt_tokens=[len(row) for row in cases[0]["batch_steps"][0]],
            numerical_policy=dict(name="shared_bf16", version=1),
            comparisons=[],
            failures=[],
            structural_checks=[],
            parity_digests={},
            cases=[],
        )

    hf_traces, eager_traces = {}, {}
    try:
        reference = load_reference()
        try:
            for group, (_, cases) in groups.items():
                for case in cases:
                    name = case["name"]
                    print(f"{group}: collecting hf {name}", flush=True)
                    hf_logits, hf_state = collect_reference(reference, case["batch_steps"])
                    logits_path, state_path = _snapshot_paths(staging, "hf", group, name)
                    hf_traces[group, name] = (
                        disk_snapshot(logits_path, hf_logits),
                        disk_snapshot(state_path, hf_state),
                    )
        finally:
            _release_stage(reference)
            del reference

        eager = load_eager()
        try:
            for group, (_, cases) in groups.items():
                for case in cases:
                    name = case["name"]
                    print(f"{group}: collecting eager {name}", flush=True)
                    eager_logits, eager_state = collect_explicit(eager, case)
                    logits_path, state_path = _snapshot_paths(staging, "eager", group, name)
                    eager_traces[group, name] = (
                        disk_snapshot(logits_path, eager_logits),
                        disk_snapshot(state_path, eager_state),
                    )
        finally:
            _release_stage(eager)
            del eager

        aoti = load_aoti()
        try:
            for group, (texts, cases) in groups.items():
                output = args.output / group
                report = reports[group]
                batch_size = len(texts)
                base_logits = None
                base_row_state = None
                for case in cases:
                    name = case["name"]
                    print(f"{group}: validating {name}", flush=True)
                    actual_logits, actual_state = collect_explicit(aoti, case)
                    hf_logits, hf_state = hf_traces.pop((group, name))
                    eager_logits, eager_state = eager_traces.pop((group, name))
                    stages = [
                        ("eager/hf", (eager_logits, eager_state), (hf_logits, hf_state)),
                        ("aoti/eager", (actual_logits, actual_state), (eager_logits, eager_state)),
                        ("aoti/hf", (actual_logits, actual_state), (hf_logits, hf_state)),
                    ]
                    for stage, actual, reference_data in stages:
                        compare_stage(
                            report, name, stage, actual[0], reference_data[0], logits=True
                        )
                        compare_stage(
                            report, name, stage, actual[1], reference_data[1], logits=False
                        )
                    del hf_logits, hf_state, eager_logits, eager_state
                    for stage_name in ("hf", "eager"):
                        for path in _snapshot_paths(staging, stage_name, group, name):
                            path.unlink(missing_ok=True)
                    if name == "prompts":
                        base_logits = actual_logits
                        base_row_state = [tensor.clone() for tensor in actual_state]
                    elif name in ("reset", "poison", "perturb", "padding"):
                        for current, base in zip(actual_logits, base_logits):
                            require(
                                bits_equal(
                                    current[1:] if name == "perturb" else current,
                                    base[1:] if name == "perturb" else base,
                                ),
                                f"{name} changed outputs",
                            )
                        for index, (current, base) in enumerate(zip(actual_state, base_row_state)):
                            if name != "perturb" or index % batch_size != 0:
                                require(
                                    bits_equal(current, base),
                                    f"{name} changed unaffected state row {index % batch_size}",
                                )
                        report["structural_checks"].append(name)
                    stem = f"{group}_{name}"
                    report["parity_digests"][name] = {
                        ".states": write_trace(
                            expected_dir / f"{stem}.{case['state_steps'][0]}.states",
                            capture_state(aoti, meta["state"], case.get("fill", 0)),
                        ),
                        ".f32": write_trace(
                            expected_dir / f"{stem}.f32", (t.float() for t in actual_logits)
                        ),
                    }
                    if case.get("check_overflow"):
                        before = [tensor.clone() for tensor in aoti.state]
                        positions = list(aoti.positions)
                        try:
                            aoti.step_batch([[1] for _ in range(batch_size)])
                        except ValueError as error:
                            require("capacity" in str(error), str(error))
                        else:
                            raise ValueError("Overflow accepted")
                        require(aoti.positions == positions, "Overflow changed position")
                        require(
                            all(bits_equal(a, b) for a, b in zip(aoti.state, before)),
                            "Overflow mutated state",
                        )
                        report["structural_checks"].append("overflow")
                        del before
                    report["cases"].append(
                        dict(
                            name=name,
                            position=aoti.position,
                            positions=list(aoti.positions),
                            steps=len(actual_logits),
                        )
                    )
                    write_json(output / "report.json", report)
        finally:
            _release_stage(aoti)
            del aoti
    finally:
        hf_traces.clear()
        eager_traces.clear()
        shutil.rmtree(staging, ignore_errors=True)
    return list(groups)


def validate_native(args, groups):
    cases = []
    for group in groups:
        manifest = read_json(args.output / group / "cases.json")
        cases.extend(dict(case, name=f"{group}_{case['name']}") for case in manifest["cases"])
    expected_dir = args.output / "expected"
    write_json(
        args.output / "cases.json",
        dict(
            trace_format=2,
            cases=cases,
            expected_directory=str(expected_dir.resolve()),
        ),
    )
    subprocess.run(
        [
            str(args.native.resolve()),
            str(args.packages.resolve()),
            str((args.output / "cases.json").resolve()),
            str((args.output / "native").resolve()),
        ],
        env=native_environment(),
        check=True,
    )
    native = read_json(args.output / "native/native.json")
    require(
        native["loader_pairs"] == 1 and native["no_python_library"] and native["attention_library"],
        "Invalid native execution environment",
    )
    require(
        [row["name"] for row in native["cases"]] == [case["name"] for case in cases],
        "Native case coverage differs",
    )
    rows = {row["name"]: row for row in native["cases"]}
    results = {}
    for group in groups:
        output = args.output / group
        report = read_json(output / "report.json")
        selected = []
        for expected in report["cases"]:
            row = rows[f"{group}_{expected['name']}"]
            require(row["positions"] == expected["positions"], "Native per-row positions differ")
            require(row["bitwise_parity"], "Native parity check missing")
            selected.append(dict(row, name=expected["name"]))
        report["native"] = dict(native, cases=selected)
        report["native_bitwise_parity"] = True
        write_json(output / "report.json", report)
        results[group] = dict(required_failures=len(report["failures"]))
    # Discard large expected state dumps only after every native parity check passes.
    for case in cases:
        for step in case.get("state_steps", ()):
            (expected_dir / f"{case['name']}.{step}.states").unlink(missing_ok=True)
    write_json(args.output / "summary.json", results)
    print(json.dumps(results))
    return results
