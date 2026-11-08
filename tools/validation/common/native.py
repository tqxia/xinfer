"""Run the shared native validator and normalize its model-specific trace layouts."""

import math
import os

import torch

from compiler.common.artifacts import read_json, require, write_json
from compiler.common.tests.tensors import bits_equal
from tools.common import prompt_chunks, run
from tools.validation.common.context import state_shape
from tools.validation.common.runner import Trace


def native_environment():
    """Preserve device selection while keeping workers independent of Python."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH")
    }
    env.update(PATH="/usr/bin:/bin", PYTHONHOME="/nonexistent-python-home")
    return env


def _check_replays(trace, cases, chats, specs):
    """Poison/reset invariance is strict even for numerically bounded backends."""
    for case in cases:
        if "same_as" not in case:
            continue
        for step in range(len(case["steps"])):
            key, original = (case["name"], step), (case["same_as"], step)
            require(original in trace.logits, "Missing native replay control")
            require(
                trace.positions[key] == trace.positions[original],
                "Native replay positions differ",
            )
            require(
                bits_equal(trace.logits[key], trace.logits[original]),
                "Native reset/future poison changed logits",
            )
            if key in trace.states:
                require(original in trace.states, "Missing native replay state control")
                require(
                    len(trace.states[key]) == len(trace.states[original]) == len(specs),
                    "Native replay state count differs",
                )
                position = trace.positions[key]
                for actual, expected, spec in zip(trace.states[key], trace.states[original], specs):
                    if spec["update"] == "inplace":
                        actual = actual[:, :, :position]
                        expected = expected[:, :, :position]
                    require(bits_equal(actual, expected), "Native replay changed valid state")
    for case in chats:
        if "same_as" not in case:
            continue
        require(case["same_as"] in trace.greedy, "Missing native generation replay control")
        actual, expected = trace.greedy[case["name"]], trace.greedy[case["same_as"]]
        require(
            actual["ids"] == expected["ids"] and actual["position"] == expected["position"],
            "Native reset/reuse changed generation",
        )
        require(
            len(actual["rows"]) == len(expected["rows"])
            and all(bits_equal(a, b) for a, b in zip(actual["rows"], expected["rows"])),
            "Native reset/reuse changed generation logits",
        )
        require(
            len(actual["state"]) == len(expected["state"]),
            "Native reset/reuse changed generation state count",
        )
        for a, b, spec in zip(actual["state"], expected["state"], specs):
            if spec["update"] == "inplace":
                a, b = a[:, :, : actual["position"]], b[:, :, : expected["position"]]
            require(bits_equal(a, b), "Native reset/reuse changed generation state")


def _map(path, dtype, count):
    width = torch.empty((), dtype=dtype).element_size()
    require(path.stat().st_size == count * width, f"Invalid native trace size: {path}")
    return torch.from_file(str(path), shared=False, size=count, dtype=dtype)


def _states(path, specs, position=0, compact=False):
    specs = [dict(spec, shape=state_shape(spec, position, compact)) for spec in specs]
    widths = [
        math.prod(spec["shape"])
        * torch.empty((), dtype=getattr(torch, spec["dtype"])).element_size()
        for spec in specs
    ]
    data = _map(path, torch.uint8, sum(widths))
    tensors, offset = [], 0
    for spec, width in zip(specs, widths):
        tensors.append(
            data[offset : offset + width].view(getattr(torch, spec["dtype"])).reshape(spec["shape"])
        )
        offset += width
    return tuple(tensors)


def run_native(binary, packages, output, cases, chats, metadata, *, state_keys=None):
    model = metadata["decode"]["model_type"]
    specs = metadata["decode"]["state"]
    vocab = metadata["decode"]["vocab_size"]
    eos = metadata["decode"]["eos_token_ids"]
    maximum = metadata.get("prefill", {}).get("token_range", [2, 512])[1]
    compiled = (
        model == "qwen3"
        and metadata["decode"].get("model_specific", {}).get("decode_backend") == "megaqwen"
    )
    manifest = sorted([*cases, *chats], key=lambda case: case.get("order", 1_000_000))
    require(len({case["name"] for case in manifest}) == len(manifest), "Duplicate native case")
    selected_states = state_keys is not None
    compact = selected_states and metadata["decode"].get("cache_length", 0) > 1024
    if selected_states:
        manifest = [
            dict(
                case,
                state_steps=[
                    step
                    for step in range(len(case.get("steps", [])))
                    if (case["name"], step) in state_keys
                ],
            )
            for case in manifest
        ]
    write_json(
        output / "cases.json", dict(cases=manifest, trace_format=2 if selected_states else 1)
    )
    destination = output / "native"
    command = [
        binary.resolve(),
        packages.resolve(),
        (output / "cases.json").resolve(),
        destination.resolve(),
    ]
    run(command, native_environment(), output / "native.log")
    native = read_json(destination / "native.json")
    require(
        native["loader_pairs"] == 1 and native["no_python_library"] and native["attention_library"],
        "Invalid native execution environment",
    )
    require(
        [row["name"] for row in native["cases"]] == [case["name"] for case in manifest],
        "Native case coverage differs",
    )
    rows = {row["name"]: row for row in native["cases"]}
    if compiled:
        require(
            native.get("backend_info", {}).get("backend") == "aoti"
            and native["backend_info"].get("compiled_decode_backend") == "megaqwen",
            "Wrong native compiled decode profile",
        )
    trace = Trace(compact_states=compact)
    for case in cases:
        name = case["name"]
        row = rows[name]
        lengths = [len(tokens) for tokens in case["steps"]]
        require(
            row["rows"] == len(lengths) and row["position"] == sum(lengths),
            "Invalid native position",
        )
        require(
            row["prefill_calls"] == sum(length > 1 for length in lengths)
            and row["decode_calls"] == lengths.count(1),
            "Incorrect native dispatch counts",
        )
        require(
            row["stable_kv_storage"]
            and row["write_region_checks"]
            and row["reset_check"]
            and row["kv_write_regions_preserved"]
            and row["non_kv_inputs_unchanged"],
            "Native state contract failed",
        )
        if case.get("check_overflow"):
            require(row["overflow_rejected_without_mutation"], "Native overflow mutated state")
        position = 0
        for step, length in enumerate(lengths):
            position += length
            trace.positions[name, step] = position
        if selected_states:
            indices = [step for step in range(len(lengths)) if (name, step) in state_keys]
            require(
                row["state_steps"] == indices and row["state_prefix"] == compact,
                "Native state checkpoints differ",
            )
            if model == "qwen3":
                logits = _map(
                    destination / f"{name}.f32", torch.float32, len(lengths) * vocab
                ).reshape(-1, 1, vocab)
            else:
                logits = _map(
                    destination / f"{name}.states", torch.bfloat16, len(lengths) * vocab
                ).reshape(-1, 1, vocab)
            for step, value in enumerate(logits):
                trace.logits[name, step] = value.float()
                if model != "qwen3":
                    trace.raw_logits[name, step] = value
            for step in indices:
                trace.states[name, step] = _states(
                    destination / f"{name}.{step}.states",
                    specs,
                    trace.positions[name, step],
                    compact,
                )
        elif model == "qwen3":
            logits = _map(destination / f"{name}.f32", torch.float32, len(lengths) * vocab).reshape(
                -1, 1, vocab
            )
            for step, value in enumerate(logits):
                trace.logits[name, step] = value
            if case.get("dump_cache"):
                trace.states[name, len(lengths) - 1] = _states(
                    destination / f"{name}.kv.bf16", specs
                )
        else:
            layouts = [dict(shape=[1, vocab], dtype="bfloat16"), *specs]
            widths = [
                math.prod(spec["shape"])
                * torch.empty((), dtype=getattr(torch, spec["dtype"])).element_size()
                for spec in layouts
            ]
            data = _map(destination / f"{name}.states", torch.uint8, len(lengths) * sum(widths))
            offset = 0
            for step in range(len(lengths)):
                tensors = []
                for spec, width in zip(layouts, widths):
                    tensors.append(
                        data[offset : offset + width]
                        .view(getattr(torch, spec["dtype"]))
                        .reshape(spec["shape"])
                    )
                    offset += width
                trace.raw_logits[name, step] = tensors[0]
                trace.logits[name, step] = tensors[0].float()
                trace.states[name, step] = tuple(tensors[1:])
    for case in chats:
        name = case["name"]
        row = rows[name]
        ids = row["ids"]
        require(0 < len(ids) <= case["limit"], "Invalid native generation length")
        stop_tokens = case.get("eos", eos)
        require(
            not any(token in stop_tokens for token in ids[:-1]),
            "Native generation continued past EOS",
        )
        stop = "eos" if ids[-1] in stop_tokens else "limit"
        require(
            row["stop"] == stop and (stop == "eos" or len(ids) == case["limit"]),
            "Invalid native stop reason",
        )
        require(
            row["position"] == len(case["prompt"]) + len(ids) - 1, "Invalid generation position"
        )
        lengths = list(map(len, prompt_chunks(case["prompt"], maximum))) + [1] * (len(ids) - 1)
        require(
            row["prefill_calls"] == sum(length > 1 for length in lengths)
            and row["decode_calls"] == lengths.count(1),
            "Incorrect generation dispatch",
        )
        logits = _map(destination / f"{name}.f32", torch.float32, len(ids) * vocab).reshape(
            -1, 1, vocab
        )
        trace.greedy[name] = dict(
            ids=ids,
            rows=list(logits),
            position=row["position"],
            state=_states(destination / f"{name}.states", specs, row["position"], compact)
            if model == "qwen35"
            else (),
        )
    _check_replays(trace, cases, chats, specs)
    return trace, native
