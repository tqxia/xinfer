"""Run the shared native validator and normalize its model-specific trace layouts."""

import math
import os

import torch

from compiler.common.artifacts import read_json, require, write_json
from tools.common import run
from tools.validation.common.runner import Trace


def _map(path, dtype, count):
    width = torch.empty((), dtype=dtype).element_size()
    require(path.stat().st_size == count * width, f"Invalid native trace size: {path}")
    return torch.from_file(str(path), shared=False, size=count, dtype=dtype)


def _states(path, specs):
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


def run_native(binary, packages, output, cases, chats, metadata):
    model = metadata["decode"]["model_type"]
    specs = metadata["decode"]["state"]
    vocab = metadata["decode"]["vocab_size"]
    eos = metadata["decode"]["eos_token_ids"]
    manifest = sorted([*cases, *chats], key=lambda case: case.get("order", 1_000_000))
    require(len({case["name"] for case in manifest}) == len(manifest), "Duplicate native case")
    write_json(output / "cases.json", dict(cases=manifest))
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH")
    }
    env.update(PATH="/usr/bin:/bin", PYTHONHOME="/nonexistent-python-home")
    destination = output / "native"
    run(
        [
            binary.resolve(),
            packages.resolve(),
            (output / "cases.json").resolve(),
            destination.resolve(),
        ],
        env,
        output / "native.log",
    )
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
    trace = Trace()
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
        if model == "qwen3":
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
        lengths = [len(case["prompt"])] + [1] * (len(ids) - 1)
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
            state=_states(destination / f"{name}.states", specs) if model == "qwen35" else (),
        )
    return trace, native
