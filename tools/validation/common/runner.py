"""Shared state contracts and matched-input execution for validation stages."""

from dataclasses import dataclass, field

import torch

from compiler.common.artifacts import require
from compiler.common.metadata import read_package_metadata
from compiler.common.tests.tensors import bits_equal, cpu_copy
from tools.validation.common.validation import disk_snapshot
from compiler.common.weights import SharedWeights
from tools.common import prompt_chunks
from tools.packages import validate_model_metadata, validate_shared_weights


@dataclass
class Trace:
    logits: dict = field(default_factory=dict)
    states: dict = field(default_factory=dict)
    greedy: dict = field(default_factory=dict)
    raw_logits: dict = field(default_factory=dict)
    positions: dict = field(default_factory=dict)
    compact_states: bool = False


@dataclass
class Stage:
    name: str
    runtime: object


class ExplicitRuntime:
    """Normalize logits-only and returned-state entry points using package metadata."""

    def __init__(
        self,
        functions,
        specs,
        vocab_size,
        capacity,
        *,
        logits_only=False,
        owner=None,
        require_write=False,
        max_prefill=512,
    ):
        self.functions = functions
        self.specs = specs
        self.vocab_size = vocab_size
        self.capacity = capacity
        self.logits_only = logits_only
        self.owner = owner
        self.require_write = require_write
        self.max_prefill = max_prefill
        self.position = 0
        self.positions = [0] * specs[0]["shape"][0]
        self.state = []

    def reset(self, fill=0):
        if not self.state:
            self.state = [
                torch.zeros(s["shape"], device="cuda", dtype=getattr(torch, s["dtype"]))
                for s in self.specs
            ]
        for tensor, spec in zip(self.state, self.specs):
            value = fill.get(spec["kind"], 0) if isinstance(fill, dict) else fill
            tensor.fill_(value if spec["update"] == "inplace" else 0)
        self.position = 0
        self.positions = [0] * self.specs[0]["shape"][0]

    def step(self, tokens):
        return self.step_batch([tokens])

    def step_batch(self, rows, *, padding_token=0):
        batch_size = len(rows)
        require(1 <= batch_size <= self.specs[0]["shape"][0], "Batch size outside package range")
        counts = [len(row) for row in rows]
        length = max(counts)
        tokens = [token for row in rows for token in row]
        require(1 <= length <= self.max_prefill, f"Expected 1..{self.max_prefill} input tokens")
        ends = [start + count for start, count in zip(self.positions, counts)]
        require(all(end <= self.capacity for end in ends), "Context capacity exceeded")
        require(all(end > 0 for end in ends), "Each row must start with tokens")
        require(all(type(t) is int and 0 <= t < self.vocab_size for t in tokens), "Invalid token")
        ids = torch.tensor(
            [row + [padding_token] * (length - len(row)) for row in rows],
            dtype=torch.int64,
            device="cuda",
        )
        positions = torch.tensor(
            [
                list(range(start, end)) + [0] * (length - count)
                for start, end, count in zip(self.positions, ends, counts)
            ],
            dtype=torch.int64,
            device="cuda",
        )
        attention_inputs = (
            torch.tensor(ends, dtype=torch.int32),
            torch.tensor(counts, dtype=torch.int32),
        )
        before = tuple(t.clone() for t in (ids, positions, *self.state))
        pointers = [t.data_ptr() for t in self.state]
        result = self.functions["decode" if length == 1 else "prefill"](
            ids,
            positions,
            tuple(t[:batch_size] for t in self.state)
            if self.logits_only
            else [t[:batch_size] for t in self.state],
            *attention_inputs,
        )
        if self.logits_only:
            logits, state = result, [t[:batch_size] for t in self.state]
        else:
            require(len(result) == len(self.specs) + 1, "Incorrect output/state count")
            logits, state = result[0], list(result[1:])
        require(logits.shape == (batch_size, self.vocab_size), "Incorrect logit shape")
        require(logits.dtype == torch.bfloat16, "Incorrect logit dtype")
        require(
            bits_equal(ids, before[0]) and bits_equal(positions, before[1]),
            "Token/position input mutated",
        )
        require(len(state) == len(self.specs), "Incorrect state count")
        for index, (incoming, output, original, spec) in enumerate(
            zip(self.state, state, before[2:], self.specs)
        ):
            require(list(output.shape) == [batch_size, *spec["shape"][1:]], "Incorrect state shape")
            require(output.dtype == getattr(torch, spec["dtype"]), "Incorrect state dtype")
            require(
                output.device == incoming.device and output.is_contiguous(), "Invalid state layout"
            )
            if spec["update"] == "inplace":
                require(
                    output.data_ptr() == pointers[index] and output.stride() == incoming.stride(),
                    "KV output lost caller-owned storage",
                )
                require(
                    bits_equal(incoming[batch_size:], original[batch_size:]),
                    "Unused state row overwritten",
                )
                for row, (start, end) in enumerate(zip(self.positions, ends)):
                    require(
                        bits_equal(output[row, :, :start], original[row, :, :start]),
                        "Past KV overwritten",
                    )
                    require(
                        bits_equal(output[row, :, end:], original[row, :, end:]),
                        "Future KV overwritten",
                    )
                    if self.require_write and start != end:
                        require(
                            not bits_equal(output[row, :, start:end], original[row, :, start:end]),
                            "Cache update missing",
                        )
                state[index] = incoming
            else:
                require(bits_equal(incoming, original), "Functional input state mutated")
                require(output.data_ptr() != incoming.data_ptr(), "Functional output aliases input")
                state[index] = (
                    torch.cat((output, incoming[batch_size:]), dim=0)
                    if batch_size < incoming.shape[0]
                    else output
                )
            for row, count in enumerate(counts):
                if count == 0:
                    require(bits_equal(state[index][row], original[row]), "Inactive state changed")
        for row, count in enumerate(counts):
            if count == 0:
                require(
                    torch.count_nonzero(logits[row]).item() == 0, "Inactive logits must be zero"
                )
        require(all(bool(torch.isfinite(t).all()) for t in (logits, *state)), "Nonfinite output")
        self.state = list(state)
        self.positions[:batch_size] = ends
        self.position = max(self.positions)
        return logits

    def close(self):
        self.functions = {}
        self.owner = None
        self.state = []


def load_aoti_runtime(packages, *, model_type, specs):
    """Validate the bundle before loading its operators and shared GPU weights."""
    pair = read_package_metadata(packages, model_type=model_type)
    for entry in pair.values():
        validate_model_metadata(entry)
    decode = pair["decode"]
    validate_shared_weights(packages, pair["prefill"], decode)
    require(specs == decode["state"], "Runtime state descriptors differ from package")
    weights = SharedWeights(packages, decode["shared_weights"])
    return ExplicitRuntime(
        {kind: weights.load(kind) for kind in ("prefill", "decode")},
        specs,
        decode["vocab_size"],
        decode["cache_length"],
        logits_only=model_type == "qwen3",
        owner=weights,
        require_write=model_type == "qwen3",
        max_prefill=pair["prefill"]["token_range"][1],
    )


def _snapshot(runtime, specs):
    state = list(runtime.state)
    require(len(state) == len(specs), "Incorrect state count")
    for tensor, spec in zip(state, specs):
        require(list(tensor.shape) == spec["shape"], "Incorrect state shape")
    return cpu_copy(state)


def capture_state(runtime, specs, fill=0):
    """Copy valid KV only; unchanged future slots are checked independently."""
    compact = getattr(runtime, "capacity", 0) > 1024
    require(len(runtime.state) == len(specs), "Incorrect state count")
    tensors = []
    for tensor, spec in zip(runtime.state, specs):
        require(list(tensor.shape) == spec["shape"], "Incorrect state shape")
        require(bool(torch.isfinite(tensor).all()), "Nonfinite state")
        if compact and spec["update"] == "inplace":
            future = tensor[:, :, runtime.position :]
            value = fill.get(spec["kind"], 0) if isinstance(fill, dict) else fill
            require(bits_equal(future, torch.full_like(future, value)), "Future KV overwritten")
            tensor = tensor[:, :, : runtime.position]
        tensors.append(tensor)
    return cpu_copy(tensors)


def teacher_trace(runtime, cases, specs, capture_states, *, state_directory=None):
    result = Trace(compact_states=getattr(runtime, "capacity", 0) > 1024)
    if state_directory is not None:
        state_directory.mkdir(parents=True, exist_ok=False)
    names = set()
    for case in cases:
        name = case["name"]
        require(name not in names, "Duplicate validation case")
        names.add(name)
        require(case["steps"], "Empty validation case")
        fill = case.get("poison_kv", case.get("fill", 0))
        runtime.reset(fill)
        require(runtime.position == 0, "Reset did not clear position")
        position = 0
        for step, tokens in enumerate(case["steps"]):
            require(tokens, "Empty validation step")
            logits = runtime.step(tokens)
            position += len(tokens)
            require(runtime.position == position, "Incorrect position advance")
            require(
                all(bool(torch.isfinite(tensor).all()) for tensor in runtime.state),
                "Nonfinite state",
            )
            key = name, step
            result.positions[key] = position
            result.raw_logits[key] = cpu_copy((logits,))[0]
            result.logits[key] = cpu_copy((logits.float(),))[0]
            if capture_states(case, step):
                state = capture_state(runtime, specs, fill)
                result.states[key] = (
                    disk_snapshot(state_directory / f"{name}.{step}.states", state)
                    if state_directory is not None
                    else state
                )
                del state
            if "same_as" in case:
                original = case["same_as"], step
                require(original in result.logits, "Missing clean/repeated case")
                require(
                    bits_equal(result.logits[key], result.logits[original]),
                    "Reset/poison changed logits",
                )
                if key in result.states:
                    require(original in result.states, "Missing clean/repeated state")
                    for actual, expected, spec in zip(
                        result.states[key], result.states[original], specs
                    ):
                        if spec["update"] == "inplace":
                            require(
                                bits_equal(actual[:1, :, :position], expected[:1, :, :position]),
                                "Reset/poison changed valid KV",
                            )
                            require(
                                bool((actual[1:] == fill).all()), "Unused state row overwritten"
                            )
                            if not result.compact_states:
                                require(
                                    bool((actual[:, :, position:] == fill).all()),
                                    "Future KV overwritten",
                                )
                        else:
                            require(
                                bits_equal(actual, expected),
                                "Reset/poison changed functional state",
                            )
        if case.get("check_overflow"):
            before = _snapshot(runtime, specs)
            pointers = [t.data_ptr() for t in runtime.state]
            try:
                runtime.step([1])
            except (ValueError, RuntimeError, AssertionError):
                pass
            else:
                raise ValueError("Context overflow was accepted")
            require(runtime.position == position, "Overflow changed position")
            require(
                pointers == [t.data_ptr() for t in runtime.state], "Overflow changed state storage"
            )
            require(
                all(bits_equal(a, b) for a, b in zip(_snapshot(runtime, specs), before)),
                "Overflow changed state",
            )
    return result


def greedy(runtime, prompt, limit, eos):
    require(prompt and type(limit) is int and limit > 0, "Invalid generation request")
    require(all(type(token) is int and token >= 0 for token in prompt), "Invalid token")
    if hasattr(runtime, "vocab_size"):
        require(all(token < runtime.vocab_size for token in prompt), "Invalid token")
    require(
        len(prompt) + limit <= getattr(runtime, "capacity", len(prompt) + limit),
        "Generation budget exceeds context",
    )
    runtime.reset()
    require(runtime.position == 0, "Reset did not clear position")
    tokens, ids, rows = prompt, [], []
    position = 0
    for index in range(limit):
        for chunk in prompt_chunks(tokens, getattr(runtime, "max_prefill", 512)):
            logits = runtime.step(chunk)
            position += len(chunk)
            require(runtime.position == position, "Incorrect generation position")
        rows.append(cpu_copy((logits.float(),))[0])
        token = logits.argmax(-1).item()
        ids.append(token)
        if token in eos or index + 1 == limit:
            break
        tokens = [token]
    specs = getattr(runtime, "specs", None)
    state = capture_state(runtime, specs) if specs is not None else cpu_copy(runtime.state)
    return dict(ids=ids, rows=rows, state=state, position=runtime.position)


def run_stage(runtime, cases, chats, specs, capture_states, eos, *, state_directory=None):
    result = teacher_trace(runtime, cases, specs, capture_states, state_directory=state_directory)
    for case in chats:
        name = case["name"]
        require(name not in result.greedy, "Duplicate generation case")
        value = greedy(runtime, case["prompt"], case["limit"], case.get("eos", eos))
        if "same_as" in case:
            require(case["same_as"] in result.greedy, "Missing repeated generation")
            previous = result.greedy[case["same_as"]]
            require(value["ids"] == previous["ids"], "Repeated generation changed tokens")
            require(
                len(value["rows"]) == len(previous["rows"])
                and all(bits_equal(a, b) for a, b in zip(value["rows"], previous["rows"])),
                "Repeated generation changed logits",
            )
            require(
                value["position"] == previous["position"], "Repeated generation changed position"
            )
            require(
                len(value["state"]) == len(previous["state"])
                and all(
                    bits_equal(actual, expected)
                    for actual, expected in zip(value["state"], previous["state"])
                ),
                "Repeated generation changed state",
            )
        result.greedy[name] = value
    return result
