"""Shared state contracts and matched-input execution for validation stages."""

from dataclasses import dataclass, field
from importlib import import_module

import torch

from compiler.common.artifacts import require
from compiler.common.attention import cache_extent, load_attention_ops
from compiler.common.metadata import read_package_metadata
from compiler.common.tests.tensors import bits_equal, cpu_copy
from compiler.common.weights import SharedWeights, validate_shared_weights


@dataclass
class Trace:
    logits: dict = field(default_factory=dict)
    states: dict = field(default_factory=dict)
    greedy: dict = field(default_factory=dict)
    raw_logits: dict = field(default_factory=dict)


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
    ):
        self.functions = functions
        self.specs = specs
        self.vocab_size = vocab_size
        self.capacity = capacity
        self.logits_only = logits_only
        self.owner = owner
        self.require_write = require_write
        self.position = 0
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

    def step(self, tokens):
        length = len(tokens)
        require(1 <= length <= 512, "Expected 1..512 input tokens")
        require(self.position + length <= self.capacity, "Context capacity exceeded")
        require(all(type(t) is int and 0 <= t < self.vocab_size for t in tokens), "Invalid token")
        ids = torch.tensor([tokens], dtype=torch.int64, device="cuda")
        positions = torch.arange(self.position, self.position + length, device="cuda")
        before = tuple(t.clone() for t in (ids, positions, *self.state))
        pointers = [t.data_ptr() for t in self.state]
        result = self.functions["decode" if length == 1 else "prefill"](
            ids,
            positions,
            tuple(self.state) if self.logits_only else self.state,
            cache_extent(self.position + length),
        )
        if self.logits_only:
            logits, state = result, self.state
        else:
            require(len(result) == len(self.specs) + 1, "Incorrect output/state count")
            logits, state = result[0], list(result[1:])
        require(logits.shape == (1, self.vocab_size), "Incorrect logit shape")
        require(logits.dtype == torch.bfloat16, "Incorrect logit dtype")
        require(
            bits_equal(ids, before[0]) and bits_equal(positions, before[1]),
            "Token/position input mutated",
        )
        require(len(state) == len(self.specs), "Incorrect state count")
        start, end = self.position, self.position + length
        for index, (incoming, output, original, spec) in enumerate(
            zip(self.state, state, before[2:], self.specs)
        ):
            require(list(output.shape) == spec["shape"], "Incorrect state shape")
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
                    bits_equal(output[:, :, :start], original[:, :, :start]), "Past KV overwritten"
                )
                require(
                    bits_equal(output[:, :, end:], original[:, :, end:]), "Future KV overwritten"
                )
                if self.require_write:
                    require(
                        not bits_equal(output[:, :, start:end], original[:, :, start:end]),
                        "Cache update missing",
                    )
            else:
                require(bits_equal(incoming, original), "Functional input state mutated")
        require(all(bool(torch.isfinite(t).all()) for t in (logits, *state)), "Nonfinite output")
        self.state = list(state)
        self.position = end
        return logits

    def close(self):
        self.functions = {}
        self.owner = None
        self.state = []


def load_aoti_runtime(packages, *, model_type, specs):
    """Validate the bundle before loading its operators and shared GPU weights."""
    pair = read_package_metadata(packages, model_type=model_type)
    validate_model_metadata = import_module(
        f"compiler.{model_type}.packages"
    ).validate_model_metadata
    for entry in pair.values():
        validate_model_metadata(entry)
    decode = pair["decode"]
    validate_shared_weights(packages, pair["prefill"], decode)
    require(specs == decode["state"], "Runtime state descriptors differ from package")
    load_attention_ops(model=model_type)
    weights = SharedWeights(packages, decode["shared_weights"])
    return ExplicitRuntime(
        {kind: weights.load(kind) for kind in ("prefill", "decode")},
        specs,
        decode["vocab_size"],
        decode["cache_length"],
        logits_only=model_type == "qwen3",
        owner=weights,
        require_write=model_type == "qwen3",
    )


def _snapshot(runtime, specs):
    state = list(runtime.state)
    require(len(state) == len(specs), "Incorrect state count")
    for tensor, spec in zip(state, specs):
        require(list(tensor.shape) == spec["shape"], "Incorrect state shape")
    return cpu_copy(state)


def teacher_trace(runtime, cases, specs, capture_states):
    result = Trace()
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
            key = name, step
            result.raw_logits[key] = cpu_copy((logits,))[0]
            result.logits[key] = cpu_copy((logits.float(),))[0]
            if capture_states(case, step):
                result.states[key] = _snapshot(runtime, specs)
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
                                bits_equal(actual[:, :, :position], expected[:, :, :position]),
                                "Reset/poison changed valid KV",
                            )
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
    runtime.reset()
    require(runtime.position == 0, "Reset did not clear position")
    tokens, ids, rows = prompt, [], []
    position = 0
    for index in range(limit):
        logits = runtime.step(tokens)
        position += len(tokens)
        require(runtime.position == position, "Incorrect generation position")
        rows.append(cpu_copy((logits.float(),))[0])
        token = logits.argmax(-1).item()
        ids.append(token)
        if token in eos or index + 1 == limit:
            break
        tokens = [token]
    return dict(ids=ids, rows=rows, state=cpu_copy(runtime.state), position=runtime.position)


def run_stage(runtime, cases, chats, specs, capture_states, eos):
    result = teacher_trace(runtime, cases, specs, capture_states)
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
        result.greedy[name] = value
    return result
