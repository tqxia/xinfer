"""Shared fixtures, persisted references and checks for isolated FP8 stages."""

from collections import Counter
from pathlib import Path
import re

import torch

from compiler.common.artifacts import MODEL_MANIFEST_NAME, json_sha256, read_json, require, sha256
from compiler.common.tests.tensors import bits_equal, cpu_copy
from tools.common import prompt_chunks
from tools.validation.common.chain import Policy, compare_traces
from tools.validation.common.runner import ExplicitRuntime, Trace
from tools.validation.common.validation import tensor_metrics, within_limits


def fixtures(seed, prompt, profile, capacity):
    """Keep every graph-call partition explicit and group only exact reset controls."""
    require(seed and prompt, "Missing prepared token IDs")
    stream = (seed * (max(capacity, 515) // len(seed) + 1))[: max(capacity, 515)]
    groups = []
    lengths = (1, 2, 96, 97, 98, 511, 512) if profile == "matrix" else (2,)
    for length in lengths:
        require(length + 2 <= capacity, "Validation profile exceeds context capacity")
        case = dict(
            name=f"length_{length}",
            steps=[stream[:length], [stream[length]], [stream[length + 1]]],
            prompt_steps=1,
        )
        group = [case]
        if profile == "matrix" and length == 98:
            group += [
                dict(case, name="repeat_98", same_as=case["name"]),
                dict(case, name="poison_98", same_as=case["name"], poison_kv=3),
            ]
        groups.append(group)
    if profile == "matrix":
        require(capacity >= 515, "The matrix profile needs at least 515 context slots")
        groups.append(
            [
                dict(
                    name="nonzero_prefill",
                    steps=[stream[:96], stream[96:194], [stream[194]]],
                    prompt_steps=2,
                )
            ]
        )
        groups.append(
            [
                dict(
                    name="multicall_513",
                    steps=prompt_chunks(stream[:513], 512) + [[stream[513]], [stream[514]]],
                    prompt_steps=2,
                )
            ]
        )
    require(len(prompt) + 3 <= capacity, "Chat and output budget exceed context capacity")
    chats = [
        dict(name="capital", prompt=prompt, limit=3),
        dict(name="limit_one", prompt=prompt, limit=1),
    ]
    return groups, chats


def storage_inventory(model):
    """Registered checkpoint storage only; temporary/global allocations are separate."""
    storages = {}
    for tensor in model.state_dict(keep_vars=True).values():
        storage = tensor.untyped_storage()
        key = (str(tensor.device), str(tensor.dtype), storage.data_ptr(), storage.nbytes())
        storages[key] = storage.nbytes()
    sizes, counts = Counter(), Counter()
    for (_, dtype, _, _), size in storages.items():
        sizes[dtype.removeprefix("torch.")] += size
        counts[dtype.removeprefix("torch.")] += 1
    return dict(
        unique_storages=len(storages),
        bytes=sum(storages.values()),
        bytes_by_dtype=dict(sizes),
        storages_by_dtype=dict(counts),
    )


def tensor_identity(model):
    return Counter(
        (id(value), value.data_ptr(), str(value.dtype), tuple(value.shape))
        for value in (*model.parameters(), *model.buffers())
    )


def memory(label):
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    process = {}
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value, _ = line.split()
                process[key.rstrip(":") + "_bytes"] = int(value) * 1024
    available = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        available = next(
            int(line.split()[1]) * 1024
            for line in meminfo.read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
    return dict(
        label=label,
        allocated_bytes=torch.cuda.memory_allocated(),
        reserved_bytes=torch.cuda.memory_reserved(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        cuda_free_bytes=free,
        cuda_total_bytes=total,
        system_available_bytes=available,
        **process,
    )


class ObservedRuntime:
    def __init__(self, runtime, model, layers, report):
        self.runtime, self.report = runtime, report
        self.case, self.step_index = "", 0
        self.case_names = None
        self.blocks, self.handles = {}, []
        for index in layers:
            self.handles.append(model.model.layers[index].register_forward_hook(self.hook(index)))

    def hook(self, index):
        def capture(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            self.blocks[self.case, self.step_index, index] = cpu_copy((value,))[0]

        return capture

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    def reset(self, fill=0):
        if self.case_names is not None:
            self.case = next(self.case_names)
        self.step_index = 0
        return self.runtime.reset(fill)

    def step(self, tokens):
        incoming = (
            [value.data_ptr() for value in self.runtime.state]
            if isinstance(self.runtime, ExplicitRuntime)
            else []
        )
        result = self.runtime.step(tokens)
        require(
            result.shape == (1, self.runtime.vocab_size) and result.dtype == torch.bfloat16,
            "Incorrect logit shape or dtype",
        )
        require(bool(torch.isfinite(result).all()), "Nonfinite logits")
        state = self.runtime.state
        require(len(state) == len(self.runtime.specs), "Incorrect state count")
        for value, spec in zip(state, self.runtime.specs):
            require(list(value.shape) == spec["shape"], "Incorrect state shape")
            require(value.dtype == getattr(torch, spec["dtype"]), "Incorrect state dtype")
        if isinstance(self.runtime, ExplicitRuntime):
            for before, value, spec in zip(incoming, self.runtime.state, self.runtime.specs):
                if spec["update"] == "replace":
                    require(value.data_ptr() != before, "Functional state output aliases its input")
        self.report["memory"].append(memory(f"{self.case}:{self.step_index}:t{len(tokens)}"))
        self.step_index += 1
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.runtime.close()


def case_trace(trace, name):
    result = Trace(compact_states=trace.compact_states)
    for field in ("logits", "raw_logits", "states", "positions"):
        setattr(
            result,
            field,
            {key: value for key, value in getattr(trace, field).items() if key[0] == name},
        )
    result.greedy = {key: value for key, value in trace.greedy.items() if key == name}
    return result


def save_trace(path, trace, blocks):
    torch.save(dict(trace=vars(trace), blocks=blocks), path)


def load_trace(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    return Trace(**data["trace"]), data["blocks"]


def load_reference_protocol(directory, identity):
    """Accept only a completed HF stage before consuming its traces or loading weights."""
    summary_path = directory / "summary.json"
    require(summary_path.is_file(), "HF reference is incomplete: missing summary.json")
    summary = read_json(summary_path)
    require(
        isinstance(summary, dict)
        and summary.get("model_type") == "qwen36"
        and summary.get("stage") == "hf"
        and summary.get("execution_passed") is True
        and summary.get("passed") is True
        and summary.get("failures") == [],
        "Reference must be a completed passing Qwen3.6 HF stage",
    )
    protocol = read_json(directory / "protocol.json")
    require(protocol["identity"] == identity, "HF reference profile/checkpoint differs")
    return protocol


def request_guards(runtime):
    """Check rejection before mutation, without evaluating a full context window."""
    position = runtime.position
    before = cpu_copy(runtime.state)
    pointers = [value.data_ptr() for value in runtime.state]

    def reject(tokens):
        try:
            runtime.step(tokens)
        except (ValueError, RuntimeError):
            pass
        else:
            raise ValueError("Invalid input request was accepted")
        require(
            [value.data_ptr() for value in runtime.state] == pointers,
            "Rejected request changed state ownership",
        )
        require(
            all(
                bits_equal(value.cpu(), original) for value, original in zip(runtime.state, before)
            ),
            "Rejected request changed state bytes",
        )

    reject([])
    require(runtime.position == position, "Empty request changed position")
    reject([0] * 513)
    require(runtime.position == position, "Oversized step changed position")
    try:
        runtime.position = runtime.capacity
        reject([0])
        require(runtime.position == runtime.capacity, "Overflow changed position")
    finally:
        runtime.position = position
    return dict(
        empty_and_oversized_step=True,
        overflow_preflight=True,
        overflow_used_synthetic_position=True,
        full_context_window_executed=False,
        rejected_requests_preserve_state_bytes_and_pointers=True,
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


def compare_stages(report, actual_name, expected_name, actual, expected, specs, *, exact):
    """Retain FP8 state/token gates and compare only matching greedy histories."""
    policy = Policy(state_required=True, tokens_required=True)
    matching = {
        name
        for name, value in actual.greedy.items()
        if value["ids"] == expected.greedy[name]["ids"]
        and value["position"] == expected.greedy[name]["position"]
    }
    compare_traces(
        report,
        actual_name,
        expected_name,
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
                actual_name,
                f"{expected_name}_greedy_prefix",
                left,
                right,
                specs,
                policy,
                state_keys=set(),
                greedy_names=set(),
            )


def compare_case(report, stage, actual, blocks, reference, specs):
    expected, expected_blocks = load_trace(reference)
    exact = stage == "fp8"
    compare_stages(report, stage, "hf_fp8", actual, expected, specs, exact=exact)
    require(blocks.keys() == expected_blocks.keys(), "Decoder hook coverage differs")
    for key in sorted(blocks):
        left, right = blocks[key], expected_blocks[key]
        metrics = tensor_metrics(
            left,
            right,
            rows=(
                left.float().reshape(-1, left.shape[-1]),
                right.float().reshape(-1, right.shape[-1]),
            ),
        )
        row = dict(
            comparison=f"{stage}_vs_hf_fp8",
            case=key[0],
            step=key[1],
            tensor=f"decoder_{key[2]}",
            required=True,
            exact=exact,
            **metrics,
        )
        report["comparisons"].append(row)
        if not (metrics["bitwise_equal"] if exact else within_limits(metrics)):
            report["failures"].append(row)
            report.setdefault("first_block_divergence", row)


def compare_output(report, actual, expected, specs, *, exact):
    compare_stages(
        report,
        "cpp_aoti" if exact else "python_aoti",
        "python_aoti" if exact else "adapted_eager",
        actual,
        expected,
        specs,
        exact=exact,
    )
