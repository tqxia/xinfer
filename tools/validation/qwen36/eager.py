"""Run one isolated HF/FP8-adapter/full-adapter eager validation stage."""

import argparse
from collections import Counter
from importlib.metadata import version
import os
from pathlib import Path
import traceback

import torch

from compiler.common.artifacts import (
    MODEL_MANIFEST_NAME,
    json_sha256,
    read_json,
    require,
    sha256,
    write_json,
)
from compiler.common.tests.tensors import bits_equal, cpu_copy
from tools.common import prompt_chunks
from tools.validation.common.chain import Policy, compare_traces, new_report
from tools.validation.common.runner import ExplicitRuntime, Trace, greedy, teacher_trace
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


def compare_case(report, stage, actual, blocks, reference, specs):
    expected, expected_blocks = load_trace(reference)
    exact = stage == "fp8"
    matching_generations = {
        name
        for name, value in actual.greedy.items()
        if value["ids"] == expected.greedy[name]["ids"]
        and value["position"] == expected.greedy[name]["position"]
    }
    compare_traces(
        report,
        stage,
        "hf_fp8",
        actual,
        expected,
        specs,
        Policy(state_required=True, tokens_required=True),
        exact=exact,
        greedy_state_names=matching_generations,
    )
    if not exact:
        for name, value in actual.greedy.items():
            other = expected.greedy[name]
            count = 0
            for token, reference_token in zip(value["ids"], other["ids"]):
                count += 1
                if token != reference_token:
                    break
            # Inputs still match at the first differing prediction. Subsequent
            # logits/state follow different token histories and cannot be compared.
            left = Trace(
                logits={(name, index): row for index, row in enumerate(value["rows"][:count])}
            )
            right = Trace(
                logits={(name, index): row for index, row in enumerate(other["rows"][:count])}
            )
            compare_traces(
                report,
                stage,
                "hf_fp8_greedy_prefix",
                left,
                right,
                specs,
                Policy(state_required=True, tokens_required=True),
                state_keys=set(),
                greedy_names=set(),
            )
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


def run(args, report):
    from transformers import AutoTokenizer

    from compiler.qwen36.model import TextStep, load_model, state_specs
    from compiler.common.context import resolve_context_length
    from tools.prepare.common import encode_prompt
    from tools.validation.common.prompts import PARAGRAPH
    from tools.validation.qwen35.model import HFStage

    require(torch.cuda.is_available(), "CUDA is required")
    report["environment"] = dict(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        transformers=version("transformers"),
        triton=version("triton"),
        kernels=version("kernels"),
        device=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        attention_library_sha256=(
            sha256(args.attention_library) if args.attention_library else None
        ),
    )
    config = read_json(args.model_dir / "config.json")["text_config"]
    resolve_context_length(config, args.capacity)
    require(1 <= args.layers <= config["num_hidden_layers"], "Invalid decoder prefix length")
    require(
        len(set(args.capture_layers)) == len(args.capture_layers)
        and all(0 <= layer < args.layers for layer in args.capture_layers),
        "Invalid captured decoder layers",
    )
    manifest = read_json(args.model_dir / MODEL_MANIFEST_NAME)
    identity = dict(
        manifest_sha256=sha256(args.model_dir / MODEL_MANIFEST_NAME),
        layers=args.layers,
        capacity=args.capacity,
        profile=args.profile,
        captured_layers=args.capture_layers,
    )
    reference_protocol = None
    if args.reference:
        reference_protocol = load_reference_protocol(args.reference, identity)
        groups, chats = reference_protocol["groups"], reference_protocol["chats"]
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        groups, chats = fixtures(
            tokenizer.encode(PARAGRAPH, add_special_tokens=False),
            encode_prompt(tokenizer, "What is the capital of France? Answer briefly."),
            args.profile,
            args.capacity,
        )
    generation = read_json(args.model_dir / "generation_config.json")
    eos = generation["eos_token_id"]
    torch.cuda.reset_peak_memory_stats()
    report["memory"].append(memory("before_load"))
    model = load_model(args.model_dir, device="cuda", num_hidden_layers=args.layers)
    print(f"{args.stage}: loaded {args.layers} decoder layers", flush=True)
    report["memory"].append(memory("after_load"))
    specs = state_specs(model.config, args.capacity)
    require(
        all(0 <= layer < len(model.model.layers) for layer in args.capture_layers),
        "Invalid captured decoder layer",
    )
    before = tensor_identity(model)
    report["checkpoint_storage"] = getattr(model, "checkpoint_storage", None)
    report["registered_storage_before"] = storage_inventory(model)
    if report["checkpoint_storage"]:
        require(
            report["registered_storage_before"]["bytes"]
            == report["checkpoint_storage"]["source_bytes"],
            "Registered checkpoint storage differs from prepared source budget",
        )
    report["hf_experts_backend"] = getattr(model.config, "_experts_implementation", None)
    if args.stage != "hf":
        from compiler.qwen36.adapters import install_fp8

        report["fp8_adapters"] = install_fp8(model)
    if args.stage == "adapted":
        from compiler.qwen35.adapters.delta import install_delta

        require(
            args.attention_library and args.attention_library.is_file(),
            "Build the Qwen3.6 attention library first",
        )
        os.environ["XINFER_ATTENTION_OPS_LIBRARY"] = str(args.attention_library.resolve())
        install_delta(model)
        wrapper = TextStep(model)
        runtime = ExplicitRuntime(
            dict(prefill=wrapper, decode=wrapper),
            specs,
            model.config.vocab_size,
            args.capacity,
            owner=model,
            require_write=True,
        )
    else:
        runtime = HFStage(model, specs, args.capacity)
    require(
        tensor_identity(model) == before, "Adapters changed registered tensors or storage pointers"
    )
    report["registered_storage_after"] = storage_inventory(model)
    require(
        report["registered_storage_before"] == report["registered_storage_after"],
        "Adapters changed checkpoint storage budget",
    )
    report["adapter_storage_shared"] = True
    report["memory"].append(memory("after_adapt"))
    protocol = dict(
        identity=identity,
        groups=groups,
        chats=chats,
        state=specs,
        eos_token_ids=eos,
        input_sha256=json_sha256(dict(groups=groups, chats=chats)),
    )
    if reference_protocol:
        require(reference_protocol == protocol, "HF reference inputs/state differ")
    write_json(args.output / "protocol.json", protocol)
    report.update(
        reference_protocol=dict(
            model="official HF FP8",
            experts="grouped_mm",
            compute="unmodified HF forwards",
            dtype="mixed FP8 weights/BF16 activations; FP32 recurrent state",
        ),
        checkpoint=dict(model=manifest["model"], revision=manifest["revision"]),
        state_tensors=len(specs),
        numerical_policy=dict(
            relative_l2=0.05,
            peak_relative=0.05,
            fp8_only="bitwise",
            adapted="required logits/state/tokens",
            bf16_block_diagnostics="unchanged, separate operator reports",
        ),
        state_snapshot_policy=dict(
            functional="complete at completed prompt/final request",
            kv="valid prefix" if args.capacity > 1024 else "full cache",
            adapted_contracts="per-call finiteness, caller KV ownership/write regions and functional input/output ownership",
        ),
        memory_scope="Python allocator peaks and synchronized boundary samples; VmRSS/VmHWM/system available; CUDA and system figures overlap on GB10 unified memory and must not be added",
    )
    observed = ObservedRuntime(runtime, model, args.capture_layers, report)
    args.output.joinpath("traces").mkdir()
    try:
        for group in groups:
            # Common teacher_trace owns exact reset/poison checks. A group has
            # at most three cases, bounding resident reference-state snapshots.
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
                blocks = {key: value for key, value in observed.blocks.items() if key[0] == name}
                save_trace(args.output / "traces" / f"{name}.pt", selected, blocks)
                if args.reference:
                    compare_case(
                        report,
                        args.stage,
                        selected,
                        blocks,
                        args.reference / "traces" / f"{name}.pt",
                        specs,
                    )
                report["teacher_steps"] += len(selected.logits)
                report["state_checkpoints"] += len(selected.states)
            observed.blocks.clear()
            del trace, selected, blocks
            print(
                f"{args.stage}: completed {', '.join(case['name'] for case in group)}; "
                f"failures={len(report['failures'])}",
                flush=True,
            )
        for chat in chats:
            name = chat["name"]
            observed.case_names = None
            observed.case = name
            value = greedy(observed, chat["prompt"], chat["limit"], eos)
            trace = Trace(greedy={name: value}, compact_states=args.capacity > 1024)
            save_trace(args.output / "traces" / f"{name}.pt", trace, {})
            if args.reference:
                compare_case(
                    report, args.stage, trace, {}, args.reference / "traces" / f"{name}.pt", specs
                )
            report["greedy_cases"] += 1
            report.setdefault("greedy_ids", {})[name] = value["ids"]
            observed.blocks.clear()
            print(f"{args.stage}: completed {name}; failures={len(report['failures'])}", flush=True)
        report["memory"].append(memory("after_requests"))
        report["request_guards"] = request_guards(runtime)
        report["memory"].append(memory("after_guards"))
        report["greedy_comparison_scope"] = (
            "Required token equality; numerical logits through first differing prediction only; final state only for matching token histories and positions"
        )
        report["execution_passed"] = True
        report["comparison_passed"] = not report["failures"] if args.reference else None
        report["passed"] = report["execution_passed"] and report["comparison_passed"] is not False
        if report["failures"]:
            report["first_divergence"] = report["failures"][0]
    finally:
        observed.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("hf", "fp8", "adapted"), required=True)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--capacity", type=int, default=1024)
    parser.add_argument("--profile", choices=("smoke", "matrix"), default="matrix")
    parser.add_argument("--capture-layers", nargs="*", type=int, default=[])
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--attention-library", type=Path, default=os.environ.get("XINFER_ATTENTION_OPS_LIBRARY")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(
        args.stage == "hf" or args.reference, "Adapter stages require a serialized HF reference"
    )
    require(args.stage != "hf" or args.reference is None, "HF stage creates the reference")
    require(
        args.stage != "adapted" or args.attention_library and args.attention_library.is_file(),
        "Build the Qwen3.6 attention library first",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    report = new_report("qwen36")
    report.update(
        stage=args.stage,
        scope="eager only; no export, AOTI or native execution",
        native_bitwise=None,
        execution_passed=False,
        comparison_passed=None,
        memory=[],
        teacher_steps=0,
        state_checkpoints=0,
        greedy_cases=0,
    )
    try:
        with torch.inference_mode():
            run(args, report)
    except Exception as error:
        report.update(passed=False, error=dict(type=type(error).__name__, message=str(error)))
        traceback.print_exc()
    write_json(args.output / "summary.json", report)
    print(f"Passed: {report['passed']}; report: {args.output / 'summary.json'}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
