"""Run one isolated HF/FP8-adapter/full-adapter eager validation stage."""

import argparse
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
from tools.validation.common.chain import new_report
from tools.validation.common.runner import ExplicitRuntime, Trace, greedy, teacher_trace
from tools.validation.qwen36.common import (
    ObservedRuntime,
    case_trace,
    compare_case,
    fixtures,
    load_reference_protocol,
    memory,
    request_guards,
    save_trace,
    storage_inventory,
    tensor_identity,
)


def run(args, report):
    from transformers import AutoTokenizer

    from compiler.qwen36.model import TextStep, load_model, state_specs
    from compiler.common.context import resolve_context_length
    from tools.prepare.common import encode_prompt
    from tools.validation.common.prompts import PARAGRAPH
    from tools.validation.common.hybrid import HybridHFStage

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
        runtime = HybridHFStage(model, specs, args.capacity)
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
