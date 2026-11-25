"""Validate dynamic Qwen3.5 batches through BF16 HF, packed eager, AOTI and C++."""

import argparse
import gc
from pathlib import Path

import torch

from compiler.common.artifacts import read_json, require
from compiler.common.metadata import read_package_metadata
from compiler.qwen35.adapters.delta import install_delta
from compiler.qwen35.export import export_step
from compiler.qwen35.config import configure
from compiler.qwen35.model import load_model
from tools.prepare.common import load_tokenizer
from tools.validation.common.attention import load_packaged_attention_ops
from tools.validation.common.batch import case_groups, validate_native, validate_python_stages
from tools.validation.common.runner import ExplicitRuntime, load_aoti_runtime
from tools.validation.qwen35.model import HFStage


def delta_cases(prompts):
    """Exercise both sides of GDN dispatch, carried state and the step-2 numerical fixture."""
    cases = [
        dict(
            name="prompt_prefix_teacher",
            batch_steps=[prompts] + [[[row[index]] for row in prompts] for index in range(3)],
        )
    ]
    for length in (97, 98):
        rows = [(row * (length // len(row) + 1))[: length - i % 2] for i, row in enumerate(prompts)]
        cases.append(
            dict(name=f"delta_{length}", batch_steps=[rows, [[row[0]] for row in prompts], rows])
        )
    for case in cases:
        case["state_steps"] = [len(case["batch_steps"]) - 1]
    return cases


def collect_hf(factory, steps):
    runtimes = [factory() for _ in range(max(map(len, steps)))]
    try:
        for runtime in runtimes:
            runtime.reset()
        logits = []
        for rows in steps:
            scores = [
                runtime.step(tokens).cpu()
                if tokens
                else torch.zeros((1, runtime.vocab_size), dtype=torch.bfloat16)
                for runtime, tokens in zip(runtimes, rows)
            ]
            require(all(torch.isfinite(score).all() for score in scores), "Nonfinite HF logits")
            logits.append(torch.cat(scores))
        states = [runtime.state for runtime in runtimes]
        state = [
            (
                values[index][:, :, : runtime.position]
                if spec["update"] == "inplace"
                else values[index]
            ).cpu()
            for index, spec in enumerate(runtimes[0].specs)
            for runtime, values in zip(runtimes, states)
        ]
        return logits, state
    finally:
        for runtime in runtimes:
            if hasattr(runtime, "close"):
                runtime.close()


@torch.inference_mode()
def validate_python(args, meta):
    configure()
    load_packaged_attention_ops(args.packages)
    groups = case_groups(load_tokenizer(args.model_dir), meta["cache_length"])
    for name, (_, cases) in groups.items():
        if name != "transitions":
            cases.extend(delta_cases(cases[0]["batch_steps"][0]))

    def load_reference():
        hf = load_model(args.model_dir)
        specs = [dict(spec, shape=[1, *spec["shape"][1:]]) for spec in meta["state"]]

        def factory():
            return HFStage(hf, specs, meta["cache_length"])

        return factory

    def load_eager():
        adapted = load_model(args.model_dir)
        install_delta(adapted)
        functions = {
            kind: export_step(adapted, kind, meta["cache_length"])[0].module()
            for kind in ("prefill", "decode")
        }
        return ExplicitRuntime(
            functions, meta["state"], meta["vocab_size"], meta["cache_length"], require_write=True
        )

    def load_aoti():
        return load_aoti_runtime(args.packages, model_type="qwen35", specs=meta["state"])

    return validate_python_stages(
        args,
        groups,
        meta,
        load_reference=load_reference,
        collect_reference=collect_hf,
        load_eager=load_eager,
        load_aoti=load_aoti,
    )


def validate(args):
    require(__debug__, "Validation requires assertions enabled")
    require(not args.output.exists(), "Use a fresh validation output directory")
    meta = read_package_metadata(args.packages, "qwen35")
    require(
        meta["checkpoint"] == read_json(args.model_dir / "model_manifest.json"),
        "Checkpoint manifest differs",
    )
    (args.output / "expected").mkdir(parents=True)
    torch.set_num_threads(1)
    groups = validate_python(args, meta)
    gc.collect()
    torch.cuda.empty_cache()
    return validate_native(args, groups)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native", type=Path, default=Path("build/qwen35/xinfer_validate"))
    args = parser.parse_args()
    results = validate(args)
    raise SystemExit(any(item["required_failures"] for item in results.values()))


if __name__ == "__main__":
    main()
