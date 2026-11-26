"""Validate dynamic Qwen3 batches through HF, eager, AOTI and C++."""

import argparse
from pathlib import Path

import torch
from transformers import DynamicCache

from compiler.common.artifacts import default_build_dir, read_json, require
from compiler.common.metadata import read_package_metadata
from compiler.qwen3.model import ModelStep, load_model
from tools.packages import validate_model_metadata, validate_shared_weights
from tools.prepare.common import load_tokenizer
from tools.validation.common.attention import load_packaged_attention_ops
from tools.validation.common.runner import ExplicitRuntime, load_aoti_runtime
from tools.validation.common.batch import (
    case_groups,
    run_batch_waves,
    validate_native,
    validate_python_stages,
)


def collect_hf(model, steps):
    caches = [DynamicCache(config=model.config) for _ in range(max(map(len, steps)))]
    try:
        logits = []
        for rows in steps:
            scores = []
            for row, cache in zip(rows, caches):
                if not row:
                    scores.append(torch.zeros((1, model.config.vocab_size), dtype=torch.bfloat16))
                    continue
                position = cache.get_seq_length()
                score = model(
                    input_ids=torch.tensor([row], device="cuda"),
                    past_key_values=cache,
                    cache_position=torch.arange(position, position + len(row), device="cuda"),
                    use_cache=True,
                    logits_to_keep=1,
                ).logits[:, -1]
                require(bool(torch.isfinite(score).all()), "Nonfinite HF logits")
                scores.append(score.cpu())
            logits.append(torch.cat(scores))
        state = [
            [getattr(cache.layers[layer], kind).cpu() for cache in caches]
            for layer in range(model.config.num_hidden_layers)
            for kind in ("keys", "values")
        ]
        return logits, [row for tensor in state for row in tensor]
    finally:
        caches.clear()


@torch.inference_mode()
def validate_python(args, meta, *, selected_groups=None):
    load_packaged_attention_ops(args.packages)
    tokenizer = load_tokenizer(args.model_dir)
    groups = case_groups(tokenizer, meta["cache_length"])
    if selected_groups is not None:
        groups = {name: groups[name] for name in selected_groups if name in groups}

    def load_reference():
        return load_model(args.model_dir)

    def load_eager():
        adapted = ModelStep(load_model(args.model_dir))
        return ExplicitRuntime(
            dict(prefill=adapted, decode=adapted),
            meta["state"],
            meta["vocab_size"],
            meta["cache_length"],
            logits_only=True,
            require_write=True,
        )

    def load_aoti():
        return load_aoti_runtime(args.packages, model_type="qwen3", specs=meta["state"])

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
    require(
        not args.output.exists() and not args.output.is_symlink(),
        "Use a fresh validation output directory",
    )
    meta = read_package_metadata(args.packages, "qwen3")
    validate_model_metadata(meta)
    validate_shared_weights(args.packages, meta)
    require(
        meta["checkpoint"] == read_json(args.model_dir / "model_manifest.json"),
        "Checkpoint manifest differs",
    )
    (args.output / "expected").mkdir(parents=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return run_batch_waves(args, meta, validate_python, validate_native)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native", type=Path, default=default_build_dir() / "xinfer_validate")
    args = parser.parse_args()
    results = validate(args)
    raise SystemExit(any(item["required_failures"] for item in results.values()))


if __name__ == "__main__":
    main()
