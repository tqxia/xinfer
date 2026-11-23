"""Validate dynamic Qwen3 batches through HF, eager, AOTI and C++."""

import argparse
import gc
from pathlib import Path

import torch
from transformers import DynamicCache

from compiler.common.artifacts import read_json, require
from compiler.common.metadata import read_package_metadata
from compiler.qwen3.model import ModelStep, load_model
from tools.prepare.common import load_tokenizer
from tools.validation.common.attention import load_packaged_attention_ops
from tools.validation.common.runner import ExplicitRuntime, load_aoti_runtime
from tools.validation.common.batch import case_groups, validate_python_group, validate_native


def collect_hf(model, steps):
    caches = [DynamicCache(config=model.config) for _ in range(max(map(len, steps)))]
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


@torch.inference_mode()
def validate_python(args, meta):
    load_packaged_attention_ops(args.packages)
    tokenizer = load_tokenizer(args.model_dir)
    groups = case_groups(tokenizer, meta["cache_length"])
    hf = load_model(args.model_dir)
    adapted = ModelStep(load_model(args.model_dir))
    eager = ExplicitRuntime(
        dict(prefill=adapted, decode=adapted),
        meta["state"],
        meta["vocab_size"],
        meta["cache_length"],
        logits_only=True,
        require_write=True,
    )
    aoti = load_aoti_runtime(args.packages, model_type="qwen3", specs=meta["state"])
    try:
        for group, (texts, cases) in groups.items():
            validate_python_group(
                args, group, texts, cases, meta, hf, eager, aoti, collect_reference=collect_hf
            )
    finally:
        eager.close()
        aoti.close()
    return list(groups)


def validate(args):
    require(__debug__, "Validation requires assertions enabled")
    require(not args.output.exists(), "Use a fresh validation output directory")
    meta = read_package_metadata(args.packages, "qwen3")["decode"]
    require(
        meta["checkpoint"]["manifest"] == read_json(args.model_dir / "model_manifest.json"),
        "Checkpoint manifest differs",
    )
    (args.output / "expected").mkdir(parents=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    groups = validate_python(args, meta)
    # Native validation starts after the Python model set has been released.
    gc.collect()
    torch.cuda.empty_cache()
    return validate_native(args, groups)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native", type=Path, default=Path("build/qwen3/xinfer_validate"))
    args = parser.parse_args()
    results = validate(args)
    raise SystemExit(any(item["required_failures"] for item in results.values()))


if __name__ == "__main__":
    main()
