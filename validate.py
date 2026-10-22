"""Step 1: save and verify eager HF references for later AOTI/C++ comparisons."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, GenerationConfig

from prepare import DEFAULT_MODEL, encode_prompt, load_tokenizer, verify_model, write_ids


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


@torch.inference_mode()
def greedy_reference(model, prompt_ids, eos_ids, max_new_tokens):
    """Prefill once, then feed one token at a time using HF's own DynamicCache."""
    tokens = torch.tensor([prompt_ids], dtype=torch.int64, device="cuda")
    cache = None
    generated, logits = [], []
    for _ in range(max_new_tokens):
        result = model(
            input_ids=tokens,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        scores = result.logits[0, -1].float()
        if not torch.isfinite(scores).all().item():
            raise ValueError("Non-finite reference logits")
        token = scores.argmax().item()
        generated.append(token)
        logits.append(scores.cpu())
        cache = result.past_key_values
        if token in eos_ids:
            break
        tokens = torch.tensor([[token]], dtype=torch.int64, device="cuda")
    return generated, torch.stack(logits)


def environment(model, model_manifest):
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "triton", "transformers", "tokenizers",
                         "safetensors", "huggingface-hub", "numpy")
        },
        "torch_runtime_version": torch.__version__,
        "torch_path": str(Path(torch.__file__).resolve()),
        "cuda_build_version": torch.version.cuda,
        "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
        "gpu": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_memory_bytes": properties.total_memory,
        "model_revision": model_manifest["revision"],
        "dtype": str(model.dtype),
        "attention": model.config._attn_implementation,
        "cache": "HF DynamicCache (unmodified eager baseline)",
        "thinking": False,
        "sampling": "greedy",
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=Path("prompts.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/reference"))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 1024:
        parser.error("--max-new-tokens must be in 1..1024")
    if args.output.exists():
        parser.error(f"Output already exists; choose a fresh directory: {args.output}")
    manifest = verify_model(args.model_dir)
    tokenizer = load_tokenizer(args.model_dir)
    prompts = json.loads(args.prompts.read_text())
    fixtures = []
    for prompt in prompts:
        name = prompt["name"]
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name):
            raise ValueError(f"Invalid fixture name: {name!r}")
        ids = encode_prompt(tokenizer, prompt["text"])
        if len(ids) + args.max_new_tokens > 1024:
            raise ValueError(f"Context budget exceeded for {name}")
        fixtures.append((name, prompt["text"], ids))
    if not fixtures or len({name for name, _, _ in fixtures}) != len(fixtures):
        raise ValueError("Expected nonempty, uniquely named fixtures")

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval().to("cuda")
    args.output.mkdir(parents=True)
    write_json(args.output / "environment.json", environment(model, manifest))
    write_json(args.output / "model_manifest.json", manifest)
    eos_ids = model.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    # A fresh config avoids the checkpoint's sampling defaults and processors.
    generation = GenerationConfig(
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_ids,
        pad_token_id=model.generation_config.pad_token_id,
        bos_token_id=model.generation_config.bos_token_id,
        use_cache=True,
    )
    summary = []
    for name, prompt, ids in fixtures:
        generated, logits = greedy_reference(model, ids, eos_ids, args.max_new_tokens)
        repeated, repeated_logits = greedy_reference(model, ids, eos_ids, args.max_new_tokens)
        if generated != repeated or not torch.equal(logits, repeated_logits):
            raise AssertionError(f"Fresh-cache repeatability failed for {name}")
        with torch.inference_mode():
            hf_ids = model.generate(
                input_ids=torch.tensor([ids], device="cuda"),
                attention_mask=torch.ones((1, len(ids)), dtype=torch.int64, device="cuda"),
                generation_config=generation,
                # 4.51 otherwise merges checkpoint defaults back into this config.
                use_model_defaults=False,
            )[0, len(ids):].tolist()
        if generated != hf_ids:
            raise AssertionError(f"Manual loop disagrees with HF generate for {name}")

        case = args.output / name
        case.mkdir()
        write_ids(case / "prompt.ids", ids)
        write_ids(case / "generated.ids", generated)
        # Plain little-endian float32 needs no Python serializer in the C++ reader.
        values = logits.numpy().astype("<f4", copy=False)
        values.tofile(case / "logits.f32")
        text = tokenizer.decode(generated, skip_special_tokens=True)
        (case / "output.txt").write_text(text + "\n")
        metadata = {
            "name": name,
            "prompt": prompt,
            "prompt_tokens": len(ids),
            "generated_tokens": len(generated),
            "max_new_tokens": args.max_new_tokens,
            "stop_reason": "eos" if generated[-1] in eos_ids else "limit",
            "eos_token_ids": eos_ids,
            "logits_shape": list(values.shape),
            "logits_dtype": "little-endian float32",
            "logits_semantics": "row i predicts generated.ids[i] after prompt + generated.ids[:i]",
            "logits_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            "matches_hf_generate": True,
            "repeat_logits_max_abs_error": 0.0,
            "output_text": text,
        }
        write_json(case / "metadata.json", metadata)
        summary.append(metadata)
        print(f"{name}: {len(ids)} prompt, {len(generated)} generated, {metadata['stop_reason']}", flush=True)
        print(text, flush=True)
    write_json(args.output / "summary.json", summary)
    print(f"PASS: {len(summary)} eager fixtures saved in {args.output}")


if __name__ == "__main__":
    main()
