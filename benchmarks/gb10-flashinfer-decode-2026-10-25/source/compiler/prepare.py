"""Offline checkpoint verification, chat tokenization, and output decoding."""

import argparse
import hashlib
import json
import os
from pathlib import Path


DEFAULT_MODEL = Path(os.environ.get(
    "XINFER_MODEL_DIR", Path(__file__).resolve().parents[1] / "work/models/qwen3-0.6b"
)).expanduser()


def verify_model(model_dir):
    manifest = json.loads((model_dir / "model_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = model_dir / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != expected["bytes"] or digest != expected["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {path}")
        print(f"verified {name}", flush=True)
    return manifest


def load_tokenizer(model_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir, local_files_only=True)


def encode_prompt(tokenizer, prompt):
    # Keep the official template, including its non-thinking assistant prefix.
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not 1 <= len(ids) <= 512:
        raise ValueError(f"Expected 1..512 prompt tokens, got {len(ids)}")
    return ids


def write_ids(path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(" ".join(map(str, ids)) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    encode = commands.add_parser("encode")
    encode.add_argument("--prompt", required=True)
    encode.add_argument("--output", type=Path, required=True)
    decode = commands.add_parser("decode")
    decode.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        manifest = verify_model(args.model_dir)
        print(f"Verified {manifest['repo_id']} at {manifest['revision']}")
        return
    tokenizer = load_tokenizer(args.model_dir)
    if args.command == "encode":
        ids = encode_prompt(tokenizer, args.prompt)
        write_ids(args.output, ids)
        print(f"Wrote {len(ids)} tokens to {args.output}")
    else:
        ids = [int(word) for word in args.input.read_text().split()]
        if any(token < 0 or token >= len(tokenizer) for token in ids):
            raise ValueError("Token ID is outside the tokenizer vocabulary")
        print(tokenizer.decode(ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
