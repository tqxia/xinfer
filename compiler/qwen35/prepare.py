"""Offline chat tokenization and decoding; model inference stays in C++."""

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from compiler.qwen35.config import TEXT_DIR


def encode_prompt(tokenizer, prompt):
    if not prompt.strip():
        raise ValueError("Prompt must not be empty")
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False,
    )
    if not 1 <= len(ids) <= 512:
        raise ValueError(f"Expected 1..512 prompt tokens, got {len(ids)}")
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=TEXT_DIR)
    commands = parser.add_subparsers(dest="command", required=True)
    encode = commands.add_parser("encode")
    encode.add_argument("--prompt", required=True)
    encode.add_argument("--output", type=Path, required=True)
    decode = commands.add_parser("decode")
    decode.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if args.command == "encode":
        ids = encode_prompt(tokenizer, args.prompt)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(" ".join(map(str, ids)) + "\n")
        print(f"Wrote {len(ids)} tokens to {args.output}")
    else:
        ids = [int(word) for word in args.input.read_text().split()]
        if any(token < 0 or token >= len(tokenizer) for token in ids):
            raise ValueError("Token ID is outside the tokenizer vocabulary")
        print(tokenizer.decode(ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
