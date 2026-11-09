"""Download pinned checkpoints or prepare and decode offline model inputs."""

import argparse
from importlib import import_module
import json
from pathlib import Path

from compiler.common.artifacts import MODEL_MANIFEST_NAME, read_json, require
from tools.prepare.qwen3 import REVISIONS as QWEN3_REVISIONS
from tools.prepare.qwen35 import REVISIONS as QWEN35_REVISIONS
from tools.prepare.qwen36 import REVISIONS as QWEN36_REVISIONS
from tools.prepare.common import encode_prompt, load_tokenizer, write_ids
from compiler.common.context import model_context_limit


MODELS = (
    {model: "qwen3" for model in QWEN3_REVISIONS}
    | {model: "qwen35" for model in QWEN35_REVISIONS}
    | {model: "qwen36" for model in QWEN36_REVISIONS}
)


def model_hooks(model):
    require(isinstance(model, str) and model in MODELS, f"Unsupported checkpoint: {model}")
    return import_module(f"tools.prepare.{MODELS[model]}")


def checkpoint_model(directory):
    manifest = read_json(directory / MODEL_MANIFEST_NAME)
    require(isinstance(manifest, dict), "Prepared checkpoint manifest must be an object")
    model = manifest.get("repo_id", manifest.get("model"))
    require(isinstance(model, str) and model in MODELS, f"Unsupported checkpoint: {model}")
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tools.prepare", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Prepare one pinned checkpoint")
    download.add_argument("--model", choices=MODELS, required=True)
    download.add_argument("--output", type=Path)
    download.add_argument("--cache-dir", type=Path)
    verify = commands.add_parser("verify", help="Verify an existing checkpoint offline")
    verify.add_argument("--model-dir", type=Path, required=True)
    encode = commands.add_parser("encode", help="Encode a non-thinking chat prompt offline")
    encode.add_argument("--model-dir", type=Path, required=True)
    encode.add_argument("--prompt", required=True)
    encode.add_argument("--output", type=Path, required=True)
    decode = commands.add_parser("decode", help="Decode token IDs offline")
    decode.add_argument("--model-dir", type=Path, required=True)
    decode.add_argument("--input", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "download":
        hooks = model_hooks(args.model)
        directory = args.output or hooks.default_model_dir(args.model)
        manifest = hooks.prepare(directory, args.model, cache_dir=args.cache_dir)
        print(json.dumps(manifest, indent=2))
    else:
        model = checkpoint_model(args.model_dir)
        if args.command == "verify":
            manifest = model_hooks(model).verify_model(args.model_dir)
            print(f"Verified {model} at {manifest['revision']}")
        else:
            if args.command == "encode":
                require(
                    not args.output.exists() and not args.output.is_symlink(),
                    "Output already exists",
                )
                ceiling = model_context_limit(read_json(args.model_dir / "config.json"))
                tokenizer = load_tokenizer(args.model_dir)
                ids = encode_prompt(tokenizer, args.prompt)
                require(len(ids) <= ceiling, f"Expected 1..{ceiling} prompt tokens, got {len(ids)}")
                write_ids(args.output, ids)
                print(f"Wrote {len(ids)} tokens to {args.output}")
            else:
                ids = [int(word) for word in args.input.read_text().split()]
                tokenizer = load_tokenizer(args.model_dir)
                require(
                    all(0 <= token < len(tokenizer) for token in ids),
                    "Token ID is outside the tokenizer vocabulary",
                )
                print(tokenizer.decode(ids, skip_special_tokens=True))
