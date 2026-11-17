"""Shared CLI preflight and build reporting for model exports."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch

from compiler.common.artifacts import read_json
from compiler.common.context import model_context_limit, resolve_context_length


class ExportInputError(ValueError):
    """Invalid export arguments, reported as usage errors by the CLI."""


def export_parser(description, *, model_dir, output):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model-dir", type=Path, default=model_dir)
    parser.add_argument("--output", type=Path, default=output)
    parser.add_argument(
        "--max-context-length",
        type=int,
        help="Compiled KV-cache capacity; defaults to model max_position_embeddings",
    )
    return parser


def prepare_export(model_dir, output, max_context_length, max_prefill):
    """Reject invalid requests and report capacity before loading GPU weights."""
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise ExportInputError(f"Output must be a new or empty directory: {output}")
    try:
        context_config = read_json(model_dir / "config.json")
        capacity = resolve_context_length(context_config, max_context_length)
        ceiling = model_context_limit(context_config)
    except ValueError as error:
        raise ExportInputError(str(error)) from error
    print(
        json.dumps(
            dict(
                event="context",
                cache_length=capacity,
                model_max_position_embeddings=ceiling,
                max_prefill_tokens=max_prefill,
            )
        ),
        flush=True,
    )
    return capacity


def check_symbolic_config():
    from torch.fx.experimental import _config as symbolic_config

    if symbolic_config.backed_size_oblivious:
        raise ValueError("Flash must export with backed_size_oblivious disabled")


def report_export(model_type, kind, *, strict, diagnostics, extra_dependencies=(), **fields):
    """Keep build diagnostics in export logs, separate from deployment metadata."""
    print(
        json.dumps(
            dict(
                event="export",
                model_type=model_type,
                entry_point=kind,
                **fields,
                build=dict(
                    strict_export=strict,
                    torch_cuda_version=torch.version.cuda,
                    versions={
                        name: importlib.metadata.version(name)
                        for name in (
                            "transformers",
                            "triton",
                            "safetensors",
                            "tokenizers",
                            "huggingface-hub",
                            *extra_dependencies,
                        )
                    },
                ),
                diagnostics=diagnostics,
            )
        ),
        flush=True,
    )
