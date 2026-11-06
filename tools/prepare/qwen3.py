"""Download and verify the pinned, unchanged Qwen3 HF checkpoint."""

import importlib.metadata
from pathlib import Path
import re

from compiler.common.artifacts import MODEL_MANIFEST_NAME, read_json, require, sha256, write_json
from compiler.qwen3.config import DEFAULT_MODEL
from tools.prepare import common


MODEL_ID = "Qwen/Qwen3-0.6B"
REVISIONS = {MODEL_ID: "c1899de289a04d12100db370d81485cdf75e47ca"}
MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
    "LICENSE",
)


def default_model_dir(model_id):
    require(isinstance(model_id, str) and model_id in REVISIONS, "Unsupported checkpoint")
    return DEFAULT_MODEL


def prepare(directory, model=MODEL_ID, cache_dir=None):
    """Publish the original pinned HF files and their verified manifest."""
    require(isinstance(model, str) and model in REVISIONS, "Unsupported checkpoint")
    with common.checkpoint_directory(directory) as staging:
        hub = common.Hub(model, REVISIONS[model], cache_dir=cache_dir)
        files = {name: hub.copy(name, staging / name) for name in MODEL_FILES}
        manifest = dict(
            repo_id=model,
            revision=REVISIONS[model],
            status="complete",
            huggingface_hub=importlib.metadata.version("huggingface-hub"),
            files=files,
        )
        write_json(staging / MODEL_MANIFEST_NAME, manifest)
    return manifest


def verify_model(model_dir):
    """Check the pinned identity, complete file inventory and retained file bytes."""
    directory = Path(model_dir)
    manifest = read_json(directory / MODEL_MANIFEST_NAME)
    require(isinstance(manifest, dict), "Prepared checkpoint manifest must be an object")
    model_id = manifest.get("repo_id")
    require(
        isinstance(model_id, str)
        and model_id in REVISIONS
        and manifest.get("revision") == REVISIONS[model_id],
        "Prepared checkpoint must match a supported pinned revision",
    )
    files = manifest.get("files")
    require(
        isinstance(files, dict) and set(files) == set(MODEL_FILES),
        "Prepared checkpoint file manifest is incomplete or unsupported",
    )
    for name in MODEL_FILES:
        expected = files[name]
        require(
            isinstance(expected, dict)
            and set(expected) == {"bytes", "sha256"}
            and type(expected["bytes"]) is int
            and expected["bytes"] >= 0
            and isinstance(expected["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]) is not None,
            f"Invalid checkpoint file metadata: {name}",
        )
        path = directory / name
        require(path.is_file(), f"Checkpoint file is missing: {path}")
        require(
            path.stat().st_size == expected["bytes"] and sha256(path) == expected["sha256"],
            f"Checkpoint checksum mismatch: {path}",
        )
        print(f"verified {name}", flush=True)
    return manifest
