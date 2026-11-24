"""Download and verify the pinned, unchanged Qwen3 HF checkpoint."""

import importlib.metadata
from pathlib import Path
import re

from compiler.common.artifacts import MODEL_MANIFEST_NAME, read_json, require, sha256, write_json
from compiler.qwen3.config import DEFAULT_MODEL
from tools.prepare import common


REVISIONS = {
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}
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
BASE_FILES = tuple(name for name in MODEL_FILES if name != "model.safetensors")
WEIGHT_INDEX = "model.safetensors.index.json"


def default_model_dir(model_id):
    require(isinstance(model_id, str) and model_id in REVISIONS, "Unsupported checkpoint")
    return DEFAULT_MODEL.parent / model_id.rsplit("/", 1)[-1].lower()


def _shard_names(index_path):
    index = read_json(index_path)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    require(isinstance(weight_map, dict) and bool(weight_map), "Invalid checkpoint weight index")
    shards = sorted(set(weight_map.values()))
    for shard in shards:
        relative = Path(shard) if isinstance(shard, str) else None
        # Allow only flat .safetensors filenames in the checkpoint directory (no path traversal).
        require(
            relative is not None
            and not relative.is_absolute()
            and relative.parts == (shard,)
            and shard.endswith(".safetensors"),
            f"Invalid checkpoint weight shard: {shard}",
        )
    return shards


def _check_descriptor(name, expected):
    require(
        isinstance(expected, dict)
        and set(expected) == {"bytes", "sha256"}
        and type(expected["bytes"]) is int
        and expected["bytes"] >= 0
        and isinstance(expected["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]) is not None,
        f"Invalid checkpoint file metadata: {name}",
    )


def _verify_file(directory, name, expected):
    path = directory / name
    require(path.is_file(), f"Checkpoint file is missing: {path}")
    require(
        path.stat().st_size == expected["bytes"] and sha256(path) == expected["sha256"],
        f"Checkpoint checksum mismatch: {path}",
    )
    print(f"verified {name}", flush=True)


def prepare(directory, model, cache_dir=None):
    """Publish the original pinned HF files and their verified manifest."""
    require(isinstance(model, str) and model in REVISIONS, "Unsupported checkpoint")
    with common.checkpoint_directory(directory) as staging:
        hub = common.Hub(model, REVISIONS[model], cache_dir=cache_dir)
        hub_files = getattr(hub, "files", {})
        if isinstance(hub_files, dict) and WEIGHT_INDEX in hub_files:
            # Larger Qwen3 checkpoints shard weights; copy the index first to discover its shards.
            files = {name: hub.copy(name, staging / name) for name in BASE_FILES[:2]}
            files[WEIGHT_INDEX] = hub.copy(WEIGHT_INDEX, staging / WEIGHT_INDEX)
            for shard in _shard_names(staging / WEIGHT_INDEX):
                files[shard] = hub.copy(shard, staging / shard)
            for name in BASE_FILES[2:]:
                files[name] = hub.copy(name, staging / name)
        else:
            # Single-file checkpoints (such as Qwen3-0.6B) carry weights in model.safetensors.
            files = {name: hub.copy(name, staging / name) for name in MODEL_FILES}
        manifest = dict(
            model=model,
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
    model_id = manifest.get("model")
    require(
        isinstance(model_id, str)
        and model_id in REVISIONS
        and manifest.get("revision") == REVISIONS[model_id],
        "Prepared checkpoint must match a supported pinned revision",
    )
    files = manifest.get("files")
    require(
        isinstance(files, dict) and set(BASE_FILES) <= set(files),
        "Prepared checkpoint file manifest is incomplete or unsupported",
    )
    extra = set(files) - set(BASE_FILES)
    if extra == {"model.safetensors"}:
        ordered = list(MODEL_FILES)
    else:
        # Sharded checkpoints must declare the weight index plus flat .safetensors shard names.
        require(
            WEIGHT_INDEX in extra
            and len(extra) > 1
            and all(
                isinstance(name, str)
                and Path(name).parts == (name,)
                and (name == WEIGHT_INDEX or name.endswith(".safetensors"))
                for name in extra
            ),
            "Prepared checkpoint file manifest is incomplete or unsupported",
        )
        ordered = [*BASE_FILES[:2], WEIGHT_INDEX, *sorted(extra - {WEIGHT_INDEX}), *BASE_FILES[2:]]
    # Validate every descriptor before hashing any files on disk.
    for name in ordered:
        _check_descriptor(name, files[name])
    if WEIGHT_INDEX in files:
        # Verify the index checksum first, then ensure the manifest covers its exact shard set.
        _verify_file(directory, WEIGHT_INDEX, files[WEIGHT_INDEX])
        shards = _shard_names(directory / WEIGHT_INDEX)
        require(
            extra == {WEIGHT_INDEX, *shards},
            "Prepared checkpoint file manifest is incomplete or unsupported",
        )
    for name in ordered:
        if name != WEIGHT_INDEX:
            _verify_file(directory, name, files[name])
    return manifest
