"""Prepare pinned Qwen3.5 text checkpoints from their multimodal sources."""

import json
from pathlib import Path

from compiler.common.artifacts import MODEL_MANIFEST_NAME, require, sha256, write_json
from compiler.qwen35.config import TEXT_DIR
from tools.prepare import common


MODEL_ID = "Qwen/Qwen3.5-0.8B"
REVISIONS = {
    MODEL_ID: "2fc06364715b967f1860aea9cf38778875588b17",
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
}
MODEL_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def default_model_dir(model_id):
    require(
        isinstance(model_id, str) and model_id in REVISIONS, f"Unsupported checkpoint: {model_id}"
    )
    return TEXT_DIR.parent.parent / model_id.rsplit("-", 1)[1].lower() / "text"


def prepare(directory, model=MODEL_ID, cache_dir=None):
    """Extract only language-model tensors and publish one verified text checkpoint."""
    require(isinstance(model, str) and model in REVISIONS, f"Unsupported checkpoint: {model}")
    from safetensors import safe_open
    from safetensors.torch import save_file

    from compiler.qwen35.model import _text_config

    with common.checkpoint_directory(directory) as staging:
        hub = common.Hub(model, REVISIONS[model], cache_dir=cache_dir)
        _text_config(hub.fetch("config.json"))
        index = json.loads(hub.fetch("model.safetensors.index.json").read_text())["weight_map"]
        prefix = "model.language_model."
        names = {name: shard for name, shard in index.items() if name.startswith(prefix)}
        require(bool(names), "No text weights found")
        weights, shards = {}, {}
        for shard in sorted(set(names.values())):
            path = hub.fetch(shard)
            shards[shard] = hub.verified[shard]["sha256"]
            with safe_open(path, framework="pt", device="cpu") as reader:
                for name, filename in names.items():
                    if filename == shard:
                        weights["model." + name.removeprefix(prefix)] = reader.get_tensor(name)
        save_file(weights, staging / "text.safetensors")
        files = {name: hub.copy(name, staging / name)["sha256"] for name in MODEL_FILES}
        files["text.safetensors"] = sha256(staging / "text.safetensors")
        manifest = dict(
            model=model,
            revision=REVISIONS[model],
            text_parameters=len(weights),
            source_shards=shards,
            files=files,
        )
        write_json(staging / MODEL_MANIFEST_NAME, manifest)
    return manifest


def verify_model(directory):
    """Verify a prepared checkpoint offline when explicitly requested."""
    directory = Path(directory)
    manifest = json.loads((directory / MODEL_MANIFEST_NAME).read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Prepared checkpoint manifest must be an object")
    model_id = manifest.get("model")
    if (
        not isinstance(model_id, str)
        or model_id not in REVISIONS
        or manifest.get("revision") != REVISIONS[model_id]
    ):
        raise ValueError("Prepared checkpoint must match a supported pinned revision")
    if not isinstance(manifest.get("files"), dict) or set(manifest["files"]) != {
        *MODEL_FILES,
        "text.safetensors",
    }:
        raise ValueError("Prepared checkpoint file manifest is incomplete or unsupported")
    for name, expected in manifest["files"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    return manifest
