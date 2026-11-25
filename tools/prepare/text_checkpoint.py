"""Shared extraction and verification of Qwen multimodal text checkpoints."""

import json
from pathlib import Path

from compiler.common.artifacts import MODEL_MANIFEST_NAME, sha256, write_json, require
from tools.prepare import common


MODEL_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def prepare(directory, model, *, revision, model_files, text_config, cache_dir=None):
    """Extract only language-model tensors and publish one verified text checkpoint."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    with common.checkpoint_directory(directory) as staging:
        hub = common.Hub(model, revision, cache_dir=cache_dir)
        config = text_config(hub.fetch("config.json"))
        index = json.loads(hub.fetch("model.safetensors.index.json").read_text())["weight_map"]
        prefix = "model.language_model."
        names = {
            name: shard
            for name, shard in index.items()
            if name.startswith(prefix)
            or (not config.tie_word_embeddings and name == "lm_head.weight")
        }
        require(bool(names), "No text weights found")
        weights, shards = {}, {}
        for shard in sorted(set(names.values())):
            path = hub.fetch(shard)
            shards[shard] = hub.verified[shard]["sha256"]
            with safe_open(path, framework="pt", device="cpu") as reader:
                for name, filename in names.items():
                    if filename == shard:
                        destination = (
                            "model." + name.removeprefix(prefix)
                            if name.startswith(prefix)
                            else name
                        )
                        weights[destination] = reader.get_tensor(name)
        save_file(weights, staging / "text.safetensors")
        files = {name: hub.copy(name, staging / name)["sha256"] for name in model_files}
        files["text.safetensors"] = sha256(staging / "text.safetensors")
        manifest = dict(
            model=model,
            revision=revision,
            text_parameters=len(weights),
            source_shards=shards,
            files=files,
        )
        write_json(staging / MODEL_MANIFEST_NAME, manifest)
    return manifest


def verify_model(directory, *, revisions, model_files):
    """Verify a prepared checkpoint offline when explicitly requested."""
    directory = Path(directory)
    manifest = json.loads((directory / MODEL_MANIFEST_NAME).read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Prepared checkpoint manifest must be an object")
    model_id = manifest.get("model")
    if (
        not isinstance(model_id, str)
        or model_id not in revisions
        or manifest.get("revision") != revisions[model_id]
    ):
        raise ValueError("Prepared checkpoint must match a supported pinned revision")
    if not isinstance(manifest.get("files"), dict) or set(manifest["files"]) != {
        *model_files,
        "text.safetensors",
    }:
        raise ValueError("Prepared checkpoint file manifest is incomplete or unsupported")
    for name, expected in manifest["files"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    return manifest
