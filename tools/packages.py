"""Package ABI and shared-weight bundle checks for benchmark and validation tools."""

from pathlib import Path

from compiler.common.artifacts import require, sha256
from compiler.common.metadata import validate_metadata


def validate_shared_weights(directory, metadata):
    """Validate the complete bundle without allocating CUDA storage."""
    from compiler.common.weights import read_shared_weight_manifest

    require(
        isinstance(metadata, dict) and "shared_weights" in metadata,
        "Shared weights metadata is required; re-export both packages",
    )
    if "model_type" in metadata:
        validate_metadata(metadata)
        for kind, entry in metadata["entries"].items():
            package = Path(directory) / f"{kind}.pt2"
            require(package.is_file(), f"Package file is missing: {kind}")
            require(package.stat().st_size == entry["bytes"], "Package file size mismatch")
            require(sha256(package) == entry["sha256"], f"Package checksum mismatch: {kind}")
        for name, entry in metadata["tokenizer"].items():
            tokenizer_path = Path(directory) / name
            require(tokenizer_path.is_file(), f"Tokenizer file is missing: {name}")
            require(
                tokenizer_path.stat().st_size == entry["bytes"],
                f"Tokenizer file size mismatch: {name}",
            )
            require(
                sha256(tokenizer_path) == entry["sha256"],
                f"Tokenizer checksum mismatch: {name}",
            )
    manifest = read_shared_weight_manifest(directory, metadata["shared_weights"])
    if "model_type" in metadata:
        model_type = {"qwen3": "qwen3", "qwen3.5": "qwen35", "qwen3.8": "qwen38"}.get(
            manifest["identity"].get("model")
        )
        require(model_type == metadata["model_type"], "Package model identity mismatch")
    return manifest


def validate_qwen3_metadata(metadata):
    """Check the Qwen3 KV-only ABI after common metadata validation."""
    state = metadata["state"]
    require(bool(state) and len(state) % 2 == 0, "Unsupported package metadata: state")
    for layer in range(len(state) // 2):
        key, value = state[2 * layer : 2 * layer + 2]
        require(
            key["layer"] == layer
            and value["layer"] == layer
            and key["kind"] == "key"
            and value["kind"] == "value"
            and key["dtype"] == "bfloat16"
            and value["dtype"] == "bfloat16",
            "Unsupported package metadata: state",
        )
        require(key["shape"] == value["shape"], "K/V shapes differ")


def validate_qwen35_metadata(metadata):
    """Keep adapter assumptions separate from the shared metadata reader."""
    state = metadata["state"]
    require(bool(state) and len(state) % 2 == 0, "Invalid state count")
    for layer in range(len(state) // 2):
        first, second = state[2 * layer : 2 * layer + 2]
        expected = {"conv": "recurrent", "key": "value"}.get(first["kind"])
        require(
            expected is not None
            and first["layer"] == layer
            and second["layer"] == layer
            and second["kind"] == expected,
            "Invalid state order or dtype",
        )
        if first["kind"] == "key":
            require(first["shape"] == second["shape"], "K/V shapes differ")


def validate_model_metadata(metadata):
    """Check the selected model ABI after common metadata validation."""
    validators = {
        "qwen3": validate_qwen3_metadata,
        "qwen35": validate_qwen35_metadata,
        "qwen38": validate_qwen35_metadata,
    }
    validators[metadata["model_type"]](metadata)
