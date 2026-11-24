"""Package ABI and shared-weight bundle checks for benchmark and validation tools."""

from pathlib import Path

from compiler.common.artifacts import require, sha256
from compiler.common.metadata import validate_metadata


def validate_shared_weights(directory, prefill, decode):
    """Validate the complete bundle without allocating CUDA storage."""
    from compiler.common.weights import SHARED_WEIGHT_OPTIONS, read_shared_weight_manifest

    require(
        "shared_weights" in prefill and "shared_weights" in decode,
        "Shared weights metadata is required; re-export both packages",
    )
    require(
        prefill["shared_weights"] == decode["shared_weights"], "Shared weights contracts differ"
    )
    for metadata in (prefill, decode):
        options = metadata.get("inductor_configs", {})
        require(
            all(
                key in options and options[key] == value
                for key, value in SHARED_WEIGHT_OPTIONS.items()
            ),
            "Shared weights require weightless package compiler options",
        )
        if "model_type" in metadata:
            validate_metadata(metadata)
            package = Path(directory) / f"{metadata['entry_point']}.pt2"
            require(package.is_file(), f"Package file is missing: {metadata['entry_point']}")
            require(
                package.stat().st_size == metadata["package_bytes"], "Package file size mismatch"
            )
            require(
                sha256(package) == metadata["package_sha256"],
                f"Package checksum mismatch: {metadata['entry_point']}",
            )
    manifest = read_shared_weight_manifest(directory, decode["shared_weights"])
    if "model_type" in decode:
        model_type = {"qwen3": "qwen3", "qwen3.5": "qwen35"}.get(manifest["identity"].get("model"))
        require(model_type == decode["model_type"], "Package model identity mismatch")
    return manifest


def validate_qwen3_metadata(metadata):
    """Check the Qwen3 KV-only ABI after common metadata validation."""
    require(metadata["model_specific"] == {}, "Unsupported package metadata: model_specific")
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
            and value["dtype"] == "bfloat16"
            and key["update"] == "inplace"
            and value["update"] == "inplace",
            "Unsupported package metadata: state",
        )
        require(key["shape"] == value["shape"], "K/V shapes differ")


def validate_qwen35_metadata(metadata):
    """Keep adapter assumptions separate from the shared metadata reader."""
    extra = metadata["model_specific"]
    require(extra.get("delta_backend") == "triton_op", "Unsupported DeltaNet backend")
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
    validators = {"qwen3": validate_qwen3_metadata, "qwen35": validate_qwen35_metadata}
    validators[metadata["model_type"]](metadata)
