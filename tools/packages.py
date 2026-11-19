"""Model-specific package ABI checks shared by benchmark and validation tools."""

from compiler.common.artifacts import require


def validate_qwen3_metadata(metadata):
    """Check the Qwen3 KV-only ABI after common metadata validation."""
    require(metadata["model_specific"] == {}, "Unsupported package metadata: model_specific")
    state = metadata["state"]
    require(bool(state) and len(state) % 2 == 0, "Unsupported package metadata: state")
    for index, item in enumerate(state):
        require(
            item["layer"] == index // 2
            and item["kind"] == ("value" if index % 2 else "key")
            and item["dtype"] == "bfloat16"
            and item["update"] == "inplace",
            "Unsupported package metadata: state",
        )
        require(
            item["shape"] == [metadata["state"][0]["shape"][0], 8, metadata["cache_length"], 128],
            "Unsupported Flash cache shape",
        )


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
