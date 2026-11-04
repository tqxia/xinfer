"""Qwen3.5 ABI checks after reading the common package metadata format."""

from compiler.common.artifacts import require


def validate_model_metadata(metadata):
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
