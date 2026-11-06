"""Validate the model-specific Qwen3 package ABI."""

from compiler.common.artifacts import require


def validate_model_metadata(metadata):
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
        require(item["shape"] == [1, 8, 1024, 128], "Unsupported Flash cache shape")
