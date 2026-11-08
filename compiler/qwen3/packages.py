"""Validate the model-specific Qwen3 package ABI."""

from compiler.common.artifacts import require


MEGAQWEN_PROFILE = {"decode_backend": "megaqwen"}


def decode_backend(metadata):
    """The compiled decode implementation is a property of the package pair."""
    return metadata["model_specific"].get("decode_backend", "aoti")


def validate_model_metadata(metadata):
    """Check the Qwen3 KV-only ABI after common metadata validation."""
    extra = metadata["model_specific"]
    require(
        extra == {} or extra == MEGAQWEN_PROFILE,
        "Unsupported package metadata: model_specific",
    )
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
            item["shape"] == [1, 8, metadata["cache_length"], 128], "Unsupported Flash cache shape"
        )
    if decode_backend(metadata) == "megaqwen":
        require(
            len(state) == 56 and metadata["vocab_size"] == 151936,
            "MegaQwen decode requires Qwen3-0.6B geometry",
        )
        require(metadata["cache_length"] <= 40960, "MegaQwen context capacity exceeds 40960")
