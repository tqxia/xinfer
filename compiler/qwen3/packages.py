"""Validate Qwen3 attention packages and load their native dependency."""

from compiler.common.attention import load_attention_ops, native_attention_library
from compiler.common.artifacts import require
from compiler.common.metadata import validate_metadata


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


def load_package_attention_ops(metadata, native=None):
    validate_metadata(metadata, model_type="qwen3")
    validate_model_metadata(metadata)
    library = native_attention_library(native) if native is not None else None
    if native is not None and library is None:
        raise ValueError(
            "Qwen3 runtime is missing its required native attention library; rebuild the runtime"
        )
    load_attention_ops(library, model="qwen3")
