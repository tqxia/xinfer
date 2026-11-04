"""Validate Qwen3 attention packages and load their native dependency."""

from compiler.common.attention import load_attention_ops, native_attention_library
from compiler.common.artifacts import require
from compiler.common.metadata import validate_metadata


def validate_model_metadata(metadata):
    """Check the Qwen3 KV-only ABI after common metadata validation."""
    extra = metadata["model_specific"]
    integers = ("layers", "kv_heads", "head_dim", "cache_padding")
    vectors = ("cache_shape", "cache_stride")
    require(
        all(type(extra.get(key)) is int and -(2**63) <= extra[key] < 2**63 for key in integers)
        and all(
            isinstance(extra.get(key), list)
            and all(type(value) is int and -(2**63) <= value < 2**63 for value in extra[key])
            for key in vectors
        ),
        "Unsupported package metadata: model_specific",
    )
    layers, heads, dim = (extra[key] for key in ("layers", "kv_heads", "head_dim"))
    require(0 < layers <= (2**63 - 1) // 2 and heads > 0 and dim > 0, "Invalid model dimensions")
    require(extra["cache_padding"] == 0, "Unsupported cache padding")
    shape, stride = extra["cache_shape"], extra["cache_stride"]
    require(shape == [1, 8, 1024, 128], "Unsupported Flash cache shape")
    require(shape == [1, heads, metadata["cache_length"], dim], "Invalid KV cache shape")
    require(
        stride == [heads * metadata["cache_length"] * dim, metadata["cache_length"] * dim, dim, 1],
        "Unexpected KV cache strides",
    )
    state = metadata["state"]
    require(len(state) == 2 * layers, "Unsupported package metadata: state")
    for index, item in enumerate(state):
        kind = "value" if index % 2 else "key"
        require(
            item["layer"] == index // 2
            and item["kind"] == kind
            and item["name"] == f"layer_{index // 2}.{kind}"
            and item["shape"] == shape
            and item["stride"] == stride
            and item["dtype"] == "bfloat16"
            and item["update"] == "inplace",
            "Unsupported package metadata: state",
        )
    require(
        metadata["output_order"] == ["last_logits"], "Unsupported package metadata: output_order"
    )


def load_package_attention_ops(metadata, native=None):
    validate_metadata(metadata, model_type="qwen3")
    validate_model_metadata(metadata)
    library = native_attention_library(native) if native is not None else None
    if native is not None and library is None:
        raise ValueError(
            "Qwen3 runtime is missing its required native attention library; rebuild the runtime"
        )
    load_attention_ops(library, model="qwen3")
