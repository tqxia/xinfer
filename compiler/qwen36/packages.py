"""The complete Qwen3.6 FP8 text package contract."""

from compiler.common.artifacts import require
from compiler.common.fp8 import KERNEL_REPO, KERNEL_REVISION
from compiler.qwen35.packages import validate_model_metadata as validate_hybrid_metadata
from compiler.qwen36.config import MODEL_ID, REVISION


def validate_model_metadata(metadata):
    """Check the pinned FP8 model after shared schema and hybrid-state checks."""
    validate_hybrid_metadata(metadata)
    extra = metadata["model_specific"]
    expected = dict(
        attention_backend="flash_attention",
        fp8_backend="triton_op",
        fp8_format="e4m3fn",
        fp8_block_size=[128, 128],
        fp8_scale_dtype="bfloat16",
        fp8_kernel=dict(repo=KERNEL_REPO, revision=KERNEL_REVISION),
        linear_packing=dict(enabled=False),
    )
    require(
        all(extra.get(name) == value for name, value in expected.items()),
        "Unsupported Qwen3.6 FP8 backend",
    )
    require(
        metadata["model_type"] == "qwen36"
        and metadata["checkpoint"]["model_id"] == MODEL_ID
        and metadata["checkpoint"]["revision"] == REVISION
        and metadata["vocab_size"] == 248320
        and metadata["eos_token_ids"] == [248046, 248044]
        and 512 <= metadata["cache_length"] <= 262144
        and metadata["token_range"]
        == ([2, 512] if metadata["entry_point"] == "prefill" else [1, 1]),
        "Unsupported Qwen3.6 checkpoint or execution bounds",
    )
    state = metadata["state"]
    require(len(state) == 80, "Qwen3.6 packages require all 40 decoder layers")
    for layer in range(40):
        kinds = ("key", "value") if layer % 4 == 3 else ("conv", "recurrent")
        shapes = (
            ([1, 2, metadata["cache_length"], 256],) * 2
            if layer % 4 == 3
            else ([1, 8192, 4], [1, 32, 128, 128])
        )
        for item, kind, shape in zip(state[2 * layer : 2 * layer + 2], kinds, shapes):
            require(
                item["kind"] == kind and item["shape"] == shape,
                "Unsupported Qwen3.6 state geometry",
            )
