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
            item["shape"] == [1, 8, metadata["cache_length"], 128], "Unsupported Flash cache shape"
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


def validate_qwen36_metadata(metadata):
    """Check the pinned FP8 model after shared schema and hybrid-state checks."""
    from compiler.common.kernels.fp8 import KERNEL_REPO, KERNEL_REVISION
    from compiler.qwen36.config import MODEL_ID, REVISION

    validate_qwen35_metadata(metadata)
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


def validate_model_metadata(metadata):
    """Check the selected model ABI after common metadata validation."""
    validators = {
        "qwen3": validate_qwen3_metadata,
        "qwen35": validate_qwen35_metadata,
        "qwen36": validate_qwen36_metadata,
    }
    validators[metadata["model_type"]](metadata)
