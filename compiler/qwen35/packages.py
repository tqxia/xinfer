"""Qwen3.5 ABI checks after reading the common package metadata format."""

from compiler.common.artifacts import require


def validate_model_metadata(metadata):
    """Keep adapter assumptions separate from the shared metadata reader."""
    extra = metadata["model_specific"]
    require(extra.get("delta_backend") == "triton_op", "Unsupported DeltaNet backend")
    kinds = extra.get("layer_types")
    require(isinstance(kinds, list) and bool(kinds) and len(metadata["state"]) == 2 * len(kinds),
            "Invalid state count")
    for layer, kind in enumerate(kinds):
        require(kind in ("linear_attention", "full_attention"), "Unsupported layer type")
        expected = ("conv", "recurrent") if kind == "linear_attention" else ("key", "value")
        for index, name in enumerate(expected):
            item = metadata["state"][2 * layer + index]
            require(item["layer"] == layer and item["kind"] == name and
                    item["name"] == f"layer_{layer}.{name}", "Invalid state order or dtype")
        if kind == "full_attention":
            require(metadata["state"][2 * layer]["shape"] == metadata["state"][2 * layer + 1]["shape"],
                    "K/V shapes differ")
    require(metadata["output_order"] == ["last_logits"] + [s["name"] for s in metadata["state"]],
            "Invalid tensor ABI order")
