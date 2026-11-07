"""Validate a compiled cache capacity against the model's declared context ceiling."""

from compiler.common.artifacts import require


def model_context_limit(config):
    """Read the declared ceiling from a checkpoint dictionary or HF configuration.

    Multimodal checkpoints declare the language model's ceiling in text_config.
    RoPE scaling factors are deliberately not applied to this declared value.
    """

    def field(value, name):
        return value.get(name) if isinstance(value, dict) else getattr(value, name, None)

    text = field(config, "text_config")
    ceiling = field(text if text is not None else config, "max_position_embeddings")
    require(
        type(ceiling) is int and 0 < ceiling < 2**63 - 1,
        "Model configuration must declare a positive integer max_position_embeddings",
    )
    return ceiling


def resolve_context_length(config, override=None):
    """Use the model ceiling by default, or validate an explicit compiled capacity."""
    ceiling = model_context_limit(config)
    length = ceiling if override is None else override
    require(
        type(length) is int and 512 <= length <= ceiling,
        f"--max-context-length must be an integer in 512..{ceiling}",
    )
    return length
