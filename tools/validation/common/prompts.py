"""Validation prompt text shared by reference and runtime checks."""

PARAGRAPH = "The capital of France is Paris. A small compiler can run a language model. "

_CHAT_PROMPTS = {
    "qwen3": (
        ("capital", "What is the capital of France? Answer in one short sentence."),
        ("arithmetic", "What is 17 + 25? Reply with only the number."),
        ("chinese", "请用一句简短的中文解释什么是机器学习。"),
    ),
    "qwen35": (
        ("capital", "What is the capital of France? Answer briefly."),
        ("arithmetic", "What is 17 + 25? Answer with only the number."),
        ("chinese", "用一句话解释什么是编译器。"),
    ),
}


def chat_prompts(model_type):
    """Return independent named prompts while preserving each model's fixtures."""
    try:
        prompts = _CHAT_PROMPTS[model_type]
    except KeyError:
        raise ValueError(f"Unsupported validation model: {model_type}") from None
    return [dict(name=name, text=text) for name, text in prompts]
