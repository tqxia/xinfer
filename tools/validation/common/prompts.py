"""Validation prompt text shared by reference and runtime checks."""

PARAGRAPH = "The capital of France is Paris. A small compiler can run a language model. "

_CHAT_PROMPTS = (
    ("capital", "What is the capital of France? Answer briefly."),
    ("arithmetic", "What is 17 + 25? Answer with only the number."),
    ("chinese", "用一句话解释什么是编译器。"),
)


def chat_prompts():
    """Return independent named chat prompts shared across all validation suites."""
    return [dict(name=name, text=text) for name, text in _CHAT_PROMPTS]
