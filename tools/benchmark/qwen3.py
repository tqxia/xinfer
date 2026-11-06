"""Qwen3 checkpoint loading and tokenization for the shared benchmark."""

from compiler.qwen3.packages import validate_model_metadata as validate_model_metadata
from compiler.qwen3.config import DEFAULT_MODEL
from tools.benchmark.common import CONTINUATION_TEXT, PROMPT_TEXT, hf_eager_loader


def default_model_dir(metadata):
    return DEFAULT_MODEL


def prepare_tokens(model_dir):
    from tools.prepare.common import encode_prompt, load_tokenizer

    tokenizer = load_tokenizer(model_dir)
    return encode_prompt(tokenizer, PROMPT_TEXT), tokenizer.encode(
        CONTINUATION_TEXT, add_special_tokens=False
    )


def eager_loader():
    from compiler.qwen3.model import load_model

    return hf_eager_loader(load_model)
