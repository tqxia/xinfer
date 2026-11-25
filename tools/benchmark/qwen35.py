"""Qwen3.5 checkpoint loading and tokenization for the shared benchmark."""

from tools.prepare.qwen35 import default_model_dir as prepared_model_dir
from tools.benchmark.common import CONTINUATION_TEXT, PROMPT_TEXT, hf_eager_loader


def default_model_dir(metadata):
    return prepared_model_dir(metadata["checkpoint"]["model"])


def prepare_tokens(model_dir):
    from tools.prepare.common import encode_prompt, load_tokenizer

    tokenizer = load_tokenizer(model_dir)
    return encode_prompt(tokenizer, PROMPT_TEXT), tokenizer.encode(
        CONTINUATION_TEXT, add_special_tokens=False
    )


def eager_loader():
    from compiler.qwen35.config import configure
    from compiler.qwen35.model import load_model

    configure()
    return hf_eager_loader(load_model)
