"""Qwen3.5 checkpoint loading and tokenization for the shared benchmark."""

from compiler.common.artifacts import require
from compiler.qwen35.config import TEXT_DIR
from tools.prepare.qwen35 import REVISIONS
from compiler.qwen35.packages import validate_model_metadata as validate_model_metadata
from tools.benchmark.common import CONTINUATION_TEXT, PROMPT_TEXT, hf_eager_loader


def default_model_dir(metadata):
    model_id = metadata["decode"]["checkpoint"]["model_id"]
    require(model_id in REVISIONS, "Unsupported Qwen3.5 checkpoint")
    return TEXT_DIR.parent.parent / model_id.rsplit("-", 1)[1].lower() / "text"


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
