"""Qwen3.5 checkpoint pins and text preparation policy."""

from compiler.common.artifacts import require
from compiler.qwen35.config import TEXT_DIR
from tools.prepare import text_checkpoint
from tools.prepare.text_checkpoint import MODEL_FILES


REVISIONS = {
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
}


def default_model_dir(model_id):
    require(
        isinstance(model_id, str) and model_id in REVISIONS, f"Unsupported checkpoint: {model_id}"
    )
    return TEXT_DIR.parent.parent / model_id.rsplit("-", 1)[1].lower() / "text"


def prepare(directory, model, cache_dir=None):
    require(isinstance(model, str) and model in REVISIONS, f"Unsupported checkpoint: {model}")
    from compiler.qwen35.model import _text_config

    return text_checkpoint.prepare(
        directory,
        model,
        revision=REVISIONS[model],
        model_files=MODEL_FILES,
        text_config=_text_config,
        cache_dir=cache_dir,
    )


def verify_model(directory):
    return text_checkpoint.verify_model(directory, revisions=REVISIONS, model_files=MODEL_FILES)
