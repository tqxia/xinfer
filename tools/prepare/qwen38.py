"""Qwen3.8 FP8 checkpoint pin and text preparation policy."""

from compiler.common.artifacts import require
from compiler.qwen38.config import TEXT_DIR
from tools.prepare import text_checkpoint


REVISIONS = {"Qwen/Qwen3.8-27B-FP8": "c3a462cc3d38cf351f3b26cbd763469eeb74ef81"}
MODEL_FILES = (*text_checkpoint.MODEL_FILES, "generation_config.json")


def default_model_dir(model_id):
    require(
        isinstance(model_id, str) and model_id in REVISIONS, f"Unsupported checkpoint: {model_id}"
    )
    return TEXT_DIR


def prepare(directory, model, cache_dir=None):
    require(isinstance(model, str) and model in REVISIONS, f"Unsupported checkpoint: {model}")
    from compiler.qwen38.model import _text_config

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
