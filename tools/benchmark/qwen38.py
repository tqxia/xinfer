"""Qwen3.8 checkpoint defaults and original FP8 HF benchmark loading."""

from tools.prepare.qwen38 import default_model_dir as prepared_model_dir
from tools.benchmark.qwen35 import prepare_tokens as prepare_tokens
from tools.benchmark.common import hf_eager_loader


def default_model_dir(metadata):
    return prepared_model_dir(metadata["checkpoint"]["model"])


def eager_loader():
    from compiler.qwen35.config import configure
    from compiler.qwen38.model import load_model

    configure()
    return hf_eager_loader(load_model)
