"""Qwen3.8 FP8 checkpoint rules and loading for the shared text architecture."""

import json
from pathlib import Path

import torch
from transformers import GenerationConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextConfig

from compiler.common.artifacts import require
from compiler.common.fp8_adapter import load_weights, validate_quantization
from compiler.qwen35.model import create_text_model, finalize_text_model
from compiler.qwen38.config import TEXT_DIR


def _text_config(path):
    document = json.loads(path.read_text())
    quantization = document.get("quantization_config")
    require(isinstance(quantization, dict), "Qwen3.8 text export requires an FP8 checkpoint")
    validate_quantization(quantization)
    config = Qwen3_5TextConfig(**document["text_config"])
    require(
        not config.tie_word_embeddings and config.rope_parameters["rope_type"] == "default",
        "Qwen3.8 FP8 text export requires untied embeddings and default RoPE",
    )
    config.quantization_config = quantization
    config._attn_implementation = "sdpa"
    return config


def load_model(directory=TEXT_DIR, dtype=torch.bfloat16):
    require(dtype == torch.bfloat16, "FP8 models require BF16 activations")
    directory = Path(directory)
    config = _text_config(directory / "config.json")
    model = create_text_model(config)
    load_weights(model, directory / "text.safetensors")
    model.generation_config = GenerationConfig.from_pretrained(directory, local_files_only=True)
    return finalize_text_model(model)
