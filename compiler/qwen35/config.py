"""Pinned Qwen3.5 checkpoints, execution settings and DeltaNet state shapes."""

import importlib.metadata
import importlib.util
from pathlib import Path


MODEL_ID = "Qwen/Qwen3.5-0.8B"
REVISIONS = {
    MODEL_ID: "2fc06364715b967f1860aea9cf38778875588b17",
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
}
TEXT_DIR = Path("work/qwen35/0.8b/text")
ARTIFACT_DIR = Path("artifacts/qwen35/0.8b")
PREFILL = 64
CACHE_LENGTH = 1024


def deltanet_state_shapes(config):
    key_width = config.linear_num_key_heads * config.linear_key_head_dim
    value_width = config.linear_num_value_heads * config.linear_value_head_dim
    return ((1, 2 * key_width + value_width, config.linear_conv_kernel_dim),
            (1, config.linear_num_value_heads, config.linear_key_head_dim, config.linear_value_head_dim))


def configure():
    import torch

    if importlib.metadata.version("transformers") != "5.20.2":
        raise RuntimeError("This adapter targets transformers==5.20.2")
    for package in ("fla", "causal_conv1d"):
        if importlib.util.find_spec(package) is not None:
            raise RuntimeError(f"Use an environment without {package} for the PyTorch fallback baseline")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
