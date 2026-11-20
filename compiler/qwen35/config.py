"""Qwen3.5 default paths, limits and execution settings."""

from pathlib import Path


TEXT_DIR = Path("work/qwen35/0.8b/text")
ARTIFACT_DIR = Path("artifacts/qwen35/0.8b")
MAX_PREFILL = 512
MAX_BATCH_SIZE = 8


def configure():
    import torch

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
