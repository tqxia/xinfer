"""Pinned Qwen3.6 configuration and HF FP8 dependency initialization."""

from pathlib import Path

from compiler.common.artifacts import require
from compiler.qwen35.config import MAX_PREFILL, configure

MODEL_ID = "Qwen/Qwen3.6-35B-A3B-FP8"
REVISION = "c653bba46dfa06e9658dcc0b0dc87fe4685f14a2"
TEXT_DIR = Path("work/qwen36/35b-a3b-fp8/text")
ARTIFACT_DIR = Path("artifacts/qwen36/35b-a3b-fp8")

__all__ = [
    "MODEL_ID",
    "REVISION",
    "TEXT_DIR",
    "ARTIFACT_DIR",
    "MAX_PREFILL",
    "configure",
    "configure_hf_fp8",
]


def configure_hf_fp8():
    """Warm HF's official cache using the pinned dependency, without changing math."""
    configure()
    from transformers.integrations import finegrained_fp8 as hf_fp8

    from compiler.common.fp8 import load_fp8_kernels

    require(hf_fp8.is_kernels_available(), "HF FP8 requires the configured kernels package")
    kernel = load_fp8_kernels()

    def resolve(name):
        require(name == "finegrained-fp8", f"Unexpected HF kernel dependency: {name}")
        return kernel

    # Initialization happens before forward/export. Restore HF's resolver while
    # retaining its official cached entry points; no runtime download is needed.
    original = hf_fp8.lazy_load_kernel
    hf_fp8.lazy_load_kernel = resolve
    try:
        loaded = hf_fp8.load_finegrained_fp8_kernel()
    finally:
        hf_fp8.lazy_load_kernel = original
    require(
        loaded.matmul is kernel.matmul_2d
        and loaded.batched_matmul is kernel.matmul_batched
        and loaded.grouped_matmul is kernel.matmul_grouped,
        "HF cached another FP8 kernel build; initialize a fresh process",
    )
    return loaded
