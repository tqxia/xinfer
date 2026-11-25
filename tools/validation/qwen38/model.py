"""Qwen3.8 FP8 references and frontend hooks for the shared validation chain."""

from compiler.common.kernels.gemm.finegrained_fp8 import (
    KERNEL_REPO,
    KERNEL_REVISION,
    KERNEL_VERSION,
)
from compiler.qwen38.model import load_model
from compiler.qwen38.export import export_step
from tools.prepare.qwen38 import default_model_dir as prepared_model_dir
from tools.validation.qwen35.model import Suite as SharedSuite


def default_model_dir(metadata):
    return prepared_model_dir(metadata["checkpoint"]["model"])


class Suite(SharedSuite):
    model_type = "qwen38"

    def __init__(self, model_dir, metadata):
        super().__init__(
            model_dir,
            metadata,
            model_loader=load_model,
            step_exporter=export_step,
        )
        self.reference_protocol = dict(
            dtype="bfloat16",
            weight_dtype="float8_e4m3fn",
            cache="HF StaticCache",
            teacher_inputs="matched tokens and partitions",
            quantization="dynamic activation FP8; 128x128 weight blocks",
            kernel_repo=KERNEL_REPO,
            kernel_version=KERNEL_VERSION,
            kernel_revision=KERNEL_REVISION,
            scope="FP8 checkpoint execution; not a comparison with unquantized weights",
        )
