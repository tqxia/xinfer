"""Block-scaled FP8 projections using pinned upstream Triton operators.

Call load_fp8_kernels before export or eager execution. The launch wrappers
reuse upstream Triton kernels and decompose during AOTI compilation; deployed
packages do not load this Python module or download kernels.
"""

import torch

from compiler.common.artifacts import require

KERNEL_REPO = "kernels-community/finegrained-fp8"
KERNEL_REVISION = "03ae31933313f38fae2b050dbe2731024386c369"
_KERNEL = None
_LINEAR_KERNEL = None


def load_fp8_kernels():
    """Preload the immutable CUDA build, honoring HF_HUB_OFFLINE/KERNELS_CACHE."""
    global _KERNEL, _LINEAR_KERNEL
    if _KERNEL is None:
        from importlib import import_module

        from kernels import get_kernel

        kernel = get_kernel(KERNEL_REPO, revision=KERNEL_REVISION, backend="cuda")
        require(
            callable(getattr(kernel, "matmul_2d", None)),
            "The pinned FP8 kernel build is missing its projection operators",
        )
        module = import_module(kernel.matmul_2d.__module__)
        _LINEAR_KERNEL = module.w8a8_block_dynamic_fp8_matmul_kernel
        _KERNEL = kernel
    return _KERNEL


def _loaded():
    require(_KERNEL is not None, "Call load_fp8_kernels() before export or execution")
    return _KERNEL


def _weight(weight, scale, ndim):
    require(
        weight.ndim == ndim and weight.dtype == torch.float8_e4m3fn and weight.is_contiguous(),
        "FP8 weights must be contiguous E4M3 tensors",
    )
    rows, columns = weight.shape[-2:]
    require(
        rows > 0 and columns > 0 and rows % 128 == 0 and columns % 128 == 0,
        "FP8 projection dimensions must be positive multiples of 128",
    )
    require(
        scale.shape == (*weight.shape[:-2], rows // 128, columns // 128)
        and scale.dtype in (torch.bfloat16, torch.float32)
        and scale.device == weight.device
        and scale.is_contiguous(),
        "FP8 weight scales must be contiguous BF16/FP32 tensors with 128x128 blocks",
    )


@torch.library.triton_op("xinfer::fp8_linear", mutates_args={})
def _linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # The upstream host wrapper picks next_power_of_2(M), which specializes a
    # dynamic token dimension. Keep its kernel math and fix only the launch tile.
    rows = x.numel() // x.shape[-1]
    width, columns = weight.shape
    output = x.new_empty((*x.shape[:-1], width))
    grid = ((rows + 31) // 32, width // 128)
    with torch.cuda.device(x.device):
        torch.library.wrap_triton(_LINEAR_KERNEL)[grid](
            x,
            weight,
            output,
            scale,
            rows,
            width,
            columns,
            x.stride(-2),
            x.stride(-1),
            weight.stride(1),
            weight.stride(0),
            output.stride(-2),
            output.stride(-1),
            scale.stride(1),
            scale.stride(0),
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=128,
            GROUP_SIZE_M=8,
        )
    return output


def linear(x, weight, weight_scale_inv):
    """BF16 x[..., K] times E4M3 weight[N, K], with dynamic activation quantization."""
    _weight(weight, weight_scale_inv, 2)
    require(
        x.ndim >= 2
        and x.dtype == torch.bfloat16
        and x.is_cuda
        and x.device == weight.device
        and x.shape[-1] == weight.shape[-1]
        and x.is_contiguous(),
        "FP8 linear requires contiguous CUDA BF16 inputs matching the weight",
    )
    _loaded()
    return _linear(x, weight, weight_scale_inv)
