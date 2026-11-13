"""Experimental singleton projections with FP32 row reductions."""

import torch
import triton
import triton.language as tl


ROWS_PER_PROGRAM = 4
NUM_WARPS = 4


@triton.jit
def projection_gemv_kernel(
    X,
    Weight,
    Output,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    columns = tl.arange(0, BLOCK_K)
    x = tl.load(X + columns, columns < K, other=0).to(tl.float32)
    weights = tl.load(
        Weight + rows[:, None] * K + columns[None, :],
        (rows[:, None] < N) & (columns[None, :] < K),
        other=0,
    ).to(tl.float32)
    result = tl.sum(weights * x[None, :], axis=1)
    tl.store(Output + rows, result, rows < N)


@torch.library.triton_op("xinfer::projection_gemv", mutates_args={})
def projection_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return F.linear-shaped output for one contiguous input row; no mutation."""
    if x.device.type != "cuda" or weight.device != x.device:
        raise ValueError("Projection GEMV requires colocated CUDA tensors")
    if x.dtype not in (torch.bfloat16, torch.float32) or weight.dtype != x.dtype:
        raise ValueError("Projection GEMV requires matching BF16 or FP32 tensors")
    if x.ndim < 2 or weight.ndim != 2 or any(size != 1 for size in x.shape[:-1]):
        raise ValueError("Projection GEMV requires one input row and a 2D weight")
    if x.shape[-1] != weight.shape[1] or x.stride(-1) != 1 or not weight.is_contiguous():
        raise ValueError("Projection GEMV requires matching contiguous reduction dimensions")
    n, k = weight.shape
    if n <= 0 or k <= 0:
        raise ValueError("Projection GEMV requires positive dimensions")
    output = x.new_empty((*x.shape[:-1], n))
    rows_per_program = 1 if k >= 2048 else ROWS_PER_PROGRAM
    torch.library.wrap_triton(projection_gemv_kernel)[(triton.cdiv(n, rows_per_program),)](
        x,
        weight,
        output,
        k,
        n,
        triton.next_power_of_2(k),
        rows_per_program,
        num_warps=NUM_WARPS,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return output
