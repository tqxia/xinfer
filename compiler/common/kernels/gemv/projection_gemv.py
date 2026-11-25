"""Dynamic small-batch (B=1..8) projections with weight-stationary FP32 reductions."""

import torch
import triton
import triton.language as tl


MAX_BATCH_SIZE = 8
ROWS_PER_PROGRAM = 4
NUM_WARPS = 4


@triton.jit
def projection_gemv_kernel(
    X,
    Weight,
    Output,
    M,
    stride_xm,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    columns = tl.arange(0, BLOCK_K)
    row_mask = rows < N
    col_mask = columns < K
    weights = tl.load(
        Weight + rows[:, None] * K + columns[None, :],
        row_mask[:, None] & col_mask[None, :],
        other=0,
    ).to(tl.float32)
    for batch_idx in range(M):
        x = tl.load(X + batch_idx * stride_xm + columns, col_mask, other=0).to(tl.float32)
        result = tl.sum(weights * x[None, :], axis=1)
        tl.store(Output + batch_idx * N + rows, result, row_mask)


@torch.library.triton_op("xinfer::projection_gemv", mutates_args={})
def projection_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return F.linear-shaped output for 1..8 contiguous input rows; no mutation."""
    if x.device.type != "cuda" or weight.device != x.device:
        raise ValueError("Projection GEMV requires colocated CUDA tensors")
    if x.dtype not in (torch.bfloat16, torch.float32) or weight.dtype != x.dtype:
        raise ValueError("Projection GEMV requires matching BF16 or FP32 tensors")
    if x.ndim < 2 or weight.ndim != 2 or any(size != 1 for size in x.shape[1:-1]):
        raise ValueError("Projection GEMV requires [B, 1..., K] input and a 2D weight")
    batch = x.shape[0]
    if batch < 1 or batch > MAX_BATCH_SIZE:
        raise ValueError(f"Projection GEMV requires 1..{MAX_BATCH_SIZE} input rows")
    stride_xm = x.stride(0)
    if (
        x.shape[-1] != weight.shape[1]
        or x.stride(-1) != 1
        or stride_xm < 0
        or not weight.is_contiguous()
    ):
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
        batch,
        stride_xm,
        k,
        n,
        triton.next_power_of_2(k),
        rows_per_program,
        num_warps=NUM_WARPS,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return output
