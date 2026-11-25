"""Block-scaled FP8 projections and pinned HF reference kernel loading.

Exported graphs execute xinfer's own ``xinfer::fp8_linear`` Triton kernel with
a fixed 32-row tile so dynamic batch and token dimensions stay symbolic. AOTI
packages the compiled kernel directly, so deployed bundles need neither Python
nor a Hub connection. ``load_fp8_kernels`` pins the immutable upstream Hub
commit used by unmodified HF ``FP8Linear`` reference execution.
"""

from pathlib import Path

import torch
import triton
import triton.language as tl

from compiler.common.artifacts import require

KERNEL_REPO = "kernels-community/finegrained-fp8"
KERNEL_VERSION = 4
KERNEL_REVISION = "03ae31933313f38fae2b050dbe2731024386c369"
_KERNEL = None


def _verify_snapshot_revision(kernel):
    path = getattr(kernel, "__file__", None)
    if not path:
        return
    parts = Path(path).resolve().parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        revision = parts[index + 1] if index + 1 < len(parts) else None
        require(
            revision == KERNEL_REVISION,
            f"Expected FP8 kernel revision {KERNEL_REVISION}, got {revision}",
        )


def load_fp8_kernels():
    """Preload the pinned CUDA build for HF reference execution, honoring offline cache."""
    global _KERNEL
    if _KERNEL is None:
        from kernels import get_kernel

        kernel = get_kernel(KERNEL_REPO, revision=KERNEL_REVISION, backend="cuda")
        require(
            callable(getattr(kernel, "matmul_2d", None)),
            "The FP8 kernel build is missing its projection operators",
        )
        _verify_snapshot_revision(kernel)
        _KERNEL = kernel
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


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=warps, num_stages=stages)
        for warps in (2, 4, 8, 16)
        for stages in (2, 3, 4)
    ],
    key=["N", "K", "BLOCK_SIZE_M"],
)
@triton.jit
def _fp8_linear_kernel(
    A,
    B,
    C,
    Bs,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bs_k,
    stride_bs_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    bs_ptrs = Bs + (offs_bn // BLOCK_SIZE_N) * stride_bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a_raw = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0).to(tl.float32)
        a_scale = tl.max(tl.abs(a_raw), axis=1) / 448.0
        a = (a_raw / tl.maximum(a_scale[:, None], 1e-12)).to(tl.float8e4nv)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        b_scale = tl.load(bs_ptrs)
        accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        bs_ptrs += stride_bs_k

    c = accumulator.to(C.dtype.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@torch.library.triton_op("xinfer::fp8_linear", mutates_args={})
def _linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # Keep a fixed 32-row tile so dynamic batch and token dimensions stay symbolic.
    rows = x.numel() // x.shape[-1]
    width, columns = weight.shape
    output = x.new_empty((*x.shape[:-1], width))
    grid = ((rows + 31) // 32, width // 128)
    with torch.cuda.device(x.device):
        torch.library.wrap_triton(_fp8_linear_kernel)[grid](
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
    return _linear(x, weight, weight_scale_inv)
