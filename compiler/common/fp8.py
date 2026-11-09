"""Block-scaled FP8 projections using pinned upstream Triton operators.

Call load_fp8_kernels before export or eager execution. The launch wrappers
reuse upstream Triton kernels and decompose during AOTI compilation; deployed
packages do not load this Python module or download kernels.
"""

import torch
from torch.nn import functional as F

from compiler.common.artifacts import require

KERNEL_REPO = "kernels-community/finegrained-fp8"
KERNEL_REVISION = "03ae31933313f38fae2b050dbe2731024386c369"
_KERNEL = None
_LINEAR_KERNEL = None
_GROUPED_KERNEL = None


def load_fp8_kernels():
    """Preload the immutable CUDA build, honoring HF_HUB_OFFLINE/KERNELS_CACHE."""
    global _KERNEL, _LINEAR_KERNEL, _GROUPED_KERNEL
    if _KERNEL is None:
        from importlib import import_module

        from kernels import get_kernel

        kernel = get_kernel(KERNEL_REPO, revision=KERNEL_REVISION, backend="cuda")
        require(
            all(callable(getattr(kernel, name, None)) for name in ("matmul_2d", "matmul_grouped")),
            "The pinned FP8 kernel build is missing its projection operators",
        )
        module = import_module(kernel.matmul_2d.__module__)
        _LINEAR_KERNEL = module.w8a8_block_dynamic_fp8_matmul_kernel
        module = import_module(kernel.matmul_grouped.__module__)
        _GROUPED_KERNEL = module.w8a8_block_dynamic_fp8_matmul_grouped_kernel
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


@torch.library.triton_op("xinfer::fp8_grouped", mutates_args={})
def _grouped(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    offsets: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    rows, columns = x.shape
    num_experts, width, _ = weight.shape
    output = x.new_empty((rows, width))
    # Keep the upstream GPU tile layout and data-independent upper-bound grid;
    # only replace its symbolic next_power_of_2 tile selection with M=16.
    tile_offsets = torch.cumsum((counts + 15) // 16, dim=0).to(torch.int32)
    grid = ((rows + 15) // 16 + num_experts, width // 128)
    with torch.cuda.device(x.device):
        torch.library.wrap_triton(_GROUPED_KERNEL)[grid](
            x,
            weight,
            output,
            scale,
            offsets,
            tile_offsets,
            rows,
            width,
            columns,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            weight.stride(2),
            weight.stride(1),
            output.stride(0),
            output.stride(1),
            scale.stride(0),
            scale.stride(2),
            scale.stride(1),
            offsets.stride(0),
            tile_offsets.stride(0),
            num_experts=num_experts,
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=128,
            NUM_EXPERTS_BIT_LENGTH=num_experts.bit_length(),
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


def experts(x, expert_ids, routing_weights, gate_up, gate_up_scale_inv, down, down_scale_inv):
    """Apply routed SwiGLU experts; the caller supplies valid top-k expert IDs.

    Inputs are BF16 x[T,H], int64 IDs[T,K], BF16 routing weights[T,K], E4M3
    gate_up[E,2I,H] and down[E,H,I], with their BF16/FP32 block scales. Routing
    and the shared expert stay outside this function. No expert weights are
    gathered into per-token copies and no routing values are read on the host.
    """
    _weight(gate_up, gate_up_scale_inv, 3)
    _weight(down, down_scale_inv, 3)
    num_experts, twice_intermediate, hidden = gate_up.shape
    require(
        num_experts > 0
        and twice_intermediate % 2 == 0
        and down.shape == (num_experts, hidden, twice_intermediate // 2)
        and down.device == gate_up.device,
        "FP8 expert projections must have matching gate/up/down geometry",
    )
    require(
        x.ndim == 2
        and x.dtype == torch.bfloat16
        and x.is_cuda
        and x.device == gate_up.device
        and x.shape[1] == hidden
        and x.is_contiguous()
        and expert_ids.ndim == 2
        and expert_ids.shape[0] == x.shape[0]
        and 1 <= expert_ids.shape[1] <= num_experts
        and expert_ids.dtype == torch.int64
        and routing_weights.shape == expert_ids.shape
        and routing_weights.dtype == torch.bfloat16
        and expert_ids.device == x.device
        and routing_weights.device == x.device,
        "FP8 experts require CUDA BF16 tokens/weights and matching int64 top-k routing",
    )
    _loaded()
    top_k = expert_ids.shape[1]
    sorted_ids, permutation = torch.sort(expert_ids.reshape(-1))
    selected = x[permutation // top_k]
    counts = torch.histc(sorted_ids.int(), bins=num_experts, min=0, max=num_experts - 1)
    offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
    projected = _grouped(selected, gate_up, gate_up_scale_inv, offsets, counts)
    gate, up = projected.chunk(2, dim=-1)
    activated = (F.silu(gate) * up).contiguous()
    projected = _grouped(activated, down, down_scale_inv, offsets, counts)
    weighted = projected * routing_weights.reshape(-1)[permutation, None]
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel(), device=x.device)
    return weighted[inverse].view(x.shape[0], top_k, hidden).sum(dim=1)
