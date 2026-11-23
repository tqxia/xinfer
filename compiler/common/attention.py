"""Load native FlashAttention operators and register their shape contracts."""

import os
from pathlib import Path

from compiler.common.artifacts import default_build_dir, require


ATTENTION_LIBRARY = "libxinfer_attention_ops.so"
ATTENTION_C_SHIM = """AOTITorchError aoti_torch_cuda_flash_attention(
    AtenTensorHandle q, AtenTensorHandle k, AtenTensorHandle v,
    AtenTensorHandle extent, double scale, AtenTensorHandle* output)"""
BATCHED_ATTENTION_C_SHIM = """AOTITorchError aoti_torch_cuda_batched_flash_attention(
    AtenTensorHandle q, AtenTensorHandle k, AtenTensorHandle v,
    AtenTensorHandle offsets, AtenTensorHandle cuda_offsets, AtenTensorHandle blocks,
    double scale, AtenTensorHandle* output)"""
_loaded = None


def cache_extent(valid_length):
    """Only this CPU tensor's shape is read; +1 avoids AOTI's size-one guards."""
    import torch

    return torch.empty(valid_length + 1, dtype=torch.uint8)


def batched_attention_metadata(kv_lengths, query_lengths, device):
    """Build cumulative query/KV lengths once per step, shared by every layer.

    CPU offsets drive packing without a device readback. Their CUDA copy drives
    FlashAttention and the KV writer. Each request owns one fixed cache block.
    """
    import torch

    lengths = torch.stack((query_lengths, kv_lengths))
    offsets = torch.cat(
        (torch.zeros((2, 1), dtype=torch.int32), lengths.cumsum(1, dtype=torch.int32)), dim=1
    )
    blocks = torch.arange(kv_lengths.shape[0], device=device, dtype=torch.int32)[:, None]
    return offsets, offsets.to(device), blocks


def load_attention_ops(library=None, *, model=None):
    global _loaded
    if library is None and _loaded is not None:
        return
    selected = library or os.environ.get("XINFER_ATTENTION_OPS_LIBRARY")
    if not selected:
        if model is None:
            raise ValueError("Select a model to locate its default native attention library")
        selected = default_build_dir(model) / ATTENTION_LIBRARY
    library = Path(selected).resolve()
    if _loaded is not None:
        if _loaded != library:
            raise ValueError("Cannot load two different attention libraries in one Python process")
        return
    if not library.is_file():
        raise FileNotFoundError(f"Build the native attention library first: {library}")
    import torch

    torch.ops.load_library(str(library))

    @torch.library.register_fake("xinfer::flash_attention")
    def fake(q, k, v, kv_extent, scale):
        # Valid-prefix length is checked natively; output metadata depends only on Q.
        torch._check(q.dim() == 4 and k.dim() == 4 and v.dim() == 4)
        torch._check(q.shape[0] > 0)
        torch._check(q.shape[1] > 0)
        torch._check(k.shape[1] > 0)
        torch._check(q.shape[1] % k.shape[1] == 0)
        torch._check(q.shape[3] >= 8)
        torch._check(q.shape[3] <= 256)
        torch._check(q.shape[3] % 8 == 0)
        torch._check(q.shape[2] >= 1)
        torch._check(q.shape[2] <= 512)
        torch._check(k.shape[2] > 0)
        torch._check(k.shape == (q.shape[0], k.shape[1], k.shape[2], q.shape[3]))
        torch._check(v.shape == k.shape)
        torch._check(q.dtype == k.dtype == v.dtype == torch.bfloat16)
        torch._check(q.device.type == "cuda" and k.device == q.device and v.device == q.device)
        torch._check(kv_extent.device.type == "cpu" and kv_extent.dtype == torch.uint8)
        torch._check(kv_extent.dim() == 1)
        return q.new_empty(q.shape)

    # Prefix-only libraries may carry only the original flash_attention operator.
    if hasattr(torch.ops.xinfer, "batched_flash_attention"):

        @torch.library.register_fake("xinfer::batched_flash_attention")
        def fake_batched(q, k, v, offsets, cuda_offsets, blocks, scale):
            torch._check(offsets.device.type == "cpu" and offsets.dtype == torch.int32)
            torch._check(offsets.shape == (2, q.shape[0] + 1))
            torch._check(
                cuda_offsets.shape == offsets.shape
                and cuda_offsets.device == q.device
                and cuda_offsets.dtype == torch.int32
            )
            torch._check(
                blocks.shape == (q.shape[0], 1)
                and blocks.device == q.device
                and blocks.dtype == torch.int32
            )
            return fake(q, k, v, torch.empty(2, dtype=torch.uint8), scale)

    _loaded = library


def attention_library():
    """Return the implementation selected by the eager/export adapter."""
    require(_loaded is not None, "Load the native attention library before export")
    return _loaded


def attention_c_shims():
    """Return Inductor's call declarations for the registered attention operators."""
    import torch

    return {
        torch.ops.xinfer.flash_attention.default: [ATTENTION_C_SHIM],
        torch.ops.xinfer.batched_flash_attention.default: [BATCHED_ATTENTION_C_SHIM],
    }
