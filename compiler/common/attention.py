"""Load the shared native prefix-Flash operator and register its shape contract."""

import os
from pathlib import Path

from compiler.common.artifacts import default_build_dir, require


ATTENTION_LIBRARY = "libxinfer_attention_ops.so"
ATTENTION_C_SHIM = """AOTITorchError aoti_torch_cuda_flash_attention(
    AtenTensorHandle q, AtenTensorHandle k, AtenTensorHandle v,
    AtenTensorHandle extent, double scale, AtenTensorHandle* output)"""
_loaded = None


def cache_extent(valid_length):
    """Only this CPU tensor's shape is read; +1 avoids AOTI's size-one guards."""
    import torch

    return torch.empty(valid_length + 1, dtype=torch.uint8)


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
        torch._check(q.shape[0] == 1)
        torch._check(q.shape[1] > 0)
        torch._check(k.shape[1] > 0)
        torch._check(q.shape[1] % k.shape[1] == 0)
        torch._check(q.shape[3] >= 8)
        torch._check(q.shape[3] <= 256)
        torch._check(q.shape[3] % 8 == 0)
        torch._check(q.shape[2] >= 1)
        torch._check(q.shape[2] <= 512)
        torch._check(k.shape[2] > 0)
        torch._check(k.shape == (1, k.shape[1], k.shape[2], q.shape[3]))
        torch._check(v.shape == k.shape)
        torch._check(q.dtype == k.dtype == v.dtype == torch.bfloat16)
        torch._check(q.device.type == "cuda" and k.device == q.device and v.device == q.device)
        torch._check(kv_extent.device.type == "cpu" and kv_extent.dtype == torch.uint8)
        torch._check(kv_extent.dim() == 1)
        return q.new_empty(q.shape)

    _loaded = library


def attention_library():
    """Return the implementation selected by the eager/export adapter."""
    require(_loaded is not None, "Load the native attention library before export")
    return _loaded


def attention_c_shims():
    """Return Inductor's call declarations for the registered attention operator."""
    import torch

    return {torch.ops.xinfer.flash_attention.default: [ATTENTION_C_SHIM]}
