"""Load the native DeltaNet operator and register its in-place shape contract."""

import os
from pathlib import Path

from compiler.common.artifacts import default_build_dir, require


DELTANET_LIBRARY = "libxinfer_deltanet_ops.so"
BATCHED_DELTANET_C_SHIM = """AOTITorchError aoti_torch_cuda_batched_deltanet(
    AtenTensorHandle qkv, AtenTensorHandle b, AtenTensorHandle a,
    AtenTensorHandle conv, AtenTensorHandle state,
    AtenTensorHandle weight, AtenTensorHandle A_log, AtenTensorHandle dt_bias,
    AtenTensorHandle cu_query, AtenTensorHandle cuda_cu_query,
    AtenTensorHandle* output)"""
DIM = 128
KEY_HEADS = 16
CONV_WIDTH = 4
_loaded = None


def load_deltanet_ops(library=None):
    global _loaded
    if library is None and _loaded is not None:
        return
    selected = library or os.environ.get("XINFER_DELTANET_OPS_LIBRARY")
    if not selected:
        selected = default_build_dir() / DELTANET_LIBRARY
    library = Path(selected).resolve()
    if _loaded is not None:
        if _loaded != library:
            raise ValueError("Cannot load two different DeltaNet libraries in one Python process")
        return
    if not library.is_file():
        raise FileNotFoundError(f"Build the native DeltaNet library first: {library}")
    import torch

    torch.ops.load_library(str(library))

    @torch.library.register_fake("xinfer::batched_deltanet")
    def fake_batched(qkv, b, a, conv, state, weight, A_log, dt_bias, cu_query, cuda_cu_query):
        torch._check(qkv.dim() == 3)
        torch._check(qkv.shape[0] >= 1)
        torch._check(qkv.shape[0] <= 8)
        torch._check(qkv.shape[1] >= 1)
        torch._check(qkv.shape[1] <= 512)
        torch._check(b.dim() == 3 and a.dim() == 3)
        torch._check(b.shape[0] == qkv.shape[0] and b.shape[1] == qkv.shape[1])
        torch._check(a.shape == b.shape)
        value_heads = b.shape[2]
        torch._check(value_heads == 16 or value_heads == 32 or value_heads == 48)
        channels = (2 * KEY_HEADS + value_heads) * DIM
        torch._check(qkv.shape[2] == channels)
        torch._check(conv.shape == (qkv.shape[0], channels, CONV_WIDTH))
        torch._check(weight.shape == (channels, CONV_WIDTH))
        torch._check(state.shape == (qkv.shape[0], value_heads, DIM, DIM))
        torch._check(A_log.shape == (value_heads,) and dt_bias.shape == (value_heads,))
        torch._check(qkv.dtype == torch.bfloat16 or qkv.dtype == torch.float32)
        torch._check(
            b.dtype == qkv.dtype
            and a.dtype == qkv.dtype
            and conv.dtype == qkv.dtype
            and weight.dtype == qkv.dtype
        )
        torch._check(A_log.dtype == qkv.dtype or A_log.dtype == torch.float32)
        torch._check(dt_bias.dtype == qkv.dtype or dt_bias.dtype == torch.float32)
        torch._check(state.dtype == torch.float32)
        torch._check(
            qkv.device.type == "cuda"
            and b.device == qkv.device
            and a.device == qkv.device
            and conv.device == qkv.device
            and state.device == qkv.device
            and weight.device == qkv.device
            and A_log.device == qkv.device
            and dt_bias.device == qkv.device
        )
        torch._check(cu_query.device.type == "cpu" and cu_query.dtype == torch.int32)
        torch._check(cu_query.shape == (qkv.shape[0] + 1,))
        torch._check(
            cuda_cu_query.shape == cu_query.shape
            and cuda_cu_query.device == qkv.device
            and cuda_cu_query.dtype == torch.int32
        )
        return qkv.new_empty((qkv.shape[0], qkv.shape[1], value_heads, DIM))

    _loaded = library


def deltanet_library():
    """Return the implementation selected by the eager/export adapter."""
    require(_loaded is not None, "Load the native DeltaNet library before export")
    return _loaded


def deltanet_c_shims():
    """Return Inductor's call declarations for the registered DeltaNet operator."""
    import torch

    return {
        torch.ops.xinfer.batched_deltanet.default: [BATCHED_DELTANET_C_SHIM],
    }


def cumulative_query_lengths(query_lengths, batch, tokens, device):
    """Build CPU/CUDA cumulative query lengths [B+1] for standalone DeltaNet calls."""
    import torch

    if query_lengths is None:
        cu_query = (torch.arange(batch + 1, dtype=torch.int32) * tokens).to(torch.int32)
        return cu_query, cu_query.to(device)
    if (
        query_lengths.ndim != 1
        or query_lengths.shape[0] != batch
        or query_lengths.dtype != torch.int32
    ):
        raise ValueError("Expected int32 query_lengths [B]")
    cpu_lengths = query_lengths if query_lengths.device.type == "cpu" else query_lengths.cpu()
    cu_query = torch.cat(
        (torch.zeros((1,), dtype=torch.int32), cpu_lengths.cumsum(0, dtype=torch.int32))
    )
    return cu_query, cu_query.to(device)


def batched_deltanet(
    qkv,
    b,
    a,
    conv,
    state,
    weight,
    A_log,
    dt_bias,
    cu_query=None,
    cuda_cu_query=None,
    *,
    query_lengths=None,
):
    """Update conv and recurrent state in place and return the core output [B,T,Hv,128]."""
    import torch

    if qkv.ndim != 3 or not 1 <= qkv.shape[0] <= 8:
        raise ValueError("Expected QKV [B,T,channels] with B=1..8")
    batch, tokens = qkv.shape[:2]
    if tokens < 1 or tokens > 512:
        raise ValueError("Token length must be 1..512")
    if b.ndim != 3 or b.shape[0] != batch or b.shape[1] != tokens or a.shape != b.shape:
        raise ValueError("Expected matching A/B [B,T,value_heads]")
    value_heads = b.shape[2]
    key_heads = (qkv.shape[2] // DIM - value_heads) // 2
    channels = (2 * key_heads + value_heads) * DIM
    if key_heads != KEY_HEADS or value_heads not in (16, 32, 48) or qkv.shape[2] != channels:
        raise ValueError("Expected Hk=16, Hv=16, 32 or 48 and K=V=128")
    if conv.shape != (batch, channels, CONV_WIDTH) or weight.shape != (channels, CONV_WIDTH):
        raise ValueError("Expected matching width-4 convolution state and weights")
    if (
        state.shape != (batch, value_heads, DIM, DIM)
        or A_log.shape != (value_heads,)
        or dt_bias.shape != (value_heads,)
    ):
        raise ValueError("Expected per-value-head recurrent state and decay parameters")
    tensors = (qkv, b, a, conv, weight, A_log, dt_bias)
    if (
        qkv.dtype not in (torch.bfloat16, torch.float32)
        or any(x.dtype != qkv.dtype for x in (b, a, conv, weight))
        or any(x.dtype not in (qkv.dtype, torch.float32) for x in (A_log, dt_bias))
    ):
        raise ValueError("Expected matching BF16 or FP32 inputs and parameters")
    if state.dtype != torch.float32:
        raise ValueError("Recurrent state must be FP32")
    if qkv.device.type != "cuda" or any(x.device != qkv.device for x in (*tensors, state)):
        raise ValueError("Expected CUDA tensors on one device")
    if not conv.is_contiguous() or not state.is_contiguous():
        raise ValueError("In-place DeltaNet state buffers must be contiguous")
    if cu_query is None or cuda_cu_query is None:
        if cu_query is not None or cuda_cu_query is not None:
            raise ValueError("Provide both cu_query and cuda_cu_query, or neither")
        cu_query, cuda_cu_query = cumulative_query_lengths(query_lengths, batch, tokens, qkv.device)
    load_deltanet_ops()
    return torch.ops.xinfer.batched_deltanet(
        qkv.contiguous(),
        b.contiguous(),
        a.contiguous(),
        conv,
        state,
        weight.contiguous(),
        A_log.contiguous(),
        dt_bias.contiguous(),
        cu_query.contiguous(),
        cuda_cu_query.contiguous(),
    )
