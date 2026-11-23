"""Fused recurrence and chunked prefill with graph-level shape dispatch.

Contract: B=1..8, Hk=16, Hv=16 or 32, K=V=128; T is a runtime scalar (1..512).
Query lengths are CUDA int32 [B]; only their valid prefixes advance state.
Short prefill and decode stay fused; long prefill uses tiled FP32 chunk transforms.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from compiler.qwen35.kernels.chunked import launch_chunked

DIM, VALUE_TILE = 128, 32
SHORT_THRESHOLD = 97
OPTIONS = dict(num_warps=4, num_stages=1)


@triton.jit
def _recurrent_step(q, k, v, g, beta, state, D: tl.constexpr):
    q = q * tl.rsqrt(tl.sum(q * q, 0) + 1e-6) * (D**-0.5)
    k = k * tl.rsqrt(tl.sum(k * k, 0) + 1e-6)
    state = state * tl.exp(g)
    prediction = tl.sum(state * k[:, None], 0)
    delta = (v - prediction) * beta
    state = state + k[:, None] * delta[None, :]
    out = tl.sum(state * q[:, None], 0)
    return out, state


@triton.jit
def _causal_conv(
    QKV, Conv, Weight, NextConv, channels, t, length, write_state, N: tl.constexpr, C: tl.constexpr
):
    # Scalar taps avoid a BF16 shared-layout conversion that miscompiled V on
    # the tested Triton build. Both history/input loads stay explicitly masked.
    total = tl.full(channels.shape, 0, tl.float32)
    for tap in tl.static_range(C):
        sample = t - (C - 1) + tap
        old = tl.load(Conv + channels * C + C + sample, mask=sample < 0, other=0)
        current = tl.load(
            QKV + sample * N + channels, mask=(sample >= 0) & (sample < length), other=0
        )
        history = tl.where(sample < 0, old, current)
        # Empty rows retain all taps; nonempty rows save their last real window.
        original = tl.load(Conv + channels * C + tap, mask=length == 0, other=0)
        tl.store(
            NextConv + channels * C + tap,
            tl.where(length == 0, original, history),
            mask=(t == tl.maximum(length - 1, 0)) & write_state,
        )
        weight = tl.load(Weight + channels * C + tap)
        total += history.to(tl.float32) * weight.to(tl.float32)
    dtype = QKV.dtype.element_ty
    # HF conv1d accumulates in FP32, then rounds before the SiLU activation.
    value = total.to(dtype).to(tl.float32)
    return (value * tl.sigmoid(value)).to(dtype).to(tl.float32)


@triton.jit
def fused_kernel(
    QKV,
    B,
    A,
    Conv,
    Initial,
    Weight,
    ALog,
    DTBias,
    Output,
    NextConv,
    Final,
    Lengths,
    tokens,
    HK: tl.constexpr,
    HV: tl.constexpr,
    D: tl.constexpr,
    BV: tl.constexpr,
    C: tl.constexpr,
):
    head, tile, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    group: tl.constexpr = HV // HK
    channels: tl.constexpr = (2 * HK + HV) * D
    key_head = head // group
    length = tl.load(Lengths + batch)
    QKV += batch * tokens * channels
    B += batch * tokens * HV
    A += batch * tokens * HV
    Conv += batch * channels * C
    NextConv += batch * channels * C
    Initial += batch * HV * D * D
    Final += batch * HV * D * D
    Output += batch * tokens * HV * D
    rows = tl.arange(0, D)
    cols = tile * BV + tl.arange(0, BV)
    offsets = head * D * D + rows[:, None] * D + cols[None, :]
    state = tl.load(Initial + offsets)
    scale = -libdevice.exp(tl.load(ALog + head).to(tl.float32))
    dt = tl.load(DTBias + head).to(tl.float32)
    for t in range(tokens):
        # Grouped value heads share Q/K history. One head and tile own its writes.
        owns_qk = (head % group == 0) & (tile == 0)
        q = _causal_conv(
            QKV, Conv, Weight, NextConv, key_head * D + rows, t, length, owns_qk, channels, C
        )
        k = _causal_conv(
            QKV, Conv, Weight, NextConv, (HK + key_head) * D + rows, t, length, owns_qk, channels, C
        )
        v = _causal_conv(
            QKV, Conv, Weight, NextConv, (2 * HK + head) * D + cols, t, length, True, channels, C
        )
        beta = (
            tl.sigmoid(tl.load(B + t * HV + head).to(tl.float32))
            .to(B.dtype.element_ty)
            .to(tl.float32)
        )
        x = tl.load(A + t * HV + head).to(tl.float32) + dt
        softplus = tl.where(x > 20.0, x, libdevice.log1p(libdevice.exp(x)))
        g = scale * softplus
        out, updated = _recurrent_step(q, k, v, g, beta, state, D)
        state = tl.where(t < length, updated, state)
        tl.store(Output + (t * HV + head) * D + cols, tl.where(t < length, out, 0))
    tl.store(Final + offsets, state)


@triton.jit
def prepare_kernel(
    QKV,
    B,
    A,
    Conv,
    Weight,
    ALog,
    DTBias,
    Q,
    K,
    V,
    G,
    Beta,
    NextConv,
    Lengths,
    tokens,
    HK: tl.constexpr,
    HV: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
):
    t, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    group: tl.constexpr = HV // HK
    channels: tl.constexpr = (2 * HK + HV) * D
    key_head = head // group
    length = tl.load(Lengths + batch)
    QKV += batch * tokens * channels
    B += batch * tokens * HV
    A += batch * tokens * HV
    Conv += batch * channels * C
    NextConv += batch * channels * C
    Q += batch * tokens * HV * D
    K += batch * tokens * HV * D
    V += batch * tokens * HV * D
    G += batch * tokens * HV
    Beta += batch * tokens * HV
    rows = tl.arange(0, D)
    owns_qk = head % group == 0
    q = _causal_conv(
        QKV, Conv, Weight, NextConv, key_head * D + rows, t, length, owns_qk, channels, C
    )
    k = _causal_conv(
        QKV, Conv, Weight, NextConv, (HK + key_head) * D + rows, t, length, owns_qk, channels, C
    )
    v = _causal_conv(
        QKV, Conv, Weight, NextConv, (2 * HK + head) * D + rows, t, length, True, channels, C
    )
    q = q * tl.rsqrt(tl.sum(q * q, 0) + 1e-6) * (D**-0.5)
    k = k * tl.rsqrt(tl.sum(k * k, 0) + 1e-6)
    beta = (
        tl.sigmoid(tl.load(B + t * HV + head).to(tl.float32)).to(B.dtype.element_ty).to(tl.float32)
    )
    scale = -libdevice.exp(tl.load(ALog + head).to(tl.float32))
    x = tl.load(A + t * HV + head).to(tl.float32) + tl.load(DTBias + head).to(tl.float32)
    softplus = tl.where(x > 20.0, x, libdevice.log1p(libdevice.exp(x)))
    # Materialize one Q/K vector per value head for the existing chunked scan.
    base = (t * HV + head) * D + rows
    tl.store(Q + base, tl.where(t < length, q, 0))
    tl.store(K + base, tl.where(t < length, k, 0))
    tl.store(V + base, tl.where(t < length, v, 0))
    tl.store(G + t * HV + head, tl.where(t < length, scale * softplus, 0))
    tl.store(Beta + t * HV + head, tl.where(t < length, beta, 0))


@torch.library.triton_op("xinfer::delta_cond_short", mutates_args={})
def short(
    qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    conv: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    query_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run fused preparation/recurrence without a device predicate."""
    batch, tokens = qkv.shape[:2]
    value_heads = state.shape[1]
    key_heads = (qkv.shape[2] // DIM - value_heads) // 2
    output = qkv.new_empty((batch, tokens, value_heads, DIM))
    next_conv, final = torch.empty_like(conv), torch.empty_like(state)
    torch.library.wrap_triton(fused_kernel)[(value_heads, DIM // VALUE_TILE, batch)](
        qkv,
        b,
        a,
        conv,
        state,
        weight,
        A_log,
        dt_bias,
        output,
        next_conv,
        final,
        query_lengths,
        tokens,
        key_heads,
        value_heads,
        DIM,
        VALUE_TILE,
        4,
        **OPTIONS,
    )
    return output, next_conv, final


@torch.library.triton_op("xinfer::delta_cond_long", mutates_args={})
def long(
    qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    conv: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    query_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the C64 chunk path with the measured B3 transform implementation."""
    batch, tokens = qkv.shape[:2]
    value_heads = state.shape[1]
    key_heads = (qkv.shape[2] // DIM - value_heads) // 2
    output = qkv.new_empty((batch, tokens, value_heads, DIM))
    next_conv, final = torch.empty_like(conv), torch.empty_like(state)
    shape = (batch, tokens, value_heads, DIM)
    q, k, v = (torch.empty(shape, device=qkv.device, dtype=torch.float32) for _ in range(3))
    g, beta = (
        torch.empty((batch, tokens, value_heads), device=qkv.device, dtype=torch.float32)
        for _ in range(2)
    )
    torch.library.wrap_triton(prepare_kernel)[(tokens, value_heads, batch)](
        qkv,
        b,
        a,
        conv,
        weight,
        A_log,
        dt_bias,
        q,
        k,
        v,
        g,
        beta,
        next_conv,
        query_lengths,
        tokens,
        key_heads,
        value_heads,
        DIM,
        4,
        **OPTIONS,
    )
    launch_chunked(q, k, v, g, beta, state, output, final, query_lengths)
    return output, next_conv, final


def _short_branch(*inputs):
    return short(*inputs)


def _long_branch(*inputs):
    return long(*inputs)


def fused(
    qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    conv: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    query_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select fused recurrence or chunked FP32 prefill from the padded width.

    query_lengths contains valid prefix counts in 0..T, or defaults to all T
    tokens for unpadded calls. Zero-length rows preserve both input states.
    """
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
    if key_heads != 16 or value_heads not in (16, 32) or qkv.shape[2] != channels:
        raise ValueError("Expected Hk=16, Hv=16 or 32 and K=V=128")
    if conv.shape != (batch, channels, 4) or weight.shape != (channels, 4):
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
    qkv, b, a, conv, weight, A_log, dt_bias, state = (x.contiguous() for x in (*tensors, state))
    if query_lengths is None:
        query_lengths = torch.full((batch,), tokens, device=qkv.device, dtype=torch.int32)
    if (
        query_lengths.shape != (batch,)
        or query_lengths.dtype != torch.int32
        or query_lengths.device != qkv.device
    ):
        raise ValueError("Expected CUDA int32 query_lengths [B]")
    operands = (qkv, b, a, conv, state, weight, A_log, dt_bias, query_lengths.contiguous())
    if tokens == 1:
        return short(*operands)
    return torch.cond(tokens <= SHORT_THRESHOLD, _short_branch, _long_branch, operands)
