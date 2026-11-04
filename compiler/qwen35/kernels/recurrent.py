"""Fused recurrence and chunked prefill with graph-level shape dispatch.

Contract: B=1, Hk=16, Hv=16 or 32, K=V=128; T is a runtime scalar (1..512).
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
    QKV, Conv, Weight, NextConv, channels, t, tokens, write_state, N: tl.constexpr, C: tl.constexpr
):
    # Scalar taps avoid a BF16 shared-layout conversion that miscompiled V on
    # the tested Triton build. Both history/input loads stay explicitly masked.
    total = tl.full(channels.shape, 0, tl.float32)
    for tap in tl.static_range(C):
        sample = t - (C - 1) + tap
        old = tl.load(Conv + channels * C + C + sample, mask=sample < 0, other=0)
        current = tl.load(QKV + sample * N + channels, mask=sample >= 0, other=0)
        history = tl.where(sample < 0, old, current)
        tl.store(NextConv + channels * C + tap, history, mask=(t == tokens - 1) & write_state)
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
    tokens,
    HK: tl.constexpr,
    HV: tl.constexpr,
    D: tl.constexpr,
    BV: tl.constexpr,
    C: tl.constexpr,
):
    head, tile = tl.program_id(0), tl.program_id(1)
    group: tl.constexpr = HV // HK
    channels: tl.constexpr = (2 * HK + HV) * D
    key_head = head // group
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
            QKV, Conv, Weight, NextConv, key_head * D + rows, t, tokens, owns_qk, channels, C
        )
        k = _causal_conv(
            QKV, Conv, Weight, NextConv, (HK + key_head) * D + rows, t, tokens, owns_qk, channels, C
        )
        v = _causal_conv(
            QKV, Conv, Weight, NextConv, (2 * HK + head) * D + cols, t, tokens, True, channels, C
        )
        beta = (
            tl.sigmoid(tl.load(B + t * HV + head).to(tl.float32))
            .to(B.dtype.element_ty)
            .to(tl.float32)
        )
        x = tl.load(A + t * HV + head).to(tl.float32) + dt
        softplus = tl.where(x > 20.0, x, libdevice.log1p(libdevice.exp(x)))
        g = scale * softplus
        out, state = _recurrent_step(q, k, v, g, beta, state, D)
        tl.store(Output + (t * HV + head) * D + cols, out)
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
    tokens,
    HK: tl.constexpr,
    HV: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
):
    t, head = tl.program_id(0), tl.program_id(1)
    group: tl.constexpr = HV // HK
    channels: tl.constexpr = (2 * HK + HV) * D
    key_head = head // group
    rows = tl.arange(0, D)
    owns_qk = head % group == 0
    q = _causal_conv(
        QKV, Conv, Weight, NextConv, key_head * D + rows, t, tokens, owns_qk, channels, C
    )
    k = _causal_conv(
        QKV, Conv, Weight, NextConv, (HK + key_head) * D + rows, t, tokens, owns_qk, channels, C
    )
    v = _causal_conv(
        QKV, Conv, Weight, NextConv, (2 * HK + head) * D + rows, t, tokens, True, channels, C
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
    tl.store(Q + base, q)
    tl.store(K + base, k)
    tl.store(V + base, v)
    tl.store(G + t * HV + head, scale * softplus)
    tl.store(Beta + t * HV + head, beta)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run fused preparation/recurrence without a device predicate."""
    tokens = qkv.shape[1]
    value_heads = state.shape[1]
    key_heads = (qkv.shape[2] // DIM - value_heads) // 2
    output = qkv.new_empty((1, tokens, value_heads, DIM))
    next_conv, final = torch.empty_like(conv), torch.empty_like(state)
    torch.library.wrap_triton(fused_kernel)[(value_heads, DIM // VALUE_TILE)](
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the C64 chunk path with the measured B3 transform implementation."""
    tokens = qkv.shape[1]
    value_heads = state.shape[1]
    key_heads = (qkv.shape[2] // DIM - value_heads) // 2
    output = qkv.new_empty((1, tokens, value_heads, DIM))
    next_conv, final = torch.empty_like(conv), torch.empty_like(state)
    shape = (1, tokens, value_heads, DIM)
    q, k, v = (torch.empty(shape, device=qkv.device, dtype=torch.float32) for _ in range(3))
    g, beta = (
        torch.empty((1, tokens, value_heads), device=qkv.device, dtype=torch.float32)
        for _ in range(2)
    )
    torch.library.wrap_triton(prepare_kernel)[(tokens, value_heads)](
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
        tokens,
        key_heads,
        value_heads,
        DIM,
        4,
        **OPTIONS,
    )
    launch_chunked(q, k, v, g, beta, state, output, final)
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate inputs and select fused recurrence or chunked FP32 prefill."""
    if qkv.ndim != 3 or qkv.shape[0] != 1:
        raise ValueError("Expected batch-one QKV [1,T,channels]")
    tokens = qkv.shape[1]
    if tokens < 1 or tokens > 512:
        raise ValueError("Token length must be 1..512")
    if b.ndim != 3 or b.shape[0] != 1 or b.shape[1] != tokens or a.shape != b.shape:
        raise ValueError("Expected matching A/B [1,T,value_heads]")
    value_heads = b.shape[2]
    key_heads = (qkv.shape[2] // DIM - value_heads) // 2
    channels = (2 * key_heads + value_heads) * DIM
    if key_heads != 16 or value_heads not in (16, 32) or qkv.shape[2] != channels:
        raise ValueError("Expected Hk=16, Hv=16 or 32 and K=V=128")
    if conv.shape != (1, channels, 4) or weight.shape != (channels, 4):
        raise ValueError("Expected matching width-4 convolution state and weights")
    if (
        state.shape != (1, value_heads, DIM, DIM)
        or A_log.shape != (value_heads,)
        or dt_bias.shape != (value_heads,)
    ):
        raise ValueError("Expected per-value-head recurrent state and decay parameters")
    tensors = (qkv, b, a, conv, weight, A_log, dt_bias)
    if qkv.dtype not in (torch.bfloat16, torch.float32) or any(
        x.dtype != qkv.dtype for x in tensors
    ):
        raise ValueError("Expected matching BF16 or FP32 inputs and parameters")
    if state.dtype != torch.float32:
        raise ValueError("Recurrent state must be FP32")
    if qkv.device.type != "cuda" or any(x.device != qkv.device for x in (*tensors, state)):
        raise ValueError("Expected CUDA tensors on one device")
    qkv, b, a, conv, weight, A_log, dt_bias, state = (x.contiguous() for x in (*tensors, state))
    operands = (qkv, b, a, conv, state, weight, A_log, dt_bias)
    if tokens == 1:
        return short(*operands)
    return torch.cond(tokens <= SHORT_THRESHOLD, _short_branch, _long_branch, operands)
