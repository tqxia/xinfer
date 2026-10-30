"""Fused GDN preparation/recurrence; sequential tokens, parallel value tiles.

Contract: contiguous B=1, H=16, K=V=128; T is a runtime scalar (1..512).
This deliberately keeps the FP32 state recurrence simple. TODO: chunked prefill.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

HEADS, DIM, VALUE_TILE = 16, 128, 32
OPTIONS = dict(num_warps=4, num_stages=1)


@triton.jit
def _recurrent_step(q, k, v, g, beta, state, D: tl.constexpr):
    q = q * tl.rsqrt(tl.sum(q * q, 0) + 1e-6) * (D ** -0.5)
    k = k * tl.rsqrt(tl.sum(k * k, 0) + 1e-6)
    state = state * tl.exp(g)
    prediction = tl.sum(state * k[:, None], 0)
    delta = (v - prediction) * beta
    state = state + k[:, None] * delta[None, :]
    out = tl.sum(state * q[:, None], 0)
    return out, state


@triton.jit
def _causal_conv(QKV, Conv, Weight, NextConv, channels, t, tokens, write_state,
                 N: tl.constexpr, C: tl.constexpr):
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
def fused_kernel(QKV, B, A, Conv, Initial, Weight, ALog, DTBias,
                  Output, NextConv, Final, tokens,
                  H: tl.constexpr, D: tl.constexpr, BV: tl.constexpr, C: tl.constexpr):
    head, tile = tl.program_id(0), tl.program_id(1)
    rows = tl.arange(0, D)
    cols = tile * BV + tl.arange(0, BV)
    offsets = head * D * D + rows[:, None] * D + cols[None, :]
    state = tl.load(Initial + offsets)
    scale = -libdevice.exp(tl.load(ALog + head).to(tl.float32))
    dt = tl.load(DTBias + head).to(tl.float32)
    for t in range(tokens):
        # Q/K are shared by four value tiles; only tile 0 owns their conv state.
        q = _causal_conv(QKV, Conv, Weight, NextConv, head * D + rows,
                         t, tokens, tile == 0, 3 * H * D, C)
        k = _causal_conv(QKV, Conv, Weight, NextConv, (H + head) * D + rows,
                         t, tokens, tile == 0, 3 * H * D, C)
        v = _causal_conv(QKV, Conv, Weight, NextConv, (2 * H + head) * D + cols,
                         t, tokens, True, 3 * H * D, C)
        beta = tl.sigmoid(tl.load(B + t * H + head).to(tl.float32)).to(B.dtype.element_ty).to(tl.float32)
        x = tl.load(A + t * H + head).to(tl.float32) + dt
        softplus = tl.where(x > 20.0, x, libdevice.log1p(libdevice.exp(x)))
        g = scale * softplus
        out, state = _recurrent_step(q, k, v, g, beta, state, D)
        tl.store(Output + (t * H + head) * D + cols, out)
    tl.store(Final + offsets, state)


@torch.library.triton_op("xinfer::delta_fused", mutates_args={})
def fused(qkv: torch.Tensor, b: torch.Tensor, a: torch.Tensor,
           conv: torch.Tensor, state: torch.Tensor, weight: torch.Tensor,
           A_log: torch.Tensor, dt_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse causal convolution/gates and recurrence for prefill and decode."""
    if qkv.ndim != 3 or qkv.shape[0] != 1 or qkv.shape[2] != 3 * HEADS * DIM:
        raise ValueError("Expected QKV [1,T,6144]")
    tokens = qkv.shape[1]
    if tokens < 1 or tokens > 512:
        raise ValueError("Token length must be 1..512")
    if b.shape != (1, tokens, HEADS) or a.shape != b.shape:
        raise ValueError("Expected A/B [1,T,16]")
    if conv.shape != (1, 3 * HEADS * DIM, 4) or weight.shape != (3 * HEADS * DIM, 4):
        raise ValueError("Expected convolution state [1,6144,4] and weights [6144,4]")
    if state.shape != (1, HEADS, DIM, DIM) or A_log.shape != (HEADS,) or dt_bias.shape != (HEADS,):
        raise ValueError("Expected recurrent state [1,16,128,128] and decay parameters [16]")
    tensors = (qkv, b, a, conv, weight, A_log, dt_bias)
    if qkv.dtype not in (torch.bfloat16, torch.float32) or any(x.dtype != qkv.dtype for x in tensors):
        raise ValueError("Expected matching BF16 or FP32 inputs and parameters")
    if state.dtype != torch.float32:
        raise ValueError("Recurrent state must be FP32")
    if qkv.device.type != "cuda" or any(x.device != qkv.device for x in (*tensors, state)):
        raise ValueError("Expected CUDA tensors on one device")
    qkv, b, a, conv, weight, A_log, dt_bias, state = (
        x.contiguous() for x in (*tensors, state))
    output = qkv.new_empty((1, tokens, HEADS, DIM))
    next_conv, final = torch.empty_like(conv), torch.empty_like(state)
    torch.library.wrap_triton(fused_kernel)[(HEADS, DIM // VALUE_TILE)](
        qkv, b, a, conv, state, weight, A_log, dt_bias, output, next_conv, final, tokens,
        HEADS, DIM, VALUE_TILE, 4, **OPTIONS)
    return output, next_conv, final
