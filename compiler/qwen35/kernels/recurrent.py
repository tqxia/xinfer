"""Small GDN recurrence baseline; sequential tokens, parallel heads/value tiles.

Contract: contiguous B=1, H=16, K=V=128; T is a runtime scalar (1..512).
This deliberately keeps the FP32 state recurrence simple. TODO: chunked prefill.
"""

import torch
import triton
import triton.language as tl

HEADS, DIM, VALUE_TILE = 16, 128, 32
OPTIONS = dict(num_warps=4, num_stages=1)


@triton.jit
def recurrent_kernel(Q, K, V, G, Beta, Initial, Output, Final, tokens,
                     H: tl.constexpr, D: tl.constexpr, BV: tl.constexpr):
    head, tile = tl.program_id(0), tl.program_id(1)
    rows = tl.arange(0, D)
    cols = tile * BV + tl.arange(0, BV)
    offsets = head * D * D + rows[:, None] * D + cols[None, :]
    state = tl.load(Initial + offsets)
    for t in range(tokens):
        base = (t * H + head) * D
        q = tl.load(Q + base + rows).to(tl.float32)
        k = tl.load(K + base + rows).to(tl.float32)
        v = tl.load(V + base + cols).to(tl.float32)
        q = q * tl.rsqrt(tl.sum(q * q, 0) + 1e-6) * (D ** -0.5)
        k = k * tl.rsqrt(tl.sum(k * k, 0) + 1e-6)
        decay = tl.exp(tl.load(G + t * H + head))
        beta = tl.load(Beta + t * H + head).to(tl.float32)
        state = state * decay
        prediction = tl.sum(state * k[:, None], 0)
        delta = (v - prediction) * beta
        state = state + k[:, None] * delta[None, :]
        out = tl.sum(state * q[:, None], 0)
        tl.store(Output + base + cols, out)
    tl.store(Final + offsets, state)


def run(q, k, v, g, beta, state):
    """Direct JIT launch used to cross-check the packaged operator."""
    output, final = torch.empty_like(v), torch.empty_like(state)
    recurrent_kernel[(HEADS, DIM // VALUE_TILE)](
        q, k, v, g, beta, state, output, final, q.shape[1],
        HEADS, DIM, VALUE_TILE, **OPTIONS)
    return output, final


@torch.library.triton_op("xinfer::delta_recurrent", mutates_args={})
def recurrent(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              g: torch.Tensor, beta: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Functional GDN core; AOTInductor packages the kernel with the model."""
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[2:] != (HEADS, DIM):
        raise ValueError("Expected Q/K/V [1,T,16,128]")
    tokens = q.shape[1]
    if tokens < 1 or tokens > 512:
        raise ValueError("Token length must be 1..512")
    if k.shape != q.shape or v.shape != q.shape or g.shape != (1, tokens, HEADS) or beta.shape != g.shape:
        raise ValueError("Q/K/V or gate shape mismatch")
    if state.shape != (1, HEADS, DIM, DIM):
        raise ValueError("State shape mismatch")
    if q.dtype not in (torch.bfloat16, torch.float32) or any(x.dtype != q.dtype for x in (k, v, beta)):
        raise ValueError("Expected matching BF16 or FP32 Q/K/V/beta")
    if g.dtype != torch.float32 or state.dtype != torch.float32:
        raise ValueError("Decay and state must be FP32")
    if q.device.type != "cuda" or any(x.device != q.device for x in (k, v, g, beta, state)):
        raise ValueError("Expected CUDA tensors on one device")
    q, k, v, g, beta, state = (x.contiguous() for x in (q, k, v, g, beta, state))
    output = torch.empty_like(v)
    final = torch.empty_like(state)
    torch.library.wrap_triton(recurrent_kernel)[(HEADS, DIM // VALUE_TILE)](
        q, k, v, g, beta, state, output, final, tokens,
        HEADS, DIM, VALUE_TILE, **OPTIONS)
    return output, final
