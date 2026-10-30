"""Forward-only chunked GDN with FP32 workspaces and IEEE matrix products.

Preparation supplies normalized/scaled Q and normalized K. Chunk-local work
runs in parallel; only the FP32 memory scan is sequential between chunks.
"""

import torch
import triton
import triton.language as tl

CHUNK_SIZE = 64
VALUE_TILE = 32
OPTIONS = dict(num_warps=4, num_stages=1)


@triton.jit
def _solve(K, G, Beta, Inverse, Prefix, tokens,
           H: tl.constexpr, D: tl.constexpr, C: tl.constexpr):
    chunk, head = tl.program_id(0), tl.program_id(1)
    rows = tl.arange(0, C)
    dims = tl.arange(0, D)
    time = chunk * C + rows
    valid = time < tokens
    key = tl.load(K + (time[:, None] * H + head) * D + dims[None, :],
                  mask=valid[:, None], other=0)
    beta = tl.load(Beta + time * H + head, mask=valid, other=0)
    g = tl.load(G + time * H + head, mask=valid, other=0)
    prefix = tl.cumsum(g, 0)
    tl.store(Prefix + (chunk * H + head) * C + rows, prefix)
    # Mask before exp: future-token differences can otherwise overflow.
    lower = rows[:, None] > rows[None, :]
    decay = tl.exp(tl.where(lower, prefix[:, None] - prefix[None, :], -float("inf")))
    inverse = -tl.dot(key * beta[:, None], tl.trans(key), input_precision="ieee") * decay
    # Build (I+L)^-1 by forward substitution, matching the HF chunk formula.
    for i in range(1, C):
        row = tl.sum(tl.where(rows[:, None] == i, inverse, 0), 0)
        row = row + tl.sum(row[:, None] * inverse, 0)
        inverse = tl.where(rows[:, None] == i, row[None, :], inverse)
    inverse = inverse + (rows[:, None] == rows[None, :]).to(tl.float32)
    offsets = (chunk * H + head) * C * C + rows[:, None] * C + rows[None, :]
    tl.store(Inverse + offsets, inverse)


@triton.jit
def _transforms(K, V, Beta, Inverse, Prefix, U, W, tokens,
                H: tl.constexpr, D: tl.constexpr, C: tl.constexpr):
    chunk, head = tl.program_id(0), tl.program_id(1)
    rows = tl.arange(0, C)
    dims = tl.arange(0, D)
    time = chunk * C + rows
    valid = time < tokens
    inverse = tl.load(Inverse + (chunk * H + head) * C * C
                      + rows[:, None] * C + rows[None, :])
    offsets = (time[:, None] * H + head) * D + dims[None, :]
    key = tl.load(K + offsets, mask=valid[:, None], other=0)
    value = tl.load(V + offsets, mask=valid[:, None], other=0)
    beta = tl.load(Beta + time * H + head, mask=valid, other=0)
    prefix = tl.load(Prefix + (chunk * H + head) * C + rows)
    u = tl.dot(inverse, value * beta[:, None], input_precision="ieee")
    weighted_key = (key * beta[:, None]) * tl.exp(prefix)[:, None]
    w = tl.dot(inverse, weighted_key, input_precision="ieee")
    destination = (chunk * H + head) * C * D + rows[:, None] * D + dims[None, :]
    tl.store(U + destination, u)
    tl.store(W + destination, w)


@triton.jit
def _states(K, U, W, Prefix, Initial, Incoming, Corrections, Final, tokens,
            H: tl.constexpr, D: tl.constexpr, C: tl.constexpr, BV: tl.constexpr):
    head, tile = tl.program_id(0), tl.program_id(1)
    dims = tl.arange(0, D)
    values = tile * BV + tl.arange(0, BV)
    rows = tl.arange(0, C)
    state_offsets = head * D * D + dims[:, None] * D + values[None, :]
    state = tl.load(Initial + state_offsets)
    for chunk in range(tl.cdiv(tokens, C)):
        tl.store(Incoming + chunk * H * D * D + state_offsets, state)
        time = chunk * C + rows
        valid = time < tokens
        base = (chunk * H + head) * C * D
        w = tl.load(W + base + rows[:, None] * D + dims[None, :])
        u = tl.load(U + base + rows[:, None] * D + values[None, :])
        correction = u - tl.dot(w, state, input_precision="ieee")
        tl.store(Corrections + base + rows[:, None] * D + values[None, :], correction)
        key = tl.load(K + (time[:, None] * H + head) * D + dims[None, :],
                      mask=valid[:, None], other=0)
        prefix = tl.load(Prefix + (chunk * H + head) * C + rows)
        last = tl.minimum(tokens - chunk * C, C) - 1
        last_prefix = tl.load(Prefix + (chunk * H + head) * C + last)
        weighted_key = key * tl.exp(last_prefix - prefix)[:, None]
        state = state * tl.exp(last_prefix) + tl.dot(
            tl.trans(weighted_key), correction, input_precision="ieee")
    tl.store(Final + state_offsets, state)


@triton.jit
def _outputs(Q, K, Prefix, Incoming, Corrections, Output, tokens,
             H: tl.constexpr, D: tl.constexpr, C: tl.constexpr, BV: tl.constexpr):
    chunk, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = tl.arange(0, C)
    dims = tl.arange(0, D)
    values = tile * BV + tl.arange(0, BV)
    time = chunk * C + rows
    valid = time < tokens
    offsets = (time[:, None] * H + head) * D + dims[None, :]
    query = tl.load(Q + offsets, mask=valid[:, None], other=0)
    key = tl.load(K + offsets, mask=valid[:, None], other=0)
    prefix = tl.load(Prefix + (chunk * H + head) * C + rows)
    causal = rows[:, None] >= rows[None, :]
    decay = tl.exp(tl.where(causal, prefix[:, None] - prefix[None, :], -float("inf")))
    attention = tl.dot(query, tl.trans(key), input_precision="ieee") * decay
    state = tl.load(Incoming + (chunk * H + head) * D * D
                    + dims[:, None] * D + values[None, :])
    correction = tl.load(Corrections + (chunk * H + head) * C * D
                         + rows[:, None] * D + values[None, :])
    output = tl.dot(query * tl.exp(prefix)[:, None], state, input_precision="ieee")
    output += tl.dot(attention, correction, input_precision="ieee")
    destination = (time[:, None] * H + head) * D + values[None, :]
    tl.store(Output + destination, output, mask=valid[:, None])


def launch_chunked(q, k, v, g, beta, state, output, final):
    """Write fresh output/final tensors; inputs are FP32 prepared head vectors."""
    tokens, heads, dim = q.shape[1:]
    chunks = (tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
    inverse = torch.empty((chunks, heads, CHUNK_SIZE, CHUNK_SIZE), device=q.device, dtype=torch.float32)
    prefix = torch.empty((chunks, heads, CHUNK_SIZE), device=q.device, dtype=torch.float32)
    shape = (chunks, heads, CHUNK_SIZE, dim)
    u, w, corrections = (torch.empty(shape, device=q.device, dtype=torch.float32) for _ in range(3))
    incoming = torch.empty((chunks, heads, dim, dim), device=q.device, dtype=torch.float32)
    torch.library.wrap_triton(_solve)[(chunks, heads)](
        k, g, beta, inverse, prefix, tokens, heads, dim, CHUNK_SIZE, **OPTIONS)
    torch.library.wrap_triton(_transforms)[(chunks, heads)](
        k, v, beta, inverse, prefix, u, w, tokens, heads, dim, CHUNK_SIZE, **OPTIONS)
    torch.library.wrap_triton(_states)[(heads, dim // VALUE_TILE)](
        k, u, w, prefix, state, incoming, corrections, final, tokens,
        heads, dim, CHUNK_SIZE, VALUE_TILE, **OPTIONS)
    torch.library.wrap_triton(_outputs)[(chunks, heads, dim // VALUE_TILE)](
        q, k, prefix, incoming, corrections, output, tokens,
        heads, dim, CHUNK_SIZE, VALUE_TILE, **OPTIONS)
