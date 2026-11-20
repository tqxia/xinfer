"""Forward-only chunked GDN with FP32 workspaces and IEEE matrix products.

Preparation supplies normalized/scaled Q and normalized K. Chunk-local work
runs in parallel; only the FP32 memory scan is sequential between chunks.
"""

import torch
import triton
import triton.language as tl

CHUNK_SIZE = 64
TRANSFORM_TILE = 32
VALUE_TILE = 32
OPTIONS = dict(num_warps=4, num_stages=1)


@triton.jit
def _solve(
    K, G, Beta, Inverse, Prefix, Lengths, tokens, H: tl.constexpr, D: tl.constexpr, C: tl.constexpr
):
    chunk, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    length = tl.load(Lengths + batch)
    chunks = tl.cdiv(tokens, C)
    K += batch * tokens * H * D
    G += batch * tokens * H
    Beta += batch * tokens * H
    Inverse += batch * chunks * H * C * C
    Prefix += batch * chunks * H * C
    rows = tl.arange(0, C)
    dims = tl.arange(0, D)
    time = chunk * C + rows
    valid = time < length
    key = tl.load(K + (time[:, None] * H + head) * D + dims[None, :], mask=valid[:, None], other=0)
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
def _transforms(
    K,
    V,
    Beta,
    Inverse,
    Prefix,
    U,
    W,
    Lengths,
    tokens,
    H: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    BT: tl.constexpr,
):
    tl.static_assert(BT >= 16 and (BT & (BT - 1)) == 0, "Unsupported transform tile")
    tl.static_assert(D % BT == 0, "Transform tile must divide the head dimension")
    chunk, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    length = tl.load(Lengths + batch)
    chunks = tl.cdiv(tokens, C)
    K += batch * tokens * H * D
    V += batch * tokens * H * D
    Beta += batch * tokens * H
    Inverse += batch * chunks * H * C * C
    Prefix += batch * chunks * H * C
    U += batch * chunks * H * C * D
    W += batch * chunks * H * C * D
    rows = tl.arange(0, C)
    dims = tl.program_id(2) * BT + tl.arange(0, BT)
    time = chunk * C + rows
    valid = time < length
    inverse = tl.load(Inverse + (chunk * H + head) * C * C + rows[:, None] * C + rows[None, :])
    offsets = (time[:, None] * H + head) * D + dims[None, :]
    value = tl.load(V + offsets, mask=valid[:, None], other=0)
    beta = tl.load(Beta + time * H + head, mask=valid, other=0)
    destination = (chunk * H + head) * C * D + rows[:, None] * D + dims[None, :]
    u = tl.dot(inverse, value * beta[:, None], input_precision="ieee")
    tl.store(U + destination, u)
    key = tl.load(K + offsets, mask=valid[:, None], other=0)
    prefix = tl.load(Prefix + (chunk * H + head) * C + rows)
    weighted_key = (key * beta[:, None]) * tl.exp(prefix)[:, None]
    w = tl.dot(inverse, weighted_key, input_precision="ieee")
    tl.store(W + destination, w)


@triton.jit
def _states(
    K,
    U,
    W,
    Prefix,
    Initial,
    Incoming,
    Corrections,
    Final,
    Lengths,
    tokens,
    H: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    BV: tl.constexpr,
):
    batch_head, tile = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    length = tl.load(Lengths + batch)
    chunks = tl.cdiv(tokens, C)
    K += batch * tokens * H * D
    U += batch * chunks * H * C * D
    W += batch * chunks * H * C * D
    Prefix += batch * chunks * H * C
    Initial += batch * H * D * D
    Incoming += batch * chunks * H * D * D
    Corrections += batch * chunks * H * C * D
    Final += batch * H * D * D
    dims = tl.arange(0, D)
    values = tile * BV + tl.arange(0, BV)
    rows = tl.arange(0, C)
    state_offsets = head * D * D + dims[:, None] * D + values[None, :]
    state = tl.load(Initial + state_offsets)
    # Padding must not apply decay or change the carried state.
    for chunk in range(tl.cdiv(length, C)):
        tl.store(Incoming + chunk * H * D * D + state_offsets, state)
        time = chunk * C + rows
        valid = time < length
        base = (chunk * H + head) * C * D
        w = tl.load(W + base + rows[:, None] * D + dims[None, :])
        u = tl.load(U + base + rows[:, None] * D + values[None, :])
        correction = u - tl.dot(w, state, input_precision="ieee")
        tl.store(Corrections + base + rows[:, None] * D + values[None, :], correction)
        key = tl.load(
            K + (time[:, None] * H + head) * D + dims[None, :], mask=valid[:, None], other=0
        )
        prefix = tl.load(Prefix + (chunk * H + head) * C + rows)
        last = tl.minimum(length - chunk * C, C) - 1
        last_prefix = tl.load(Prefix + (chunk * H + head) * C + last)
        weighted_key = key * tl.exp(last_prefix - prefix)[:, None]
        state = state * tl.exp(last_prefix) + tl.dot(
            tl.trans(weighted_key), correction, input_precision="ieee"
        )
    tl.store(Final + state_offsets, state)


@triton.jit
def _outputs(
    Q,
    K,
    Prefix,
    Incoming,
    Corrections,
    Output,
    Lengths,
    tokens,
    H: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    BV: tl.constexpr,
):
    chunk, batch_head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    batch, head = batch_head // H, batch_head % H
    length = tl.load(Lengths + batch)
    chunks = tl.cdiv(tokens, C)
    Q += batch * tokens * H * D
    K += batch * tokens * H * D
    Prefix += batch * chunks * H * C
    Incoming += batch * chunks * H * D * D
    Corrections += batch * chunks * H * C * D
    Output += batch * tokens * H * D
    rows = tl.arange(0, C)
    dims = tl.arange(0, D)
    values = tile * BV + tl.arange(0, BV)
    time = chunk * C + rows
    valid = time < length
    offsets = (time[:, None] * H + head) * D + dims[None, :]
    query = tl.load(Q + offsets, mask=valid[:, None], other=0)
    key = tl.load(K + offsets, mask=valid[:, None], other=0)
    prefix = tl.load(Prefix + (chunk * H + head) * C + rows)
    causal = rows[:, None] >= rows[None, :]
    decay = tl.exp(tl.where(causal, prefix[:, None] - prefix[None, :], -float("inf")))
    attention = tl.dot(query, tl.trans(key), input_precision="ieee") * decay
    state = tl.load(
        Incoming + (chunk * H + head) * D * D + dims[:, None] * D + values[None, :],
        mask=chunk * C < length,
        other=0,
    )
    correction = tl.load(
        Corrections + (chunk * H + head) * C * D + rows[:, None] * D + values[None, :],
        mask=chunk * C < length,
        other=0,
    )
    output = tl.dot(query * tl.exp(prefix)[:, None], state, input_precision="ieee")
    output += tl.dot(attention, correction, input_precision="ieee")
    destination = (time[:, None] * H + head) * D + values[None, :]
    tl.store(
        Output + destination, tl.where(valid[:, None], output, 0), mask=(time < tokens)[:, None]
    )


def launch_chunked(q, k, v, g, beta, state, output, final, query_lengths):
    """Write fresh output/final tensors; inputs are FP32 prepared head vectors."""
    batch, tokens, heads, dim = q.shape
    if (
        type(TRANSFORM_TILE) is not int
        or TRANSFORM_TILE < 16
        or TRANSFORM_TILE & (TRANSFORM_TILE - 1)
        or dim % TRANSFORM_TILE
    ):
        raise ValueError(
            "TRANSFORM_TILE must be a power of two >= 16 that divides the head dimension"
        )
    chunks = (tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
    # Kernels use flat pointer offsets, so workspace shape metadata is unused.
    # Flatten without padding: a sample with one chunk must not specialize the
    # dynamic chunk dimension to 1 during AOTInductor's tensor-layout checks.
    inverse = torch.empty(
        (batch * chunks * heads * CHUNK_SIZE * CHUNK_SIZE,), device=q.device, dtype=torch.float32
    )
    prefix = torch.empty(
        (batch * chunks * heads * CHUNK_SIZE,), device=q.device, dtype=torch.float32
    )
    shape = (batch * chunks * heads * CHUNK_SIZE * dim,)
    u, w, corrections = (torch.empty(shape, device=q.device, dtype=torch.float32) for _ in range(3))
    incoming = torch.empty(
        (batch * chunks * heads * dim * dim,), device=q.device, dtype=torch.float32
    )
    torch.library.wrap_triton(_solve)[(chunks, batch * heads)](
        k, g, beta, inverse, prefix, query_lengths, tokens, heads, dim, CHUNK_SIZE, **OPTIONS
    )
    torch.library.wrap_triton(_transforms)[(chunks, batch * heads, dim // TRANSFORM_TILE)](
        k,
        v,
        beta,
        inverse,
        prefix,
        u,
        w,
        query_lengths,
        tokens,
        heads,
        dim,
        CHUNK_SIZE,
        TRANSFORM_TILE,
        **OPTIONS,
    )
    torch.library.wrap_triton(_states)[(batch * heads, dim // VALUE_TILE)](
        k,
        u,
        w,
        prefix,
        state,
        incoming,
        corrections,
        final,
        query_lengths,
        tokens,
        heads,
        dim,
        CHUNK_SIZE,
        VALUE_TILE,
        **OPTIONS,
    )
    torch.library.wrap_triton(_outputs)[(chunks, batch * heads, dim // VALUE_TILE)](
        q,
        k,
        prefix,
        incoming,
        corrections,
        output,
        query_lengths,
        tokens,
        heads,
        dim,
        CHUNK_SIZE,
        VALUE_TILE,
        **OPTIONS,
    )
