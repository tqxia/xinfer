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
def _invert_16(neg_l):
    # Forward-substitute a 16x16 strictly lower-triangular block (-L) into (I + L)^-1.
    rows = tl.arange(0, 16)
    inv = neg_l
    for i in range(1, 16):
        row = tl.sum(tl.where(rows[:, None] == i, inv, 0), 0)
        row = row + tl.sum(row[:, None] * inv, 0)
        inv = tl.where(rows[:, None] == i, row[None, :], inv)
    return inv + (rows[:, None] == rows[None, :]).to(tl.float32)


@triton.jit
def _solve(
    K, G, Beta, Inverse, Prefix, Lengths, tokens, H: tl.constexpr, D: tl.constexpr, C: tl.constexpr
):
    tl.static_assert(C == 64, "Block inversion expects C == 64")
    chunk, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    length = tl.minimum(tl.maximum(tl.load(Lengths + batch), 0), tokens)
    chunks = tl.cdiv(tokens, C)
    K += batch * tokens * H * D
    G += batch * tokens * H
    Beta += batch * tokens * H
    Inverse += batch * chunks * H * C * C
    Prefix += batch * chunks * H * C
    r = tl.arange(0, 16)
    dims = tl.arange(0, D)

    t0 = chunk * C + r
    t1 = chunk * C + 16 + r
    t2 = chunk * C + 32 + r
    t3 = chunk * C + 48 + r
    v0, v1, v2, v3 = t0 < length, t1 < length, t2 < length, t3 < length

    # Chain four 16-token cumsums so prefix sums and block inverses stay on 16-wide tiles.
    g0 = tl.load(G + t0 * H + head, mask=v0, other=0)
    g1 = tl.load(G + t1 * H + head, mask=v1, other=0)
    g2 = tl.load(G + t2 * H + head, mask=v2, other=0)
    g3 = tl.load(G + t3 * H + head, mask=v3, other=0)
    p0 = tl.cumsum(g0, 0)
    s0 = tl.sum(g0, 0)
    p1 = tl.cumsum(g1, 0) + s0
    s1 = s0 + tl.sum(g1, 0)
    p2 = tl.cumsum(g2, 0) + s1
    s2 = s1 + tl.sum(g2, 0)
    p3 = tl.cumsum(g3, 0) + s2
    p_base = (chunk * H + head) * C
    tl.store(Prefix + p_base + r, p0)
    tl.store(Prefix + p_base + 16 + r, p1)
    tl.store(Prefix + p_base + 32 + r, p2)
    tl.store(Prefix + p_base + 48 + r, p3)

    k0 = tl.load(K + (t0[:, None] * H + head) * D + dims[None, :], mask=v0[:, None], other=0)
    k1 = tl.load(K + (t1[:, None] * H + head) * D + dims[None, :], mask=v1[:, None], other=0)
    k2 = tl.load(K + (t2[:, None] * H + head) * D + dims[None, :], mask=v2[:, None], other=0)
    k3 = tl.load(K + (t3[:, None] * H + head) * D + dims[None, :], mask=v3[:, None], other=0)
    b0 = tl.load(Beta + t0 * H + head, mask=v0, other=0)
    b1 = tl.load(Beta + t1 * H + head, mask=v1, other=0)
    b2 = tl.load(Beta + t2 * H + head, mask=v2, other=0)
    b3 = tl.load(Beta + t3 * H + head, mask=v3, other=0)
    kb0, kb1, kb2, kb3 = k0 * b0[:, None], k1 * b1[:, None], k2 * b2[:, None], k3 * b3[:, None]

    # Hierarchical 16x16 block Schur inversion of M = I + L (4x4 grid of 16x16 blocks).
    # Mask strictly-lower diagonals before exp so future-token differences cannot overflow.
    # Step 1: invert the four 16x16 diagonal blocks A_kk = M_kk^-1.
    lower = r[:, None] > r[None, :]
    a00 = _invert_16(
        -tl.dot(kb0, tl.trans(k0), input_precision="ieee")
        * tl.exp(tl.where(lower, p0[:, None] - p0[None, :], -float("inf")))
    )
    a11 = _invert_16(
        -tl.dot(kb1, tl.trans(k1), input_precision="ieee")
        * tl.exp(tl.where(lower, p1[:, None] - p1[None, :], -float("inf")))
    )
    a22 = _invert_16(
        -tl.dot(kb2, tl.trans(k2), input_precision="ieee")
        * tl.exp(tl.where(lower, p2[:, None] - p2[None, :], -float("inf")))
    )
    a33 = _invert_16(
        -tl.dot(kb3, tl.trans(k3), input_precision="ieee")
        * tl.exp(tl.where(lower, p3[:, None] - p3[None, :], -float("inf")))
    )

    # Strictly-lower off-diagonal 16x16 blocks of L (every row index > every col index).
    l10 = tl.dot(kb1, tl.trans(k0), input_precision="ieee") * tl.exp(p1[:, None] - p0[None, :])
    l20 = tl.dot(kb2, tl.trans(k0), input_precision="ieee") * tl.exp(p2[:, None] - p0[None, :])
    l21 = tl.dot(kb2, tl.trans(k1), input_precision="ieee") * tl.exp(p2[:, None] - p1[None, :])
    l30 = tl.dot(kb3, tl.trans(k0), input_precision="ieee") * tl.exp(p3[:, None] - p0[None, :])
    l31 = tl.dot(kb3, tl.trans(k1), input_precision="ieee") * tl.exp(p3[:, None] - p1[None, :])
    l32 = tl.dot(kb3, tl.trans(k2), input_precision="ieee") * tl.exp(p3[:, None] - p2[None, :])

    # Step 2: combine diagonal pairs into 32x32 inverses: A_10 = -A_11 L_10 A_00 (and A_32).
    a10 = -tl.dot(a11, tl.dot(l10, a00, input_precision="ieee"), input_precision="ieee")
    a32 = -tl.dot(a33, tl.dot(l32, a22, input_precision="ieee"), input_precision="ieee")
    # Step 3: lower-left 32x32 inverse = -M_B^-1 (L_BA M_A^-1), with T = L_BA M_A^-1.
    t20 = tl.dot(l20, a00, input_precision="ieee") + tl.dot(l21, a10, input_precision="ieee")
    t21 = tl.dot(l21, a11, input_precision="ieee")
    t30 = tl.dot(l30, a00, input_precision="ieee") + tl.dot(l31, a10, input_precision="ieee")
    t31 = tl.dot(l31, a11, input_precision="ieee")
    a20 = -tl.dot(a22, t20, input_precision="ieee")
    a21 = -tl.dot(a22, t21, input_precision="ieee")
    a30 = -(tl.dot(a32, t20, input_precision="ieee") + tl.dot(a33, t30, input_precision="ieee"))
    a31 = -(tl.dot(a32, t21, input_precision="ieee") + tl.dot(a33, t31, input_precision="ieee"))

    # Store the 4x4 block lower-triangular inverse into [C, C] (upper blocks are zero).
    inv_base = (chunk * H + head) * C * C
    zeros = tl.zeros((16, 16), dtype=tl.float32)
    r0, r1, r2, r3 = r[:, None], (16 + r)[:, None], (32 + r)[:, None], (48 + r)[:, None]
    c0, c1, c2, c3 = r[None, :], (16 + r)[None, :], (32 + r)[None, :], (48 + r)[None, :]
    tl.store(Inverse + inv_base + r0 * C + c0, a00)
    tl.store(Inverse + inv_base + r0 * C + c1, zeros)
    tl.store(Inverse + inv_base + r0 * C + c2, zeros)
    tl.store(Inverse + inv_base + r0 * C + c3, zeros)
    tl.store(Inverse + inv_base + r1 * C + c0, a10)
    tl.store(Inverse + inv_base + r1 * C + c1, a11)
    tl.store(Inverse + inv_base + r1 * C + c2, zeros)
    tl.store(Inverse + inv_base + r1 * C + c3, zeros)
    tl.store(Inverse + inv_base + r2 * C + c0, a20)
    tl.store(Inverse + inv_base + r2 * C + c1, a21)
    tl.store(Inverse + inv_base + r2 * C + c2, a22)
    tl.store(Inverse + inv_base + r2 * C + c3, zeros)
    tl.store(Inverse + inv_base + r3 * C + c0, a30)
    tl.store(Inverse + inv_base + r3 * C + c1, a31)
    tl.store(Inverse + inv_base + r3 * C + c2, a32)
    tl.store(Inverse + inv_base + r3 * C + c3, a33)


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
    length = tl.minimum(tl.maximum(tl.load(Lengths + batch), 0), tokens)
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
    length = tl.minimum(tl.maximum(tl.load(Lengths + batch), 0), tokens)
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
    length = tl.minimum(tl.maximum(tl.load(Lengths + batch), 0), tokens)
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
