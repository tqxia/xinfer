"""Write real tokens from padded rows into caller-owned KV buffers."""

import torch
import triton
import triton.language as tl


@triton.jit
def _append(
    Key,
    Value,
    Positions,
    CuQuery,
    CacheKey,
    CacheValue,
    tokens,
    dim: tl.constexpr,
    capacity: tl.constexpr,
    heads: tl.constexpr,
    kb,
    kh,
    kt,
    kd,
    vb,
    vh,
    vt,
    vd,
    pb,
    pt,
    BLOCK: tl.constexpr,
):
    head, row = tl.program_id(1), tl.program_id(2)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token, feature = offsets // dim, offsets % dim
    count = tl.load(CuQuery + row + 1) - tl.load(CuQuery + row)
    valid = (token < tokens) & (token < count)
    position = tl.load(Positions + row * pb + token * pt, valid, other=0)
    key = tl.load(Key + row * kb + head * kh + token * kt + feature * kd, valid, other=0)
    value = tl.load(Value + row * vb + head * vh + token * vt + feature * vd, valid, other=0)
    target = ((row * heads + head) * capacity + position) * dim + feature
    tl.store(CacheKey + target, key, valid)
    tl.store(CacheValue + target, value, valid)


@torch.library.triton_op("xinfer::append_kv_batched", mutates_args={"cache_key", "cache_value"})
def append_kv_batched(
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cu_query: torch.Tensor,
    cache_key: torch.Tensor,
    cache_value: torch.Tensor,
) -> None:
    """Positions are validated by the runner; cache tensors are contiguous [B,H,C,D]."""
    torch._check(cache_key.is_contiguous() and cache_value.is_contiguous())
    batch, heads, tokens, dim = key.shape
    torch.library.wrap_triton(_append)[(triton.cdiv(tokens * dim, 256), heads, batch)](
        key,
        value,
        positions,
        cu_query,
        cache_key,
        cache_value,
        tokens,
        dim,
        cache_key.shape[2],
        heads,
        *key.stride(),
        *value.stride(),
        *positions.stride(),
        256,
    )
