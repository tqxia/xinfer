# MegaQwen native decode provenance

The arithmetic/device implementation in `kernels.cu` derives from
[`Infatoshi/MegaQwen`](https://github.com/Infatoshi/MegaQwen), upstream commit
`0bc1fdedb82744481373000c5c7d3c58795c7a3a`, specifically
`csrc/megakernel/fused_decode_ldg.cu`. The input was the repaired local snapshot
used by the earlier standalone experiments. Its SHA-256 is
`21d42a0423538b591dfe206c33127b1b3502214196dd6683714190206bb786f9`.
The upstream dimensional header `csrc/megakernel/config.cuh` in that snapshot has
SHA-256 `c32f1cb2ce0afc8571c3875225d4319fe91df167531101b1f84c7235a4b09748`.
`config.cuh` here retains only the constants needed by the native implementation.

The upstream README at that revision declares its license as **MIT**. The supplied
snapshot has no standalone `LICENSE`, `LICENSE.md`, `LICENSE.txt`, or `COPYING`
file, and the imported kernel contains no copyright notice. This directory
records the observed declaration without inventing a copyright holder or notice.
The upstream declaration is preserved verbatim in `UPSTREAM_LICENSE_NOTICE.txt`.

The prior repaired snapshot fixed the checkpoint RoPE base, coherent loads of
mutable hidden/KV data, BF16 tensor boundaries, grid residency checks, CUDA error
checking, and explicit L2 prefetching. This native adaptation additionally:

- removes Python, PyBind, runtime compilation, CPU `.item()`, internal position,
  reset, sampling and argmax from the decode implementation;
- borrows the verified `SharedWeights` tensors by exact manifest source FQN;
- borrows all 56 existing KV tensors through a small per-layer pointer table,
  without allocating or copying model weights or KV storage;
- emits independent BF16 `[1,151936]` logits for each call, computing the
  vocabulary projection exactly once after the cooperative transformer body;
- preserves the repaired source's BF16 boundaries and FP32 accumulators, with
  build flags `--fmad=false` and no `--use_fast_math`;
- derives the cooperative grid once from device SM count and compiled occupancy;
- validates dimensions, dtype, contiguous layout, device, alignment, token and
  position bounds, and exposes addresses and workspace size for alias auditing;
- records each borrowed or owned CUDA allocation with the caching allocator on
  first use of each stream, deduplicating storage bases to protect asynchronous
  allocation lifetime without blocking or changing caller-owned stream ordering;
- constructs BF16 RoPE tables with the Qwen3-0.6B contract `theta=1000000`, fixed
  capacity 1024, 28 layers, 16 query heads, 8 KV heads and head dimension 128.

The native interface does not assert bitwise or numerical parity with AOTI or
Hugging Face. `info().validated` covers tensor/alias/launch contracts; the full
numerical validation harness must evaluate the backend separately. The caller
owns position, cache reset, prefill/decode selection and sampling.
