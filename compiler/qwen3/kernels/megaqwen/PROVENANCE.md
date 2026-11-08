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

The current implementation targets Qwen3-0.6B only. It is not bound to an exact
Hugging Face repository or revision. The selected checkpoint's identity records
the weights used for an export; preparation retains its default download pin for
reproducibility.

The prior repaired snapshot fixed the checkpoint RoPE base, coherent loads of
mutable hidden/KV data, BF16 tensor boundaries, grid residency checks, CUDA error
checking, and explicit L2 prefetching. This adaptation additionally:

- removes Python, PyBind, runtime compilation, CPU `.item()`, internal position,
  reset, sampling and argmax from the device implementation;
- accepts 311 read-only canonical weight tensors, 56 mutable caller-owned KV
  tensors, read-only BF16 RoPE tables and 12 mutable workspace tensors as explicit
  operator inputs, without allocating or copying model weights or KV storage;
- validates tensor counts, dimensions, dtype, contiguous layout, device,
  alignment, inference-only use, non-overlapping mutable storage and the
  shape-only extent's position/capacity bounds;
- constructs a per-layer pointer table from the current input tensors on every
  call and passes it by value as CUDA grid-constant launch arguments; no pointer
  addresses or mutable workspace are serialized as exported constants;
- reads the token ID from its CUDA tensor and derives position from the CPU
  extent's shape, with no GPU-to-host scalar read. Calling runtimes validate
  token values and retain position, reset and request-admission policy;
- emits independent BF16 `[1,151936]` logits for each call, computing the
  vocabulary projection exactly once after the cooperative transformer body;
- preserves the repaired source's BF16 boundaries and FP32 accumulators, with
  build flags `--fmad=false` and no `--use_fast_math`;
- derives the cooperative launch grid from device SM count and compiled occupancy;
- records the CUDA allocations used by each call with the caching allocator on
  the current stream, deduplicating storage bases within that call to protect
  asynchronous allocation lifetime while preserving caller stream ordering;
- uses the Qwen3-0.6B RoPE contract `theta=1000000`, package-selected capacity in
  512..40960, 28 layers, 16 query heads, 8 KV heads and head dimension 128. The
  exporter constructs the BF16 RoPE tables and shares them as read-only constants;
- extends attention with up to four contiguous KV partitions per query head,
  FP32 partial softmax statistics/values, and a max-rescaled cross-block merge,
  retaining the single-partition path for short contexts. The initial partition
  policy is untuned; this adaptation does not assert a performance improvement.

The shared `xinfer::megaqwen_decode` operator and its AOTI C shim call the same
implementation. The shim borrows tensor handles for the input lists, preserving
weight, KV and workspace storage, then transfers a fresh output handle to AOTI.
Tensor, alias and launch checks do not establish bitwise or numerical parity
with AOTI or Hugging Face; the numerical validation harness evaluates those
contracts separately.

The compiler builds a position-independent static kernel core for the operator
library. `attention.cuh` holds the shared device helpers and attention arithmetic
included by decode and the standalone attention test. The test launch wrappers
and declarations live in `runtime/qwen3/tests/attention.cu` and `attention.h`;
the direct dispatcher-call helper lives in `runtime/qwen3/tests/megaqwen_op.cpp`.
That driver loads the prototype AOTI package before calling its registered op.
Compiled packages carry the shared operator library alongside their wrappers;
ordinary runtimes execute the packaged AOTI
graph without linking or preloading that library themselves. The explicit-input
prototype remains a verification artifact. The production profile retains the
two-entry package interface and uses MegaQwen for every single-token call,
including position zero, while multi-token prefill remains AOTI Flash. The
caller owns cache reset, request positions, prefill/decode dispatch and sampling.
