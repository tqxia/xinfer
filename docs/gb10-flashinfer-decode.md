# FlashInfer decode experiment on GB10

`--attention-backend flashinfer-decode` selects an experimental hybrid profile:
PyTorch FlashAttention for prefill and FlashInfer for single-token decode. The
existing `flash` profile remains available unchanged. Both use the post-export
graph pass and read-only native operators. This experiment does not use CUDA
Graphs, paged caches, XQA, TRTLLM, or FlashInfer's tensor-core decode path.

## Current validation status

The initial GB10 run passes the native operator and graph mutation tests. All
185 full-model predicted tokens match Hugging Face, and C++/Python AOTI logits
and caches are bitwise equal. It does **not** pass the full numerical acceptance
criteria: relative logit L2 error reaches 5.65% for a one-token prompt (also in
its poisoned-cache repeat), above the unchanged 5% limit. The existing 17
full-cache tensor failures remain. Keep this profile experimental; see the
[recorded results](../benchmarks/gb10-flashinfer-decode-2026-10-25/REPORT.md).

## Implementation

The model still exports ordinary SDPA. For decode, the
[attention pass](../compiler/passes/attention.py) recognizes the same absolute
causal mask, GQA expansion and cache updates, and replaces 28 SDPA sites with
`xinfer::flashinfer_decode`. It preserves all 56 cache updates. It rejects
multi-token queries and unfamiliar attention patterns. Re-export rebuilds the
graph metadata before AOTInductor compilation.

The [Python registration](../compiler/custom_ops/flashinfer_decode.py) supplies
the fake shape contract. The [native adapter](../runtime/custom_ops/flashinfer_decode.cu)
calls FlashInfer's `SingleDecodeWithKVCacheDispatched` directly, specialized for
BF16, D=128, Hq=16, Hkv=8, batch one, with no internal positional encoding or
sliding window. Qwen has already applied RoPE. One query attends to every valid
key, which matches the lower-right causal semantics of decode.

The cache remains `[1,8,1024,128]`, with no padding. A CPU byte tensor of shape
`[L+1]` provides L through metadata alone; its contents are ignored. A prefix of
length L retains a head stride of `1024*128`, not `L*128`. FlashInfer's convenience
wrapper assumes compact HND strides, so calling it on this prefix would be
incorrect. The underlying native parameter struct accepts explicit head and
token strides. We pass those actual strides and avoid a prefix copy.

The kernel vector-loads 16-byte groups. The adapter copies only unsupported
layouts (including unaligned storage offsets, non-unit feature strides, and
different K/V strides). The normal graph's Q/K/V need no repacking. All inputs
are read-only; the output is freshly allocated with the fake kernel's strides.
The operator uses the caller's current CUDA stream and guards Q's device.

Upstream dispatch uses a single kernel for L <= 256. Above that it may split
the KV reduction into chunks of at least 256 tokens, then merge partial results.
We allocate per-call scratch through ATen, with a bound of
`ceil(L/256) * 16 * (128*2 + 4)` bytes: at most 16,640 bytes for this model. This
includes partial BF16 outputs and FP32 log-sum-exp values. There is no shared
scratch state or hidden request state. Occupancy queries, kernel dispatch and
allocation overhead are included in measurements.

## Build and reproduce

Run from the repository root on GB10, using the existing Torch 2.13 / CUDA 13.0
environment:

```bash
bash tools/setup/fetch_flashinfer.sh
bash scripts/build-gb10.sh -DQWEN_ENABLE_FLASHINFER=ON
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_COMPILE_THREADS=2

.venv/bin/python compiler/tests/flashinfer_decode.py --output artifacts/flashinfer-decode-unit
.venv/bin/python compiler/export.py decode --attention-backend flashinfer-decode --output artifacts/aoti-flashinfer-decode
.venv/bin/python compiler/export.py prefill --attention-backend flashinfer-decode --output artifacts/aoti-flashinfer-decode
AOTI_RUNTIME_CHECK_INPUTS=1 .venv/bin/python compiler/validate_cpp.py --packages artifacts/aoti-flashinfer-decode --output artifacts/three-path-flashinfer-decode --collect-logit-errors

.venv/bin/python compiler/benchmark.py compare \
  --baseline-packages artifacts/aoti-flash --baseline-label aoti_flash \
  --candidate-packages artifacts/aoti-flashinfer-decode --candidate-label aoti_flashinfer \
  --output artifacts/benchmark-flashinfer-decode-20261025
.venv/bin/python experiments/flashinfer_decode_probe.py --output artifacts/flashinfer-decode-probe
```

Choose fresh output directories for reruns. The unit suite checks direct FP32
and Flash comparisons, opcheck, poison beyond the valid prefix, input immutability,
unusual layouts, a nondefault CUDA stream, and AOTI cache mutation ordering at
11 lengths. The full-model validator keeps the existing 5% numerical limits;
operator-level correctness does not imply full-model numerical acceptance.
`--collect-logit-errors` records tolerance failures across the suite and still
exits nonzero; bitwise checks, cache invariants and token agreement remain strict.
The native paired benchmark alternates four process rounds with two warmups and
five measured requests per length, using 64 forced decode tokens. The isolated
probe records CUDA kernels and call timings; its Python loops include launch
overhead, so the C++ full-model benchmark determines the deployment outcome.

## Dependency and package contracts

FlashInfer is optional and disabled in a fresh CMake configuration. The build is
specialized to SM121. No FlashInfer Python package or prebuilt cubin package is
installed, and the native runtime does not link Python.

- FlashInfer v0.7.2.post1:
  `beabf5004585b03da60e927bbb803acbaa62fe36`.
- Its CCCL submodule:
  `16bd510c9b712e82b0ab6cbb630d8e29ba1f7116`.
- The initial build with CUDA 13.0's bundled CCCL failed because it lacks
  `cuda::fast_mod_div`. Fetching FlashInfer's pinned CCCL and putting its headers
  first fixes that without modifying upstream code or the CUDA installation.

CMake checks both revisions and tracked header changes. The checkout is stored
under ignored `work/deps/flashinfer`; only its required CCCL submodule is fetched.
Apache-2.0 license files are copied into `build/` as `FlashInfer-LICENSE` and
`CCCL-LICENSE`; retain them when distributing the operator library.

Both package metadata files carry the hybrid profile and pinned FlashInfer source
contract. Prefill must declare `xinfer::flash_attention`, decode must declare
`xinfer::flashinfer_decode`. Python and C++ reject a wrong source revision or
operator pairing. A runtime built without FlashInfer rejects the hybrid profile
before loading its model packages. CMake caches options; switch it off explicitly
with `bash scripts/build-gb10.sh -DQWEN_ENABLE_FLASHINFER=OFF` when desired.

Upstream references:
[single-decode implementation](https://github.com/flashinfer-ai/flashinfer/blob/beabf5004585b03da60e927bbb803acbaa62fe36/include/flashinfer/attention/decode.cuh),
[parameter struct](https://github.com/flashinfer-ai/flashinfer/blob/beabf5004585b03da60e927bbb803acbaa62fe36/include/flashinfer/attention/default_decode_params.cuh),
[convenience wrapper](https://github.com/flashinfer-ai/flashinfer/blob/beabf5004585b03da60e927bbb803acbaa62fe36/csrc/single_decode.cu).
