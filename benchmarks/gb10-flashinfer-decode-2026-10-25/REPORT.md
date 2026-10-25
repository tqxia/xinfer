# FlashInfer decode experiment — GB10, 2026-10-25

The native FlashInfer integration works, but this first implementation is not a
replacement for the current Flash profile. It improves short-context decode by
about 1%, regresses the 512-token case slightly, and introduces a full-model
logit tolerance failure. It remains opt-in as `--attention-backend flashinfer-decode`.

## What was implemented

- Qwen3-0.6B, batch one, BF16, Torch `2.13.0+cu130`, GB10 SM121.
- FlashInfer v0.7.2.post1 **single-request CUDA-core decode**, compiled directly
  into `libxinfer_attention_ops.so`. This is not XQA, TRTLLM, or tensor-core decode.
- The post-export pass replaces 28 decode SDPA sites with the read-only
  `xinfer::flashinfer_decode` op, preserving 56 KV cache updates. Prefill uses
  the existing PyTorch FlashAttention op.
- Existing unpadded full caches and the CPU shape-only extent input are retained.
  Actual cache strides are passed to the kernel, without repacking normal inputs.
  `backed_size_oblivious` remains disabled. No CUDA Graphs are used.
- Build support is optional. Python packages, Torch, and the CUDA installation
  were unchanged; the pinned FlashInfer and CCCL source headers are build inputs.

See the [implementation and reproduction guide](../../docs/gb10-flashinfer-decode.md),
[native adapter](../../runtime/custom_ops/flashinfer_decode.cu),
[registration](../../compiler/custom_ops/flashinfer_decode.py), and
[graph pass](../../compiler/passes/attention.py).

## Full native benchmark

Both profiles use the same standalone C++ benchmark executable and operator
library. Four process rounds alternate the order. Each process runs an excluded
first request, two warmups, then five measured requests per prompt length. Each
request feeds the same 64 decode tokens, regardless of predicted token or EOS.
There are 20 measured requests per table cell, 120 total. Timings include normal
runtime dispatch, argmax, host token retrieval, and synchronization; model loading
and warmup are excluded.

| Prompt tokens | Flash ms/token | FlashInfer ms/token | Latency change | Flash → FlashInfer tokens/s |
|---:|---:|---:|---:|---:|
| 25 | 7.976 | 7.892 | **−1.06%** | 125.37 → 126.72 |
| 128 | 8.098 | 8.005 | **−1.14%** | 123.49 → 124.92 |
| 512 | 8.291 | 8.350 | **+0.71%** | 120.61 → 119.76 |

Values are medians of per-request average decode latency. The direction is
consistent in each of the four paired rounds. These are modest differences for
this workload, not a general FlashInfer speed claim. Prefill medians were
7.516/8.185/11.804 ms for Flash and 7.513/8.166/11.818 ms for the hybrid profile.
All 65 predictions per case (prefill plus decode) agree between the profiles.

GPU idle checks passed before and after every process. Active telemetry samples
(GPU utilization >=60%) had median SM clocks of 2,502 MHz for both profiles;
ranges were 2,496–2,509 MHz for Flash and 2,470–2,509 MHz for FlashInfer. There was
no `--allow-busy` override. Raw requests, GPU telemetry, idle checks, workload,
package/binary hashes, and source hashes are retained alongside this report.

Evidence: [summary](summary.json), [environment](environment.json),
[workload](workload.json), [run ordering](runs.json).

## Why the isolated call benchmark looks better

The isolated Python loop makes FlashInfer look substantially faster even at
longer contexts: for L=512, median wall time is 11.50 µs/call versus 19.64 µs for
Flash. That timing includes Python/native dispatch and launch gaps. It is not
device-only kernel time and does not represent a native full-model execution.

The isolated profiler already shows the opposite GPU result at L=512:
approximately 9.48 µs of FlashInfer kernel work versus 7.77 µs for Flash. The
full-model native Nsight trace confirms that distinction:

| Prompt tokens | Flash attention GPU ms/decode | FlashInfer attention GPU ms/decode | Kernels per decode, each profile |
|---:|---:|---:|---:|
| 25 | 0.157 | 0.091 | 28 |
| 512 | 0.433 | 0.489 | 56 |

These are sums across all 28 layers, averaged over eight decode steps in one
profiled request after three warmups. They explain the direction and approximate
size of the unprofiled result; profiler wall times are not benchmark timings.
At the longer context, FlashInfer adds roughly 56 µs of attention GPU work per
step, close to the 59 µs end-to-end regression in the full benchmark.

The traces show `flashinfer::SingleDecodeWithKVCacheKernel` and, for L>256,
`flashinfer::MergeStatesLargeNumIndexSetsKernel`. The Flash baseline uses
`pytorch_flash::flash_fwd_kernel` for the short case and split-KV plus combine
kernels for the longer case. Both long-context paths launch two attention kernels
per layer. Normal inputs require no layout-copy kernels.

Linear projections and the vocabulary projection account for over 90% of GPU
kernel time in these native traces. Short-context attention savings therefore
have limited room to improve total decode latency. Before extending this backend,
resolve its numerical regression; then a separate experiment could evaluate
FlashInfer's tensor-core/paged implementations or tune its split-KV dispatch.
The current measurements do not establish that those alternatives will be faster.

Evidence: [isolated probe](operator-probe.json),
[native Flash profile](native-profile/flash-analysis.json),
[native FlashInfer profile](native-profile/flashinfer-decode-analysis.json),
[profile workload](native-profile/workload.json).

## Correctness and numerical acceptance

The isolated native operator passes all 11 lengths tested:
1, 2, 25, 26, 128, 129, 256, 257, 512, 513, and 1024. Tests cover an independent
FP32 reference, comparisons with native Flash, opcheck, immutability, NaN poison
beyond the valid prefix, ignored extent contents, five unusual layouts, and a
nondefault CUDA stream. Maximum relative L2 error against FP32 attention was
0.230%. Eleven AOTI cases reproduce direct-op output and complete cache contents
exactly, with stable storage and correctly ordered cache updates.

Full-model validation exercises 26 cases and 185 logit rows, including cache
limits, appended prefill chunks, repeated requests, poisoning, and forced tokens.
All predicted tokens agree with Hugging Face. Native C++ and Python AOTI logits
are bitwise equal for all rows; all 56 full-cache tensors are also bitwise equal
between those two execution paths.

**Full-model numerical acceptance still fails, and is worse than Flash.** The
unchanged tolerance is 5% for both relative L2 and peak-normalized error.

| Measurement versus HF BF16 | Current Flash | FlashInfer decode |
|---|---:|---:|
| Maximum logit relative L2 | 3.883% | **5.645%** |
| Maximum logit peak-normalized error | 2.599% | 3.161% |
| Maximum full-cache relative L2 | 5.441% | 5.441% |
| Maximum full-cache peak-normalized error | 44.055% | 44.055% |
| Full-cache tensors outside tolerance | 17 / 56 | 17 / 56 |
| Logit cases outside tolerance | 0 | 2: one case and its poison repeat |

The new failure is `length_1` and its poisoned-cache repeat. The initial L=1
output is bitwise equal to Flash and HF. The next decode, at L=2, reaches 5.645%
relative L2 error. That occurs before FlashInfer's split-KV path is active, so
split-KV partial-result rounding alone cannot explain it. Small attention
arithmetic differences accumulating through the model are a plausible cause;
the precise numerical source has not been isolated. Layout/mask/mutation tests
passed, but they do not override this acceptance failure.

The original validator stopped at the first logit failure. An optional
`--collect-logit-errors` mode was added to record all tolerance failures while
keeping the threshold and nonzero failure exit. Bitwise, token-agreement and
cache-invariant checks remain strict. The completed run exits 1 with 19 recorded
failures: 17 cache comparisons plus the two logit cases. Independent FP32 cache
controls are included in its report.

Standalone smoke validation passes all three generation fixtures (43 tokens),
prompt boundaries 1/2/512, and 22 invalid-input/metadata rejection checks. It
confirms Python-free native linkage and execution. A separate build with
FlashInfer disabled builds successfully and rejects the hybrid package before
CUDA/model loading. The original Flash unit/compiled suite also passes after the
integration.

Evidence: [operator tests](validation/flashinfer-unit.json),
[full-model results](validation/full-model.json),
[baseline results](validation/baseline-full-model.json),
[per-step comparison](validation/numerical-comparison.json),
[native smoke](validation/native-smoke.json),
[Flash regression](validation/existing-flash-unit.json).

## Build issues and provenance

The first native build failed because FlashInfer's `fastdiv.cuh` uses
`cuda::fast_mod_div`, which the CUDA 13.0 bundled CCCL headers lack. We fetched
FlashInfer's exact CCCL submodule and put its headers before toolkit headers.
No upstream patch, Python package upgrade, or system CUDA change was required.
A transient GitHub TLS failure during that fetch succeeded on retry.

The other integration constraint is cache layout: FlashInfer's convenience
wrapper assumes compact HND storage, but our fixed-capacity prefix has a larger
head stride. The native parameter struct accepts explicit strides, which avoids
an otherwise incorrect interpretation or a repeated cache copy. See the guide
for the scratch bound, stream handling, and supported layout fallbacks.

- FlashInfer: `beabf5004585b03da60e927bbb803acbaa62fe36` (v0.7.2.post1).
- CCCL: `16bd510c9b712e82b0ab6cbb630d8e29ba1f7116`.
- CUDA compiler: 13.0.88. Python package freeze is unchanged.
- Benchmark source was the experiment working tree based on `8cb4929`; its
  exact performance-relevant sources and hashes are retained under `source/`
  and in `environment.json`. The commit containing this report records the final
  implementation and validation tools.
- Large `.pt2`, Nsight `.nsys-rep`/SQLite, and Chrome trace files remain in
  ignored `artifacts/` on GB10. This report retains compact raw benchmark data,
  numerical summaries, complete native profile analyses, and source snapshots.

Native packages are at `artifacts/aoti-flashinfer-decode`; full numerical results
at `artifacts/three-path-flashinfer-decode-complete`; traces at
`artifacts/flashinfer-native-profile` and `artifacts/flashinfer-decode-probe`.
The [dependency record](dependencies.json) and
[initial compiler diagnostic](validation/initial-build.txt) preserve the build evidence.
