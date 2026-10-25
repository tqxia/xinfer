# Projection packing experiment — GB10, 2026-10-25

Export-time packing works, but packing both projection groups slows native decode
by **3.2–3.5%**. QKV-only results are below. Both tested packed profiles also
introduce top-token differences against Hugging Face, so packing remains an
opt-in experiment. The default is `--projection-packing none`.

## Implementation

The [post-export pass](../../compiler/passes/projections.py) concatenates frozen
BF16 weights once, replaces separate linear calls with one ordinary ATen linear,
and slices its output. It runs after the attention rewrite and before re-export
and AOTInductor compilation. There is no new runtime custom operator.

| Mode | Packed weights per layer | Linear calls per forward, including LM head |
|---|---|---:|
| `none` | Original projections | 197 |
| `qkv` | Q/K/V: `[4096,1024]` | 141 |
| `gate-up` | gate/up: `[6144,1024]` | 169 |
| `all` | Both | 113 |

The [generated native wrappers](generated-wrapper-checks.json) confirm these counts for the two full-model
packed profiles. The source model is unchanged; unused original weights are
removed from the transformed module. All 56 cache writes remain. The pass checks
the expected shapes, dtype, common input, layer set, bias absence, weight consumers
and intervening side effects before editing the graph. Metadata assertions are
preserved but may be crossed because they do not modify tensors or arithmetic state.

Q/K normalization, RoPE, attention, SiLU, multiplication and output/down
projections retain their graph operations. BF16 cast boundaries and
`emulate_precision_casts=True` are retained. Packing does not reduce the number
of weight elements or multiply-adds; it changes matrix shapes, launch counts,
output strides and vendor-kernel selection. Different reduction orders can
change BF16 results.

See the [usage guide](../../docs/gb10-projection-packing.md),
[tests](../../compiler/tests/projection_packing.py), and
[packing report](decode.packing.json). Gate/up-only passes the isolated suite;
it was not independently benchmarked as a full model.

## Full native benchmark

Qwen3-0.6B, batch one, BF16, GB10 SM121, Torch `2.13.0+cu130`, CUDA 13.0,
Triton 3.7.1. Both profiles use the existing FlashAttention custom operator,
unpadded caches and unchanged native runtime. No CUDA Graphs or quantization.

Each paired comparison has four alternating process rounds. Every process runs
an excluded first request, two warmups and five measured requests for each
prompt length. All requests feed the same 64 decode tokens, independent of
predictions and EOS. Each table cell has 20 measured requests. Times are medians;
decode includes normal runtime dispatch, argmax and host token retrieval.
Loading, warmup and profiling overhead are excluded. GPU idle checks passed
before and after each process, without `--allow-busy`.

### Both groups packed

| Prompt tokens | Baseline decode ms/token | Packed decode ms/token | Decode latency change | Baseline → packed prefill ms |
|---:|---:|---:|---:|---:|
| 25 | 7.966 | 8.223 | +3.23% | 7.534 → 7.251 |
| 128 | 8.097 | 8.359 | +3.24% | 8.214 → 8.107 |
| 512 | 8.278 | 8.569 | +3.53% | 11.827 → 12.118 |

Prefill latency changes are −3.76%, −1.30%, and +2.46%. All 65 predictions per
benchmark case agree between these profiles; this does not replace the broader
correctness suite below. Active telemetry samples (GPU utilization at least 60%)
have median SM clocks of 2,502 MHz for baseline and 2,496 MHz for packed. The
observed ranges are 2,405–2,509 and 2,476–2,502 MHz, respectively.

Evidence: [summary](summary.json), [environment and hashes](environment.json),
[workload](workload.json), [process ordering](runs.json). Raw requests, idle checks,
telemetry and a frozen source snapshot are stored alongside this report.

### QKV-only ablation

| Prompt tokens | Baseline decode ms/token | QKV decode ms/token | Decode latency change | Baseline → QKV prefill ms |
|---:|---:|---:|---:|---:|
| 25 | 7.993 | 7.984 | -0.11% | 7.538 → 7.431 |
| 128 | 8.111 | 8.099 | -0.14% | 8.178 → 8.070 |
| 512 | 8.304 | 8.285 | -0.23% | 11.802 → 11.902 |

The decode differences are only 0.1–0.2%; at the shortest prompt, the direction
is not consistent across individual paired rounds. This does not establish a
compelling decode gain. Prefill changes are −1.43%, −1.33%, and +0.85%.
All 65 predictions per benchmark case agree with the paired baseline.
This is a separate four-round comparison with a fresh baseline, not a ratio
against measurements from the first experiment.

Active SM-clock samples for `aoti_flash`: median 2,502 MHz, range 2,444–2,522 MHz.

Active SM-clock samples for `aoti_qkv`: median 2,502 MHz, range 2,405–2,509 MHz.

Evidence: [QKV summary](qkv-only/summary.json),
[QKV environment](qkv-only/environment.json), [QKV workload](qkv-only/workload.json).

## Why fewer linear calls were slower

A native Nsight Systems capture uses the production runtime with three warmups,
one request at prompt lengths 25 and 512, and eight forced decode steps each.
The following GPU times sum all 28 layers and average the eight decode steps
after the 512-token prompt. They explain kernel selection; they are not the
unprofiled benchmark times.

| Projection group | Baseline GPU ms/decode | Both packed GPU ms/decode | Observed kernel change |
|---|---:|---:|---|
| Q + K + V | 1.306 | 1.360 | Three GEMV calls become one larger GEMV |
| gate + up | 1.703 | 1.997 | Two CUTLASS WMMA kernels become one GEMV |
| Output projection | 0.791 | 0.789 | GEMV retained |
| Down projection | 1.186 | 1.181 | GEMV retained |
| LM head | 2.007 | 2.024 | GEMV retained |

The main regression is gate/up: the original `[3072,1024]` shapes select
`cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8`; the packed
`[6144,1024]` shape selects `internal::gemvx::kernel`. The packed operation is
about 0.294 ms slower across all layers, despite halving this group's launches.
QKV remains GEMV and is about 0.054 ms slower in GPU time.

Total decode kernel count falls from 425 to 341 at the longer context, and the
profiled idle/launch gaps shrink. GPU work increases enough to outweigh that
benefit. Fewer launches alone therefore do not establish an optimization.

Prefill selects different kernels and must be considered separately. At T=512,
QKV GPU time falls from 2.183 to 1.890 ms, while gate/up rises from 2.641 to
2.757 ms. Other fused pointwise/reduction work rises from 0.985 to 1.461 ms,
and SiLU/multiply from 0.190 to 0.288 ms. Packed output slices have different
strides across tokens; downstream layout and code generation are part of the
experiment, in addition to the matrix kernels. These trace measurements do not
isolate stride changes as the sole cause of the extra pointwise cost.

Projection attribution follows each package's generated wrapper order and CUDA
launch correlation within NVTX stages. The analysis checks that all linear
launches match the wrapper count and execute on one stream. The saved JSON
includes the weight order, kernel names, per-stage counts and durations.

Evidence: [baseline analysis](native-profile/flash-analysis.json),
[packed analysis](native-profile/packed-analysis.json),
[baseline projections](native-profile/flash-projections.json),
[packed projections](native-profile/packed-projections.json),
[profile inputs](native-profile/workload.json).

## Correctness and numerical limits

The isolated packing suite passes all 15 combinations: three packing modes,
decode T=1, and dynamic prefill T=2/25/128/512. It checks frozen weight content,
source-weight immutability, absence of runtime weight concatenation, compiled
output errors, cache contents and stable storage. Packed AOTI has zero measured
output error against packed eager in these tests. Six unsupported graph patterns are
rejected, including intervening mutation.

Both full-model packed profiles complete 26 cases and 185 logit rows. C++ and
Python AOTI logits are bitwise equal, as are all 56 full-cache tensors. Cache
poisoning, reset, repeated requests and capacity checks pass. The standalone
native smoke suite for `all` also passes the three generation fixtures (43
tokens), T=1/2/512 boundaries, 20 rejection cases and Python-free linkage.

**Neither packed profile passes the full numerical acceptance check.** The
unchanged limits are 5% for relative L2 and peak-normalized error, plus exact
top-token agreement. The existing baseline also has cache tolerance failures.

| Measurement versus HF BF16 | Baseline Flash | QKV + gate/up | QKV only |
|---|---:|---:|---:|
| Maximum logit relative L2 | 3.883% | 3.808% | 5.215% |
| Maximum logit peak-normalized error | 2.599% | 3.097% | 3.947% |
| Maximum full-cache relative L2 | 5.441% | 4.997% | 4.997% |
| Maximum full-cache peak-normalized error | 44.055% | 41.616% | 41.616% |
| Cache tensors outside tolerance | 17/56 | 14/56 | 14/56 |
| Logit cases outside tolerance | 0 | 0 | 2 |
| Top-token agreement | 185/185 | 183/185 | 183/185 |

For `all`, the mismatch is row 1 of `prefix_boundaries` and its poisoned-cache
repeat: HF selects token 12555, packed selects 99692. HF's top two logits tie,
but the packed-selected token is **0.125 below HF's maximum**; it is not merely
choosing a different member of HF's highest-scoring tie. The packed output's
top two logits also tie. This is one boundary scenario and its poison repeat.

For `qkv`, the mismatch is row 1 of `length_2` and its poisoned repeat. The
same token IDs differ, but HF's top-two margin and its score gap to the selected
token are both **0.25**. Those two cases also exceed the logit L2 threshold.

The complete collectors exit nonzero and record `passed: false`. The new
`--collect-top1-errors` option continues collecting diagnostics without relaxing
token agreement; `--collect-logit-errors` behaves likewise for tolerances.
Native/Python exact checks and generation-fixture assertions remain strict.
Lower cache-error counts do not cancel the new token/logit regressions.

Evidence: [unit suite](validation/unit.json),
[baseline full validation](validation/baseline.json),
[both groups](validation/packed.json), [QKV only](validation/qkv.json),
[native smoke](validation/native-smoke.json),
[direct packed/baseline logit comparison](validation/packed-vs-baseline.json),
[direct QKV/baseline logit comparison](validation/qkv-vs-baseline.json).

## Reproduction and retained artifacts

The [guide](../../docs/gb10-projection-packing.md) contains export, validation and
benchmark commands. Use `--projection-packing qkv` and fresh package/output
directories for the ablation. Full-model timings use the same existing baseline
packages; their SHA-256 hashes are recorded in both benchmark environments.
The benchmark source snapshot was based on commit `189adc3` plus the changes in
this experiment; exact source hashes identify the measured implementation.

Large packages and traces remain in ignored GB10 artifact directories:
`artifacts/aoti-flash-packed`, `artifacts/aoti-flash-packed-qkv`,
`artifacts/packing-native-profile`, and the corresponding `three-path-*` and
`benchmark-projection-packing*` directories. Only compact evidence is committed.
