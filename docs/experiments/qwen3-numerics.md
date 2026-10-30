# Qwen3 numerical history and current policy

Qwen3 now supports one fused AOTInductor/FlashAttention profile. The historical
eager-numerics profile and efficient compiled backend are retired; their saved
artifacts remain untouched. The original investigation and commands are retained
in [the pre-refactor guide](https://github.com/tqxia/xinfer/blob/799647189a6466aba938c5bf4575e0a9e4c04ae5/compiler/qwen3/README.md).
This record distinguishes arithmetic differences from runtime/state errors.

## Why eager parity was optional

The original efficient/fused AOTI packages matched C++ bitwise on 163 logit rows
and all 56 full-context cache tensors. All 163 top-token predictions matched HF
BF16, and maximum logit relative L2 was 3.9264%. However, 21/56 full-context
cache tensors exceeded the old 5% comparison limits: worst relative L2 6.5913%
and peak-normalized error 17.8385%.

The first difference was BF16 RoPE multiply-add contraction; a later difference
was RMSNorm's FP32 reduction order crossing BF16 rounding midpoints. The retired
profile disabled contraction and retained `aten.mean.dim` as a native operation.
It matched eager BF16 bitwise on the recorded matrix, without a separate native
inference loop. That established eager fidelity rather than improved accuracy:
against the independent FP32 control, original fused AOTI's worst cache errors
were 3.3822% relative L2 / 8.7940% peak-normalized, versus eager BF16's
6.0668% / 21.2242% on that stress case.

Recorded evidence: `artifacts/three-path/summary.json`,
`artifacts/three-path-eager/summary.json`, and the original RoPE/reduction probes.
These are historical efficient-attention results and do not define the Flash
profile's numerical envelope.

## Flash profile evidence — L20, Torch 2.7.0+cu126

The canonical adapter's saved `artifacts/validate-flash-default/summary.json`
covers 26 cases and 185 logit rows. Python AOTI and C++ agree bitwise on logits
and all cache tensors. Handoff, appended chunks, poisoned future slots, storage,
reset/reuse and repeated requests pass. The existing HF comparison reports:

| Check | Recorded result |
| --- | --- |
| HF BF16 top-token agreement | 183/185 rows |
| HF full-cache tensors exceeding 5% | 14/56 |
| Maximum HF logit relative L2 | 3.8492%, within 5% |
| Native CLI | Three fixtures, prompt lengths 1/2/512, 20 rejection checks |

The two token mismatches are row 1 of `prefix_boundaries` and its poisoned repeat:
HF selects 12555, compiled selects 99692. The compiled-selected token is 0.375
below HF's maximum; the HF top-two margin is 0.1875. An independently tuned build
had 20 cache-tensor differences and the same two top-token mismatches. Reusing
the baseline tuning cache removed the additional drift. Compiler/tuning changes
must therefore be validated per artifact; recorded results are not a universal
cross-GPU or model-quality guarantee.

Before this refactor those cache/token comparisons intentionally made the full
report fail. The current policy changes the meaning of acceptance explicitly;
it does not claim that the arithmetic differences were fixed.

## Current acceptance policy

Hard requirements are finite values, exact C++/Python AOTI outputs and caches,
correct cache write regions/masking, stable storage, reset/reuse, legal context
boundaries, rejection behavior and native/Python AOTI generation agreement. HF
logits retain the existing 5% relative-L2 and peak-normalized error bounds under
identical fed tokens. HF cache metrics, top-token differences and independent
greedy-sequence differences are diagnostic evidence, reported separately from
runtime correctness and from HF parity. FP32 controls remain independent.

Accepting the known profile does not disable logit regression or finite checks,
and it does not convert numerical diagnostics into evidence of model quality.
The test report must distinguish overall acceptance from HF equivalence. The
normal build retains BF16 cast emulation and TF32-off settings. Qwen3.5 keeps
its existing, separate numerical policy.
