# Qwen3.6 FP8 numerical issue — 2026-11-08

The current DeltaNet and attention adapters execute the full text model, but
both independently fail its unchanged HF FP8 numerical gate. FP8 projection and
expert replacement alone is bitwise equal to HF. A matched-input investigation
locates small reduction differences and demonstrates amplification through FP8
activation quantization and MoE routing. No numerical fix is claimed here.

Fixing this issue is deferred while export, compilation
and native execution of the **current kernels** are explored. Those functional
checks must report their own results and keep HF numerical acceptance open.
Successful compilation or Python/C++ parity will not resolve this issue.

## References and independent controls

The checkpoint is `Qwen/Qwen3.6-35B-A3B-FP8` at revision
`c653bba46dfa06e9658dcc0b0dc87fe4685f14a2`. Tests use Torch 2.13 on GB10,
Transformers 5.20.2, the unchanged official HF FP8 forwards/default `grouped_mm`
expert implementation and the pinned finegrained-FP8 kernel revision
`03ae31933313f38fae2b050dbe2731024386c369`. The supported HF client is
`kernels==0.17.2`. No separate full FP32 or BF16 model is loaded.

The recorded full eager matrix contains 31 teacher calls, 20 state checkpoints
and two short greedy cases, before the later nonzero-prefill export fixture was
added. FP8-only replacement passes all **1,918 comparisons bitwise**.
The combined adapter path executes but retains **1,369 required failures**;
28 of 34 logit comparisons exceed the 5% relative-L2/peak limits. Teacher top-1
and short greedy tokens still agree. Token agreement does not waive state or
logit failures.

Independent ablations use identical tokens, fresh caches and one retained
checkpoint bank. They cover prefill lengths 2, 96, 98 and 511, each followed by
two one-token decodes:

| Replaced computation | Worst final-logit relative L2 | Worst peak-relative |
|---|---:|---:|
| FP8 projections/experts only | 0, bitwise | 0 |
| Explicit loop/cache, original HF computation | 0, bitwise | 0 |
| DeltaNet only, original HF attention | 43.5812% | 63.1944% |
| Attention only, original HF DeltaNet | 32.1008% | 38.6968% |
| Both adapters | 40.0387% | 54.3750% |

The errors interact nonlinearly and must not be added. The explicit-HF control
preserves HF's convolution update semantics while exposing functional state.
It rules out the manual loop/cache interface as the observed source in these
cases. No weight-loading, packing or raw cache-update discrepancy was found.

## DeltaNet: first observed operation and amplification

Indices are zero-based. For fresh T=2, the earliest captured BF16 core difference
is layer 0, token 0/head 12/channel 125: HF `0.0001468658447`, adapter
`0.0001459121704`, one BF16 ULP (`9.536743e-7`). This is an input to the same
original HF gated RMSNorm, not evidence that the norm implementation changed.
Every decoder output on this call still rounds to identical BF16 values.

Matched real-input replays reproduce the original captured HF cores bitwise and
the adapted core on its adapted inputs. Convolution/SiLU, V, BF16 beta, decay
input and raw convolution history match. A staged preparation trace finds
`x*x` bitwise equal and **FP32 `sum(x*x)` as the first differing normalization
primitive**. An example Q sum is HF `4.092260837554932` versus Triton
`4.09226131439209`, one FP32 ULP. The traced normalized Q/K match the actual
production preparation kernel, while the Torch trace matches HF preparation.
This does not establish that this reduction explains every fused result.

There is also a core operation-order difference: HF uses chunk-64 and
`torch.linalg.solve_triangular` for fresh prefill, including T=2. The adapter
uses sequential recurrence through T=97 and an explicit inverse/chunk transform
above 97. HF's own recurrence differs from its chunk function by the observed
one-cell BF16 result on layer-0 T=2. Forcing the long path fixes that example,
but does not remove the T=96/T=98 differences. FMA/accurate-exp controls likewise
do not establish a general fix.

A second-decode example at layer 16 has identical hidden input, projections,
convolution history, parameters and norm gate. Incoming FP32 state differs by
at most `5.960464e-8`. One BF16 norm-output cell changes from
`-0.005096435547` to `-0.005065917969`. The pinned dynamic FP8 quantizer maps its
normalized activation from `-84.25225067` to `-83.74774933`, crossing midpoint
`-84`; the E4M3 code changes from `-88` to `-80`. All activation scales match.
The effective dequantized input jump is **15.857 times** the BF16 input change.
Actual GPU captures show 560/2,048 output-projection values change, followed by
1,185/2,048 decoder-output values. The quantizer reconstruction is a CPU
calculation; those output counts come from actual GPU execution.

State swaps show that either accumulated state rounding or current-call
arithmetic can trigger this boundary. Later routing also changes: T=96 layer 1
replaces expert 204 with 42; in the T=2 sequence, the first route-order swap is
layer 18 and the first expert-set change is layer 29 (101 replaced by 60).

## Attention: an independent reduction difference

Matched attention replay runs without DeltaNet on identical hidden input,
RoPE embeddings, positions and incoming KV. Projection outputs, Q/K norm,
sigmoid gate, post-RoPE Q and updated KV match bitwise. T=2 and both decodes are
entirely bitwise. For T=96/T=98, the first difference is the attention reduction:

- HF selects `aten::_scaled_dot_product_efficient_attention`.
- The adapter selects `aten::_scaled_dot_product_flash_attention` through
  `xinfer::flash_attention`.

These backends were observed without changing flags or inputs. Eight replay
checks reproduce the actual full-model attention outputs bitwise. For T=96,
HF-vs-Flash core error reaches 0.22181% per-row L2 / 0.66667% peak; after gated
FP8 output projection, block L2 reaches 0.86403%. Both cores remain close to a
local FP32 QK/softmax/V calculation on the same BF16 Q/K/V. That operator control
does not replace the primary HF reference or establish full-model acceptance.

## Recommended fixes, deferred

First build an HF-compatible correctness baseline, then reintroduce optimized
components independently:

1. **Attention:** preserve HF's actual mask, layout, scaling and SDPA backend in
   a small ATen-backed operator. Confirm matched real-input and full-model
   agreement before returning to the Flash replacement.
2. **DeltaNet:** preserve the HF preparation/reduction order, fresh versus cached
   dispatch, chunk-64 triangular solve and BF16 materialization boundaries. An
   opaque C++ operator calling the corresponding ATen operations is a useful
   exportable baseline: runtime tensor sizes control its host loop, avoiding
   export's Python-loop specialization without changing its internal math.
3. Register fake/shape implementations for export and native C++ implementations
   for deployment. Validate the baseline in standalone C++ before optimizing
   normalization, scans or fusion one component at a time.

These operators are recommendations, **not implemented fixes**. An ATen/C++
baseline may need a native registration library and will prioritize agreement
over speed. Matching one local cell or changing only FMA/exp is insufficient;
each candidate must pass the existing full-model state/logit/token checks.
Do not replace the HF reference backend or relax its 5% limits to obtain a pass.

## Additional compiled/eager discrepancy — 2026-11-08

The complete 40-layer model exported and AOTI-compiled with the current kernels
on GB10 at capacity 1,024, with dynamic prefill over 2–512 tokens, one-token
decode and combo kernels disabled. Python and native validation used fresh
processes and saved CPU references, avoiding simultaneous full model banks.
This experiment introduces a separate failed gate:

| Comparison | Recorded result |
|---|---|
| Adapted eager versus HF FP8, regenerated matrix | 1,526 required failures retained |
| Python AOTI versus adapted eager | 1,339 required failures / 1,957 comparisons; 501 bitwise |
| C++ AOTI versus Python AOTI | 1,991 / 1,991 bitwise; zero failures |

The matrix contains 34 teacher calls, 22 selected complete state checkpoints and
two short greedy cases. It adds nonzero-offset prefill `[96, 98, 1]` to the
earlier eager matrix. Compiler/eager failures comprise 514 conv, 495 recurrent,
147 key, 154 value and 29 logit comparisons. Worst logit error is **36.9162%
relative L2 / 48.1579% peak-relative** at the 98-token initial prefill; maximum
absolute logit error is 11.4375. The existing 5% limits and required token
checks are unchanged. All teacher argmax and greedy IDs agree.

All 163 recorded T=2 comparisons are bitwise. The T=1 sequence
has 162/163 bitwise comparisons: its second decode differs in
`layer_2.recurrent` by at most `1.946464e-7`, with relative L2 `4.813912e-6`
and peak-relative `4.857081e-6`, within the existing limits. Small passing
differences must be distinguished from the large longer-prefill failures.

These results establish an additional AOTI-versus-current-eager discrepancy;
they do **not** identify its first differing operation or root cause. The
earlier HF-versus-adapted investigation concerns eager computation and does
not explain this new compiler gap by itself. A follow-up should replay matched
eager/compiled layer inputs and locate the first differing operation before
changing kernels or compiler settings. No tolerance, reference or failure
policy was changed to obtain native parity.

Standalone C++ matched Python AOTI raw logits, selected state and greedy
outputs bitwise, with one loader pair, 80 state tensors and no Python library
loaded. Teacher resets, KV ownership/write regions and functional-input
preservation passed; greedy KV storage remained stable. The native report
keeps `compiler_eager_parity_passed: false`, the failed HF reference and
`model_accepted: false`. Native execution and exact runtime parity are proven
for this bounded profile; HF numerical acceptance and compiler/eager parity
remain open.

The Python AOTI validation allocator peaks were 33.609 GiB allocated and
33.623 GiB reserved. These figures include validation workspaces/state checks,
not total process/system unified memory or compilation peaks. The longest
teacher request consumed 515 positions; neither a complete 1,024-slot window
nor the declared 262,144-slot default was exercised. No performance result is
claimed.

The later Torch constant-handle lifetime correction required recompiling both
wrappers. Rebuilt and original bundles match in all 1,991 recorded comparisons,
including greedy final states, while the rebuilt compiler/eager comparison
retains the same 1,339 failures and 501/1,957 bitwise comparisons. The installed,
moved C++ runtime also matches rebuilt Python AOTI in all 1,991 comparisons.
Native cleanup now returns live ATen allocation to its warmed baseline; that
ownership correction does not resolve either numerical issue. The 1,526 HF
failures and `model_accepted: false` remain preserved. See the
[model guide](../qwen36-fp8.md#native-memory-and-deployment-checks) for the
bounded memory/ownership results. The installed/renamed production CLI also
passed all three ID/raw-logit parity cases and 18 CPU pre-load rejections on a
renamed bundle, with no Python library loaded and no controller CUDA
initialization. Those deployment checks do not resolve the HF/compiler
numerical gates or establish timing/full-context results.

## Retained evidence

Retained evidence is outside Git under these relative paths:

- `artifacts/qwen36/step2/`: isolated eager references, protocols and summaries.
- `artifacts/qwen36/step2/full40-adapted-export-r1/summary.json`: regenerated
  34-call adapted/HF reference, retaining 1,526 failures.
- `artifacts/qwen36/step3/python-aoti-r1/summary.json`: required compiler/eager
  failures, execution coverage and Python allocator samples.
- `artifacts/qwen36/step3/native-aoti-r1/summary.json`: exact native/Python parity
  and retained upstream numerical failures.
- `artifacts/qwen36/step4/cli-fixed-r1/summary.json`: exact production CLI parity,
  CPU rejection coverage and retained numerical failures after relocation.
- `artifacts/qwen36/numerics/ablation-{prefix4,full40}-r1/`: independent controls.
- `artifacts/qwen36/numerics/capture-full40-r1/`: canonical module/state captures.
- `work/qwen36/numerics/gdn-matched-r1-results/` and
  `norm-matched-r1-results/`: matched core and normalization controls.
- `work/qwen36/numerics/attention-layer3-matched-r2.json` and
  `attention-fidelity-r1.json`: backend observations and replay fidelity.
- `work/qwen36/numerics/state-swap-r1.json` and
  `layer16-outproj-quantization.json`: state/boundary controls and amplification.

The [model guide](../qwen36-fp8.md) describes the current loading and validation
workflow. Large captures and checkpoint tensors remain generated artifacts.
