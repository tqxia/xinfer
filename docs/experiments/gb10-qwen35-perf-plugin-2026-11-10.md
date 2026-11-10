# Qwen3.5-0.8B optimization with xinfer-perf on GB10 — 2026-11-10

The experiment found two different outcomes: global reduction tuning lowered native decode latency by
about 21% but introduced required numerical/token regressions; an isolated vocabulary-head prototype
lowered latency by about 1.75–2.21% while preserving the healthy baseline's checked state bytes and required
failure identities. The model's existing HF numerical failures remain unresolved. Production defaults are
unchanged.

This report covers the November 10 experiment, with expanded technique explanations and skill feedback
added on November 11. It separates optimization results, measurement controls, and recommendations for
the skill itself.

## 1. Optimization techniques and results

### Scope, setup and baseline

The [optimize skill](../../plugins/xinfer-perf/skills/optimize/SKILL.md) was applied on `codex/xinfer-perf`,
starting from `11dd903b2914a665f5c37adfd28dbbc4e70e1845`. Its sequence was baseline measurement → native
profile → recorded hypothesis → compiler experiment → full validation → paired timing →
prediction/observation feedback. The default budget was three candidates or two hours; calibration and
noise-control runs supported those three candidates. Runtime changes and CUDA Graphs were outside scope.

The checkpoint was `Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17`. Retained
official source weights were transferred and prepared offline using the existing preparation tools;
checkpoint verification and the environment check passed. A matching existing interpreter was reused.
The stack was Torch `2.13.0+cu130`, Triton `3.7.1`, Transformers `5.20.2`, CUDA 13.0, GB10 SM 12.1 and
Nsight Systems `2025.3.2.474`. The focused preparation/model/export/validation/benchmark CPU suite passed
**89 tests**; native state/generation contract checks also passed.

The compiled profile was batch one, BF16, capacity 1,024, dynamic prefill over 2–512 tokens per call and
one-token decode. Projection packing was already enabled in the baseline: 48 packed groups reduced 187
linear calls to 97. This experiment did not introduce that packing optimization.

The original baseline passed **19,216/19,216 bitwise C++/Python AOTI comparisons**, finite-value and
structural state/ownership checks, but retained **16,886 required failures**. A later fresh-cache default re-export,
`baseline-healthy`, retained **16,890 failures**: 5,848 eager/HF, 5,174 AOTI/eager and 5,868 direct
AOTI/HF failures. They predominantly concern conv/recurrent state, with one required tied top-token
mismatch. Both baselines remain unaccepted; their identities must remain explicit in comparisons.

### Hardware recovery before interpreting performance

The initial benchmark reproduced **611 MHz under load** in 96/97 busy samples. Those timings are retained
but excluded from healthy-performance conclusions. After the user's shutdown and PSU power disconnect,
54/54 busy samples were 2,398–2,528 MHz, with zero 611 MHz readings. Subsequent paired measurements used
normal clocks. The default export was also repeated after recovery to remove the possibility that
compile-time autotuning had selected schedules under the fault.

### Why the experiments targeted projections

Direct native captures used 8 and 32 forced decode steps. The native executable runs first-use, warmup and
measured requests. The captures had 14 explicit synchronization calls and, per request, one scalar
device-to-host prediction transfer for prefill plus one for every decode step. These observations established
decode boundaries; whole-process kernel totals were not treated as decode totals.

| Measured 32-step decode region | Launches/token | GPU kernel ms/token | Kernel-time share |
|---|---:|---:|---:|
| Decoder-layer projections, cuBLAS BF16 GEMV | 96 | 5.668516 | 69.89% |
| Vocabulary head, CUTLASS GEMM | 1 | 2.052474 | 25.31% |
| All kernels | 276 | 8.110966 | 100% |

Together, projections account for **95.19% of summed GPU-kernel duration**. These are device-time
fractions, not wall-time fractions. Generated decode contained 97 external `aoti_torch_cuda_mm_out`
calls. This supported projection implementation experiments; FlashAttention and DeltaNet core kernels
offered much smaller direct ceilings in this workload. No Nsight Compute counters were collected, so this profile does not by
itself prove a particular bandwidth or occupancy bottleneck.

### Attempt 1: global coordinate-descent reduction tuning

**Mechanism.** In the pinned Torch build, `coordinate_descent_tuning=True` does more than search launch
parameters such as warp counts and tile sizes. It also enables singleton matrix-multiplication decomposition
into multiplication and summation. For a projection, the conceptual computation is:

```text
x: [1, K], W: [N, K]
y[0, j] = sum_k(x[0, k] * W[j, k])
```

Inductor generates fused Triton reductions for this expression, reading BF16 weights and performing
opmath accumulation in FP32. This changes the implementation selected for the matrix operation, rather
than merely retuning the existing cuBLAS call. Different reduction layouts/order and other tuned kernels
can also change floating-point results.

**Prediction and artifact evidence.** Replacing the expensive projection calls was expected to lower decode
latency. The compiled artifact confirmed the change: external matrix calls went **97→0 in decode** and
**97→96 in prefill**. The dynamic multi-token layer projections stayed external in prefill, while its
singleton vocabulary head became a reduction.

**Measured result.** Four paired rounds showed **20.82–20.92% lower median decode latency** across
22/128/512-token prompts. At prompt length 128, the measured pair was **8.619→6.820 ms/token**.

**Correctness and decision.** Required failures increased **16,886→17,609**, comprising 761 new and 38
resolved failures. A new `capital_teacher` top-token mismatch had an HF score gap of 0.125, so it was not
a tie; capital greedy generation added a period. Native parity remained **19,217/19,217**, and the 5% HF
logit bounds still passed. This demonstrates that exact native/Python parity and bounded logits do not
guarantee the required token/state behavior.

The candidate was rejected and its experimental global CLI flag removed. The performance mechanism was
applied successfully, but the first operation causing the numerical regression was not localized. The flag
changes both matrix lowering and other kernel tuning, so the regression should not be attributed to one
specific operation without further investigation.

### Attempt 2: lower only the vocabulary head

**Mechanism.** The head projects one final hidden row into vocabulary scores:

```text
[1, 1024] × [1024, 248320] → [1, 248320]
```

This remains a singleton matrix product in both prefill and decode because only the last hidden row is
projected. A scoped compilation hook intercepted Torch's registered matrix decomposition and selected
CUDA BF16 products with `M=1` and a large output width. It called the original registered opmath wrapper
under the temporary decomposition-enabling flag, preserving its promotion/cast contract. Global tuning
remained off; this control did not coordinate-tune the subsequently generated head kernel.

The target was attractive because its profile cost was about 2.05 ms/token and it comes after decoder state
updates. During teacher forcing, head logits do not select the next input token. Free generation can still
change future state when argmax changes, so teacher-logit, greedy-token and final-state checks remain
necessary.

**Measured result.** Both entries retained all **96 decoder-layer external matrix calls** and replaced only
the head call. Median decode latency improved **1.46–1.70%** against the healthy baseline; at length 128,
it changed **8.684→8.557 ms/token**. No new required teacher-token or greedy failure appeared.

**Compilation confound.** Fresh default autotuning also selected different schedules for unrelated
normalization kernels. For example, unchanged gated-normalization arithmetic received `XBLOCK=1` or
`XBLOCK=8` configurations. The default re-export alone changed the failure total from 16,886 to 16,890.
Only **60/381 teacher-state payloads** matched between the head-only artifact and the healthy baseline.
The candidate had **16,891 failures** versus 16,890: nine new and eight resolved, with finite outputs,
native parity and state contracts intact.

The measured small gain was retained as evidence, but this was a confounded experiment for attributing
state or performance changes solely to the head. A focused source edit had not guaranteed a focused binary
change. The next attempt therefore controlled unchanged compilation schedules.

### Attempt 3: tune the head while freezing unchanged kernels

**Mechanism and controls.** The head-only decomposition was combined with a second scoped hook that
enabled coordinate-descent metadata only for the observed `x=248320,r=1024` head reduction. The hooks
check the pinned Torch version, revision and relevant source hashes, restore their process-global changes
after compilation, and assume sequential compilation. Installed Torch files remain unchanged.

A fork of the healthy baseline's cache retained unchanged per-kernel autotuning winners. Whole-FX-graph
cache reuse was disabled so the changed graph was regenerated. Global coordinate descent stayed off.
The head's selected configurations changed from one warp in the control to **eight warps for prefill and
four for decode**, with `XBLOCK=1` and one stage.

**Artifact isolation.** All 96 decoder-layer matrix operands were unchanged. All **29 common prefill and
21 common decode kernels** matched in arithmetic body, launch configuration and exact packaged CUBIN
SHA256. Only the output-head/final-normalization tail changed.

**Why the gain is modest.** Separate profiles measured the new head reduction at **1.898172 ms/token**,
versus **2.052474 ms/token** for the former CUTLASS head, about a 7.5% reduction. Since the head was only
about one-quarter of GPU-kernel time, an overall improvement near 2% is consistent with that observation.
Total kernel launches remained **276 per decode token**. The paired unprofiled measurements below
establish the actual inference effect; profiler durations explain the mechanism.

**Correctness.** All **381 teacher-state checkpoints and three greedy final-state files** matched the
healthy baseline by SHA256: **12,419,334,144 bytes per variant**. C++/Python AOTI passed
**19,216/19,216 bitwise comparisons**, with finite/ownership/masking contracts intact. Required failures
remained **16,890**, with **zero new and zero resolved identities**. This preserves the baseline's outcomes
without establishing full HF numerical acceptance.

The output logits were not bitwise identical to baseline. Across 381 teacher rows, maximum direct
baseline/candidate error was **0.0117302% relative L2, 0.423729% peak-relative and 0.125 absolute**, with
all top-1 IDs unchanged. Direct HF worst relative L2 changed only **3.0067809%→3.0067837%**;
peak-relative stayed **1.461314%**. Native CLI smoke returned the same four IDs
`[11751,248046,198,248044]` and decoded capital answer for both bundles.

### Controlled timing, noise and memory

Each initial comparison used the same native executable, compatible logical weights/contracts, one CPU
thread, 64 forced decode steps, two warmups and four alternating-order process rounds with five samples:
**20 samples per profile/length**, 120 measured requests and 7,680 decode tokens for the three lengths.
The focused 22-token repeat used seven samples per round: **28 per profile**, 56 requests and 3,584 decode
tokens. Decode includes argmax and host reads; prefill covers the complete prompt.

| Prompt/run | Healthy baseline median | Tuned head median | Decode reduction | Prefill median reduction |
|---|---:|---:|---:|---:|
| 22, initial | 8.645725 ms/token | 8.475276 ms/token | 1.97% | 2.74% |
| 22, focused repeat | 8.677198 ms/token | 8.485203 ms/token | 2.21% | 3.03% |
| 128 | 8.698767 ms/token | 8.531948 ms/token | 1.92% | 1.87% |
| 512 | 8.745302 ms/token | 8.592551 ms/token | 1.75% | 2.44% |

All candidate round medians improved. Initial A/A aggregate median label drift was **0.146%, 0.192% and
0.145%** at 22/128/512. At 128/512, every candidate round's gain exceeded the largest absolute A/A round
drift. However, the original **22-token A/A round-4 drift was 4.021%**, exceeding the candidate effect;
its cause remains unknown. All original data were retained.

The focused repeat showed candidate round reductions of **1.99–2.49%**. Its identical-artifact A/A median
drift was **0.0353%**, with maximum absolute round drift **0.342%**. The anomaly did not recur. These
controls support the modest gain under the tested conditions; they do not define a universal noise bound
or formal confidence interval.

All 16 before/after idle checks passed with `allow_busy=false` for each paired and A/A run. Repeat
telemetry at 200 ms recorded **1,672 samples**, including 445 busy samples: zero 611 MHz, all busy P0
with clock-event flags zero, maximum 46°C. Process snapshots observed only `xinfer_benchmark`, at most
one compute process; sampled CPU/memory/I/O pressure was low. Sampling can miss transients, so this is
no observed competing compute program, not continuous proof of exclusivity. Earlier telemetry sampled
approximately once per second.

At 512, both variants peaked at **1,661,877,760 ATen allocated bytes** and **1,734,344,704 reserved
bytes**; shorter-case allocation differences were at most **204,288 bytes**. GB10 device-used memory
includes shared system memory and must not be added to allocator peaks or called model-only memory.

## 2. Feedback on the optimize skill

### What worked

The skill's strongest contribution was disciplined investigation. It supplies a workflow and technique
guidance; the agent still selects candidates, examines compiler behavior and analyzes traces using existing
project tools. This run demonstrated the following useful requirements already present in the skill:

- **Profile before choosing candidates.** Native profiling directed work toward the 95.19% projection
  hotspot. FlashAttention core kernels were below 1% of GPU-kernel time in this decode profile.
- **Record a hypothesis and inspect the artifact.** Generated-code checks showed that the tuning flag
  changed matrix decomposition, and later exposed unrelated normalization scheduling changes.
- **Separate correctness from performance.** Existing validators and actual error/failure comparisons
  prevented the roughly 21% gain from being treated as an acceptable optimization.
- **Use paired native measurements and investigate noise.** Alternating order, repeated rounds and A/A
  controls exposed the short-prompt anomaly. Profiler timing alone would not have established the gain.
- **Retain failed-attempt feedback and bound scope.** The rejected global change guided a narrower head
  experiment. The three-candidate/two-hour budget kept the investigation focused.

### Recommended improvements

The following strengthen existing guidance; baseline profiling, artifact inspection, A/A when needed and
comparison of actual numerical errors were already required. The proposals are not new automatic checks
implemented by this report.

| Priority | Observed gap | Concrete addition |
|---|---|---|
| 1. Hardware health before export and timing | Idle checks did not detect the 611 MHz fault; compile-time autotuning can also be affected. | Require a short representative load probe with clocks, utilization, power and clock-event telemetry before autotuning and timing. After recovery, reassess artifacts tuned under the fault. |
| 2. Compilation isolation | A focused edit still changed unrelated normalization configurations and state bytes. | Describe how to compare unaffected bodies, launch settings, CUBINs and external-call operands. Where appropriate, fork baseline per-kernel winners while forcing regeneration of the changed graph; verify the result rather than assuming cache reuse succeeded. |
| 3. Relative acceptance for failing baselines | Preserving known failures does not fully specify the keep/reject decision. | Declare protected behavior and a relative non-regression gate before editing: strict finite/state/native checks, no new required failure identities or required token/greedy regressions, and explicit metric deltas. Keep full-model acceptance separate. |
| 4. Measurement triage | Pooled A/A medians hid one much noisier round. | Require per-workload/per-round inspection, retain unexplained anomalies, and repeat affected cases. Avoid treating pooled A/A drift as a universal noise limit. |
| 5. Readiness and budget allocation | Missing artifacts, preparation and hardware recovery consume the same investigation budget. | Resolve interpreter, checkpoint, native build, baseline artifacts and disk space early; estimate missing work and reserve time for validation and paired measurements. Record setup time without silently restarting the deadline. |

Relative non-regression is an exploratory comparison against a particular baseline. It must not waive full
HF acceptance, relax tolerances, replace references or suppress failures. The retained candidate's unchanged
16,890 failures illustrate that distinction.

A small reusable **offline Nsight phase-analysis helper** would reduce manual work. It could consume the
existing workload and exported trace, verify synchronization/prediction-transfer counts, and report decode
windows with attribution warnings. A generic artifact-comparison helper could similarly compare kernel
bodies/configurations and binaries. These should reuse existing xinfer export, validation, benchmark and
provenance records; model-specific Torch probes belong with their experiments.

The demonstrated decomposition, compilation-isolation and numerical-checking lessons were added to the
[common technique notes](../../plugins/xinfer-perf/skills/optimize/references/techniques/common.md), with
an experiment link in the plugin README. The other items above remain proposed improvements. Technique
notes should continue to grow from measured outcomes, including failures.

### Coverage of this assessment

The experiment exercised the skill instructions through the existing coding agent, SSH and project tools.
It did not exercise Codex/Claude plugin installation, marketplace discovery, cached loading or native skill
invocation. The evidence supports feedback on the investigation workflow for this model and stack;
host-plugin integration needs separate validation.

## Reproduction, evidence and limitations

Only documentation is retained as tracked changes. Production exporter/runtime defaults are unchanged.
The retained prototype uses guarded private Torch decomposition/codegen hooks specific to the pinned stack;
it is not a supported new CLI. Its bundle is `artifacts/perf/20261110-qwen35-gb10/head-tuned`.
Experiment drivers and the rejected global patch are archived in
`artifacts/perf/20261110-qwen35-gb10/experiment-source.tar.gz` (7,415 bytes).

Let `$REPO` be the checkout, `$PY` the matching interpreter, `$GPU_UUID` the selected device,
`$WORK=$REPO/work/perf/20261110-qwen35-gb10` and `$OUT=$REPO/artifacts/perf/20261110-qwen35-gb10`.
Restore archived scripts into `$WORK`, run from `$REPO`, and choose fresh cache/output destinations:

```bash
cp -a "$WORK/baseline-healthy-inductor-cache" "$WORK/repro-cache"
CUDA_VISIBLE_DEVICES="$GPU_UUID" TORCHINDUCTOR_CACHE_DIR="$WORK/repro-cache" TORCHINDUCTOR_FX_GRAPH_CACHE=0 \
  "$PY" "$WORK/export_head_tuned.py" --model-dir "$REPO/work/qwen35/0.8b/text" \
  --max-context-length 1024 --output "$OUT/repro-head-tuned"
```

Validate through `tools.validation.validate`, then measure with `tools.benchmark compare`. Exact executed
commands, environment overrides and statuses are in the corresponding `*-plan.json` and `*-status.json`;
use fresh destinations. Start/budget and prediction/observation notes are retained in `$WORK/notes.md`.

Evidence under `$WORK`:

- Baseline/profile: `baseline-validation-analysis.json`, `baseline-clock-comparison.json`,
  `profile-decode-analysis-r1.json`, `profile-decode-crosschecks-r1.json`.
- First two attempts: `candidate1-validation-comparison.json`, `candidate1-performance-analysis.json`,
  `candidate2-healthy-validation-comparison.json`, `candidate2-performance-analysis.json`.
- Retained candidate: `candidate3-validation-comparison.json`, `candidate3-native-state-hashes.json`,
  `candidate3-common-cubin-audit-r1.json`, `candidate3-nonhead-artifact-audit-r1.json`,
  `candidate3-head-config-audit-r1.json`, `candidate3-direct-logit-comparison.json`,
  `candidate3-cli-smoke.json`, `candidate3-performance-analysis.json`, `candidate3-profile-analysis.json`.
- Noise/repeat controls: `baseline-aa-analysis.json`, `candidate3-repeat22-analysis.json`,
  `baseline-aa-repeat22-analysis.json`, `repeat22-telemetry-analysis.json`.

Raw validation/benchmark outputs remain under matching `$OUT` directories. All **12 postflight
artifact/native integrity checks** passed, and the GPU was idle at completion. The tested capacity and finite
matrix do not establish full-default-context or HF numerical acceptance. The larger unresolved performance
opportunity is the decoder-layer projections: a future experiment should locate the global tuning path's
first numerical divergence and revisit those projections with equally controlled arithmetic.
