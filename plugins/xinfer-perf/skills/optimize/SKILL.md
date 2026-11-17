---
name: optimize
description: Profile and optimize compiled LLM inference in xinfer on NVIDIA GB10 through fusion and GPU kernel optimization. Use for performance investigations, fusion and kernel experiments, or continuing an existing xinfer optimization. Profile the baseline before choosing candidates. The default objective is native decode latency per token; runtime implementation changes are outside this skill.
---

# Optimize xinfer compilation

Optimize for NVIDIA GB10 (SM 12.1). Start from baseline profiling and iterate on
fusion and kernel optimization.
Xinfer's custom graph passes, inference adapters and custom operators are
implementation mechanisms for these goals.
Preserve the current exported interface and C++ runtime contract.

## Establish the task

Run project commands from the target xinfer checkout using the matching Python
environment, model artifacts, native binaries and GPU. Follow that checkout's
`AGENTS.md` and relevant model/environment guides.

Default to native **decode milliseconds per token**, with context length,
prefill latency and memory reported alongside it. Follow a different objective
or workload when the user supplies one. Unless specified otherwise, budget three
model-level candidate attempts or two hours of investigation, whichever comes
first; record the start and use command timeouts appropriate to the remaining
time. An attempt starts when implementing a new model-level transformation or
optimizing a different region. Local kernel revisions stay within that attempt
and all attempts share the investigation's time budget. Reserve time for
integration and native measurement.

Create or resume the
[linked investigation record](references/records/format.md). It connects
candidates, implementation revisions, evidence and decisions to the existing
reports. When continuing, start with the record helper's `show` overview, then
inspect relevant entries and linked notes/results. Reuse matching evidence and
identify unfinished work before launching another job.

## Profile the baseline

Consult the [hardware gotchas](references/hardware.md#gotchas) before collecting
performance measurements.

Establish baseline measurements with the existing benchmark, then profile the
native execution before choosing optimization candidates. Follow
[profiling.md](references/profiling.md): use `nsys` for native execution and
targeted `ncu` captures for kernel analysis. NCU evidence is required for
baseline and candidate kernel analysis in the inner loop. Invoke the profilers
directly.

Reuse captures under the matching conditions in `profiling.md`. Record the
observations that motivate the investigation. If usable profiling evidence is
unavailable, collect it or report the gap and defer candidate selection.

## Review work elimination

Briefly check the measured path for computation whose results the workload does
not consume, such as vocabulary projections for unused prompt rows, token-wise
normalization before a last-row slice or attention over unused cache capacity.
Confirm the work is present in the compiled artifact; it may already be removed.

Preserve required outputs, all cache/state updates, masks and positions. Trace
state dependencies before removing work whose output is discarded. Changed
shapes can still alter floating-point rounding; validate the compiled result.

Record applicable opportunities, already handled cases or that none apply.
Estimate the removable cost in the requested phase when evidence permits.

Reuse the review for an unchanged baseline; revisit it when the graph, workload
or new evidence warrants it. Then focus the investigation on fusion and kernel
optimization.

## Iterate on fusion and kernel optimization

Choose the next candidate from the remaining measured costs:

- **Fusion:** combine related operations to reduce launches, intermediate memory
  traffic or repeated loads. Measure the whole replacement, including
  preparation, copies and workspace work. Account for register pressure and
  validate changes to rounding boundaries.
- **Kernel optimization:** design and implement new kernels or rewrite existing
  ones, improving dataflow, layout, scheduling and instruction use. See
  [kernel optimization](references/kernel-optimization-loop.md).

For each candidate, complete the experiment loop.
[experiment-loop.md](references/experiment-loop.md) expands these seven steps with the
existing commands and evidence to retain:

1. **Form the hypothesis.** Tie it to an observation in the current baseline
   profile and explain the mechanism it could improve. Briefly compare the
   credible opportunities by attainable native benefit, evidence,
   effort/readiness and numerical/integration risk. Use the
   [candidate comparison](references/experiment-loop.md#1-form-the-hypothesis) to
   distinguish measured evidence, upper bounds, conditional estimates and
   unknown headroom. Record why the selected experiment is worth the remaining
   budget; the largest hotspot may have little headroom. Record its origin,
   readiness, expected change and refuting evidence before editing; a qualitative
   prediction is sufficient when numbers lack a basis.
2. **Implement the change.** Inspect the relevant graph, generated code or
   kernel. Use an isolated checkout or scratch experiment, preserving the
   baseline and the user's other changes. Keep the change focused; use ablations
   when changes interact. For existing kernel rewrites or custom fusion kernels,
   use the
   [kernel inner loop](references/kernel-optimization-loop.md#kernel-inner-loop)
   before full-model export, starting with baseline calibration against
   native execution. Its design loop chooses or revises the architecture; its
   implementation loop builds and evaluates it, including plausible interactions.
   Feed results back to the design loop when local revisions stop improving or
   measurements suggest a structural limit.
   Consider Triton, CUDA C++ and targeted PTX according to the measured bottleneck.
   Commits and publishing require the user's instruction.
3. **Export.** Use the project's existing export interface. Verify that the
   intended code reaches the measured compiled artifact.
4. **Validate.** Run affected local/operator checks and project numerical/native
   validation, applying the correctness criteria below.
5. **Benchmark.** Compare the candidate with the current baseline using
   controlled, unprofiled paired native measurements.
6. **Profile and analyze.** Profile the candidate's native execution with the
   relevant workload, following [profiling.md](references/profiling.md). Compare
   its kernels, launches, traffic and remaining costs with the baseline profile
   and the hypothesis. Explain measured gains, regressions or uncertainty and
   identify what now limits the requested metric. Give material measurement
   disagreements a
   [bounded follow-up](references/experiment-loop.md#measurement-disagreements).
7. **Choose the next target.** Use correctness, benchmark and profile findings
   to keep, reject or investigate the change and select the next experiment
   within the remaining budget. Promote an improvement to an accepted baseline
   only after all required checks pass and paired native measurements show a
   gain. Preserve the original baseline and failed attempts.

If a step fails, record completed steps, failures and unmeasured outcomes before
choosing the next attempt. An existing candidate or a specific kernel can focus
the investigation. Delegate independent investigation or review when useful,
while running measurements serially on the selected GPU.

### Judge and record each iteration

Apply the project validators' current policy and acceptance limits. Record
required failures, diagnostics and runtime/native errors separately, including
changes in actual numerical errors. Diagnostics do not change the validator's
pass/fail result. Required HF numerical failures permit exploratory timing while
retaining failed acceptance. Runtime/native errors, including nonfinite outputs,
invalid state or native/Python disagreement, make a candidate unsuitable for a
performance recommendation. An artifact with unresolved required HF numerical
failures used for further comparisons remains an exploratory baseline.

Judge performance using repeated, controlled, unprofiled native measurements.
Report per-workload tradeoffs and investigate noise when the gain is ambiguous.

After each assessed revision and outer attempt, use the
[feedback format](references/experiment-loop.md#compare-prediction-with-observation)
to add a linked decision with predicted versus observed performance and
correctness, including failed or inconclusive outcomes. On stopping, give the
reviewable change, measured outcome, numerical status, limitations and what to
try next.

Runtime/generation-loop changes belong to a future separate skill. If the
evidence points there, report that opportunity without expanding this task.
