---
name: optimize
description: Profile and optimize compiled LLM inference in xinfer through fusion and GPU kernel optimization. Use for performance investigations, compiler/kernel experiments, or continuing an existing xinfer optimization. Profile the baseline before choosing candidates. The default objective is native decode latency per token; runtime implementation changes are outside this skill.
---

# Optimize xinfer compilation

Start from baseline profiling and iterate on fusion and kernel optimization.
Compiler options, graph passes, inference adapters and custom operators are
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
time. Kernel revisions belong to their outer candidate and share that time
budget. Changing the target or model-level hypothesis starts another attempt;
reserve time for integration and native measurement.

When continuing an experiment, read its notes, commands and result files first.
Reuse relevant evidence instead of repeating completed work. Note any changes
in source, artifacts, workload or environment before comparing old and new runs.

## Profile the baseline

Consult the relevant [hardware notes](references/techniques/hardware.md#target-notes)
before autotuning or timing; use target-specific load-health observations.

Establish baseline measurements with the existing benchmark, then profile the
native execution before choosing optimization candidates. Follow
[profiling.md](references/profiling.md): use `nsys` for native execution and
targeted `ncu` captures for kernel analysis. NCU evidence is required for baseline
and candidate kernel analysis in the inner loop. Invoke the profilers directly.

An existing capture can satisfy this step if it matches the baseline artifacts,
hardware/software environment and relevant workload. Check that match on resume.
Record the capture/result paths, concrete observations and how they relate to
decode (or the requested objective). Source inspection, historical results from
another target and intuition alone do not replace profiling this baseline.
If usable profiling evidence is unavailable, collect it or report what is
missing and defer candidate selection.

## Review work elimination

Briefly check the measured path for computation whose results the workload does
not consume, such as vocabulary projections for unused prompt rows, token-wise
normalization before a last-row slice or attention over unused cache capacity.
Confirm the work is present in the compiled artifact; it may already be removed.

Preserve required outputs, all cache/state updates, masks and positions. Trace
state dependencies before removing work whose output is discarded. Changed
shapes can still alter floating-point rounding; validate the compiled result.

Record applicable opportunities, already handled cases or that none apply.
Estimate the removable cost in the requested phase when evidence permits;
prefill-only savings do not improve single-token decode. Choose candidates by
expected native benefit, with the same experiment loop and budget as other changes.

Reuse the review for an unchanged baseline; revisit it when the graph, workload
or new evidence warrants it. Then focus the investigation on fusion and kernel
optimization.

## Iterate on fusion and kernel optimization

Choose the next candidate from the remaining measured costs:

- **Fusion:** combine related operations to reduce launches, intermediate memory
  traffic or repeated loads. See [fusion](references/techniques/fusion.md).
- **Kernel optimization:** improve a required operation through tuning, library
  selection, layout/scheduling changes or rewriting its implementation. See
  [kernel optimization](references/techniques/kernel-optimization.md).

These goals can overlap. Consider supported compiler options as low-cost
candidates when the evidence supports them; use graph passes, adapters or custom
operators when the intervention needs them. Do not exhaust configuration options
or implement a custom kernel merely to follow a fixed sequence.

For each candidate, complete the experiment loop. [experiments.md](references/experiments.md)
expands these seven steps with the existing commands and evidence to retain:

1. **Form the hypothesis.** Tie it to an observation in the current baseline
   profile and explain the mechanism it could improve. Use the
   [symptom index](references/optimization-index.md) for relevant technique notes.
   Record its origin, readiness, expected change and refuting evidence before
   editing; a qualitative prediction is sufficient when numbers lack a basis.
2. **Implement the change.** Inspect the relevant graph, generated code or kernel.
   Use an isolated checkout or scratch experiment, preserving the baseline and
   the user's other changes. Keep the change focused; use ablations when changes
   interact. For existing kernel rewrites or custom fusion kernels, use the
   [kernel inner loop](references/techniques/kernel-optimization.md#kernel-inner-loop):
   local hypothesis → edit/compile → correctness → microbenchmark → NCU analysis
   → next revision. Consider Triton, CUDA C++ and targeted PTX according to the
   measured bottleneck. Commits and publishing require the user's instruction.
3. **Export.** Use the project's existing export interface. Verify that the
   intended code or compiler setting reaches the measured compiled artifact.
4. **Validate.** Run affected local/operator checks and project numerical/native
   validation, applying the correctness criteria below.
5. **Benchmark.** Compare the candidate with the current baseline using controlled,
   unprofiled paired native measurements. Kernel microbenchmarks guide the work;
   native model measurements establish its effect on inference.
6. **Profile and analyze.** Profile the candidate's native execution with the
   relevant workload, following [profiling.md](references/profiling.md). Compare
   its kernels, launches, traffic and remaining costs with the baseline profile
   and the hypothesis. Explain measured gains, regressions or uncertainty and
   identify what now limits the requested metric.
7. **Choose the next target.** Use correctness, benchmark and profile findings to
   keep, reject or investigate the change and select the next experiment within
   the remaining budget. When a validated improvement becomes the new baseline,
   use its measurements and profile for the next iteration; preserve the original
   baseline and failed attempts.

Reuse matching baseline evidence across attempts. Candidate profiling can reuse
an existing capture only when it matches that candidate's artifacts, environment
and workload; the original baseline capture does not describe changed code.
If an earlier step fails, record the failure and available evidence before
choosing the next attempt. An existing candidate or a specific kernel can focus
the investigation, with matching baseline evidence established first. Delegate
independent investigation or review when useful, while running measurements
serially on the selected GPU.

### Judge and record each iteration

Keep numerical correctness, measurement conditions and performance conclusions
distinct. Existing project validators own numerical policy. Preserve their
failures and compare changes in the actual errors, not only failure counts.
Known HF failures permit exploratory timing, but do not establish model
acceptance. Nonfinite outputs, invalid state or native/Python disagreement make
a candidate unsuitable for a performance recommendation.

Use repeated measurements and investigate noise when a gain is ambiguous; an
A/A comparison is useful when needed. Report per-workload tradeoffs rather than
selecting only favorable cases. Profiler timings and busy-GPU runs cannot replace
an unprofiled, controlled native comparison.

Keep ordinary notes with the hypothesis, source revision/patch, exact commands,
artifact/result paths, observations and next action in the experiment directory.
After each attempt, compare the prediction with observed performance and
correctness, including failed or inconclusive outcomes. Use the short
[feedback format](references/experiments.md#compare-prediction-with-observation)
to explain mismatches and choose the next fusion or kernel experiment within the
remaining budget. Retain failed attempts. On stopping, give the reviewable
change, measured outcome, numerical status, limitations and what to try next.

Runtime/generation-loop changes belong to a future separate skill. If the
evidence points there, report that opportunity without expanding this task.
