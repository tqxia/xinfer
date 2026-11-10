---
name: optimize
description: Profile and optimize compiled LLM inference in xinfer through compiler configuration, graph passes, inference adapters, fusion and GPU kernels. Use for performance investigations, compiler/kernel experiments, or continuing an existing xinfer optimization. Profile the baseline before choosing candidates. The default objective is native decode latency per token; runtime implementation changes are outside this skill.
---

# Optimize xinfer compilation

Use the existing coding agent to investigate, implement and evaluate focused
compiler-side experiments. Choose the approach from the evidence; a configuration
change, graph rewrite, adapter or kernel can each be the right intervention.
Preserve the current exported interface and C++ runtime contract.

## Establish the task

Locate the xinfer checkout separately from this installed plugin. Read its
`AGENTS.md`, `README.md` and relevant model/environment guides. Resolve the actual
model, interpreter, package bundle, native executable and GPU from those sources
and the user's environment. Run project commands from that checkout, using the
matching interpreter. An agent on another machine can use its ordinary SSH tools.

Default to native **decode milliseconds per token**, with context length,
prefill latency and memory reported alongside it. Follow a different objective
or workload when the user supplies one. Unless specified otherwise, budget three
candidate attempts or two hours of investigation, whichever comes first; record
the start and use command timeouts appropriate to the remaining time.

When continuing an experiment, read its notes, commands and result files first.
Reuse relevant evidence instead of repeating completed work. Note any changes
in source, artifacts, workload or environment before comparing old and new runs.

## Profile the baseline

Establish baseline measurements with the existing benchmark, then profile the
native execution before choosing optimization candidates. Follow
[profiling.md](references/profiling.md): start with `nsys` and use targeted `ncu`
captures when kernel counters would help explain an observed hotspot. Invoke the
profilers directly; both tools are not required for every investigation.

An existing capture can satisfy this step if it matches the baseline artifacts,
hardware/software environment and relevant workload. Check that match on resume.
Record the capture/result paths, concrete observations and how they relate to
decode (or the requested objective). Source inspection, historical results from
another target and intuition alone do not replace profiling this baseline.
If usable profiling evidence is unavailable, collect it or report what is
missing and defer candidate selection.

## Form hypotheses and experiment

- Tie each candidate to an observation in the baseline profile and explain the
  mechanism it could improve. Use the [symptom index](references/optimization-index.md)
  to load relevant common or hardware-specific technique notes as needed.
  Record the expected metric/latency change and what would refute the hypothesis
  before editing; a qualitative prediction is sufficient when numbers lack a basis.
  Inspect the relevant graph, generated code or kernel before changing it.
  Keep an experiment focused enough to explain its effect; use ablations when
  several changes interact.
- Work in an isolated checkout or scratch experiment, preserving the baseline
  and the user's other changes. Use normal Git and filesystem tools; commits
  and publishing require the user's instruction.
- Follow [experiments.md](references/experiments.md) for xinfer's existing
  export, validation and paired benchmark interfaces. Verify that the candidate
  code or compiler setting actually reaches the measured compiled artifact.
  Kernel microbenchmarks guide the work; native model measurements establish
  its effect on inference.

An existing candidate or a specific kernel can focus the investigation; first
establish the matching baseline profiling evidence. Reuse that evidence for
further candidates against the same baseline. Reprofile when a changed baseline
or workload makes it stale, or when results need explanation. Delegate independent
investigation or review when useful, while running measurements serially on the
selected GPU.

## Interpret and retain the result

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
to explain mismatches and guide the next hypothesis. Retain failed attempts.
On stopping, give the reviewable change, measured outcome, numerical status,
limitations and what to try next.

Runtime/generation-loop changes belong to a future separate skill. If the
evidence points there, report that opportunity without expanding this task.
