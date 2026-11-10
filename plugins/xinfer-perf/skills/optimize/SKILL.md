---
name: optimize
description: Investigate and optimize compiled LLM inference in xinfer through compiler configuration, graph passes, inference adapters, fusion and GPU kernels. Use for performance investigations, compiler/kernel experiments, or continuing an existing xinfer optimization. The default objective is native decode latency per token; runtime implementation changes are outside this skill.
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

## Investigate and experiment

- Start from the user's evidence or establish a baseline with the existing
  benchmark. Consult [profiling.md](references/profiling.md) when a capture or
  kernel-level investigation would answer a concrete question. Invoke `nsys`
  and `ncu` directly and use their built-in summaries and exports.
- Choose a hypothesis and identify what result would support or reject it.
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

Adapt this sequence to the task. A supplied profile, existing candidate or
specific kernel can be the starting point. Delegate independent investigation
or review when useful, while running measurements serially on the selected GPU.

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
Retain failed and inconclusive attempts. On stopping, give the reviewable change,
measured outcome, numerical status, limitations and what to try next.

Runtime/generation-loop changes belong to a future separate skill. If the
evidence points there, report that opportunity without expanding this task.
