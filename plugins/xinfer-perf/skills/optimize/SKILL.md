---
name: optimize
description: Profile and optimize compiled LLM inference in xinfer on NVIDIA GB10 (SM 12.1) through graph fusion and GPU kernel optimization. Use for GB10 performance investigations, fusion or kernel experiments, or continuing an existing optimization notebook. Default objective is native decode latency per token; C++ runtime changes are outside this skill.
---

# Optimize xinfer on GB10

Target **NVIDIA GB10 (SM 12.1)**. Start from native baseline profiling and
iterate on graph fusion (`compiler/common/passes/`, adapters) and GPU kernels
(`compiler/common/kernels/`, `compiler/<model>/kernels/`). Preserve the exported
interface and C++ runtime contract; runtime/CUDA Graph changes belong to a
separate future skill.

## Setup, budget, and notebook

- Run all commands in the target **xinfer checkout** using its matching GB10
  environment, model artifacts, and native binaries. Follow that checkout's
  `AGENTS.md`, model guide, `environments/README.md`, and
  `docs/development-tools.md`.
- **Objective:** native `decode_ms_per_token` (includes argmax and host reads).
  Always report context length, prefill latency, and peak memory alongside it.
  Follow any user-specified objective or workload instead.
- **Budget:** default to **3 model-level candidate attempts or 2 hours**,
  whichever comes first. A new attempt starts when implementing a new
  model-level transformation or optimizing a different region; local kernel
  revisions stay inside that attempt. Reserve time for export, validation, and
  native measurement.
- **Lab notebook:** create or resume `work/perf/<run>/log.md` using
  [references/lab-notebook.md](references/lab-notebook.md). On resumption, read
  the notebook first, check for unfinished jobs/files, and reuse matching
  evidence before running new work. Never write scratch files into the plugin
  directory.

## 1. Establish and analyze the baseline

1. Check [references/hardware.md](references/hardware.md) first — verify GB10
   clocks under load before trusting any timing.
2. Run or reuse baseline `tools.benchmark` measurements and a native `nsys`
   profile following [references/profiling.md](references/profiling.md) (reuse
   only when package, binary, device, environment, and workload match).
3. Briefly check the compiled graph/profile for **dead or avoidable work**
   (e.g., unused prompt-token projections, token-wise norms before a last-token
   slice, attention over unused cache slots) before ranking optimizations.
4. Compare the top 2–3 opportunities in the notebook by measured cost, proposed
   mechanism, effort, numerical/integration risk, and **evidence basis**:
   - **Measured:** observed directly in a matching run (local gains still need
     native confirmation).
   - **Upper bound:** ceiling of removable cost under stated assumptions, not an
     expected gain.
   - **Conditional estimate:** assumed regional speedup translated to native
     latency (accounting for overlap and added work).
   - **Unknown headroom:** plausible mechanism whose reducible cost is
     unmeasured; pick a cheap probe to resolve it.

Hotspot time and stall percentages alone do not prove attainable savings. Pick
the candidate with the best expected return for the remaining budget.

## 2. Candidate loop

1. **Hypothesize.** Before editing, log the candidate's target region,
   mechanism, expected effect (qualitative is fine unless numbers have a stated
   basis), and what observation would refute it.
2. **Implement.** Keep the baseline intact (worktree or scratch copy; reuse
   existing venvs and checkpoints). Keep edits focused; use ablations when
   changes interact.
   - For a new or rewritten kernel, run the
     [kernel inner loop](references/kernel-loop.md) first (calibrate local
     baseline against native `nsys`, iterate correctness/microbench/NCU, and
     redesign if local tweaks stall) before full-model export.
   - Do not commit or push without explicit user instruction.
3. **Export and fast-screen.** Build required native ops if touched, export to a
   fresh output directory at the baseline's compiled capacity, and confirm the
   intended change actually reached the `.pt2` (`*.graph.txt`, export logs,
   `.wrapper.cpp` — see [references/profiling.md](references/profiling.md#inspecting-compiled-packages)).
   - Run affected unit/operator tests and a quick native timing screen so
     non-improving candidates fail fast before paying for full HF validation.
4. **Validate and paired benchmark** (required before keeping any candidate).
   - Run `python -m tools.validation.validate` in the matching model
     environment.
   - Run controlled, unprofiled `python -m tools.benchmark compare` on an idle
     GPU with identical fixed-token workloads, GPU UUID, and native binary. Run
     GPU measurements serially; use A/A controls or extra repeats if noise
     obscures the effect.
5. **Profile and isolate.** Profile the candidate (`nsys`, plus `ncu` when
   diagnosing kernel bottlenecks) and compare against baseline and hypothesis.
   - Check for unrelated Inductor recompilation drift (renamed kernels can miss
     retained tuning caches, and compiler flags can change decomposition).
     Inspect unaffected kernels before attributing gains.
   - If measurements materially disagree, run at most one bounded check to
     distinguish causes; unprofiled paired native timing governs model latency.
6. **Decide and record.** Append predicted vs observed latency, correctness,
   remaining unknowns, and outcome (`keep`, `reject`, or `revise`) to the
   notebook, including failed or inconclusive runs.

## Correctness and promotion rules

Follow `AGENTS.md` and `docs/development-tools.md#shared-numerical-policy`:

- Report **required HF failures**, **diagnostics**, and **runtime/native
  errors** separately. Diagnostics do not fail acceptance.
- Runtime/native errors (nonfinite values, invalid state, native/Python
  mismatch) disqualify a candidate.
- Required HF numerical failures allow exploratory timing, but the artifact
  remains an **exploratory baseline** (failed acceptance) — never an accepted
  baseline.
- Promote a candidate to the accepted baseline only when **all required checks
  pass** and controlled paired native timing improves. Preserve the original
  baseline and failed attempts.

On stopping, report the reviewable diff/paths, measured native outcome,
numerical status, limitations, and concrete next steps.

## References

- [references/profiling.md](references/profiling.md) — `nsys` phase separation, `ncu` captures, `.pt2` inspection, and baseline calibration.
- [references/kernel-loop.md](references/kernel-loop.md) — Kernel inner loop and xinfer Triton / native C-shim integration.
- [references/hardware.md](references/hardware.md) — GB10 (SM 12.1) profile, decode patterns, and gotchas.
- [references/lab-notebook.md](references/lab-notebook.md) — Markdown notebook template for tracking and resuming runs.
