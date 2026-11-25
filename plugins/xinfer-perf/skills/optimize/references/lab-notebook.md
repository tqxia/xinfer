# Lab notebook format

Keep one append-only Markdown notebook per investigation at
`work/perf/<run>/log.md` (inside the target xinfer checkout's ignored `work/`
directory, never inside the plugin directory). Link existing benchmark,
validation, and profiler files rather than duplicating their bodies.

On resumption, read `log.md` first to see active baselines, completed attempts,
and unfinished runs. Preserve completed entries and failed attempts; append new
revisions and decisions rather than rewriting history.

## Template

```markdown
# <Investigation title>

- **Started:** YYYY-MM-DD HH:MM UTC
- **Budget:** 3 candidates / 120 min
- **Objective:** native `decode_ms_per_token` (also track context, prefill, peak memory)
- **Host / GPU:** `<host>`, GB10 (`GPU_UUID`), clocks verified under load: yes/no
- **Baseline:** commit `<sha>`, model `<model-dir>`, packages `<path>`, native `<path>`
- **Workload:** `<path/to/workload.json>` (capacity `<C>`, prompt lengths, decode steps)

## Baseline

- **Benchmark:** `<path/to/benchmark>` — decode `<X> ms/tok`, prefill `<Y> ms`, memory `<Z> MiB`
- **Validation:** `<path/to/validation>` — required failures: `<N>`, diagnostics: `<M>`, native parity: pass/fail
- **Profile:** `<path/to/nsys-rep>` (decode steps `<a..b>`), `<path/to/ncu-rep>`
- **Work-elimination check:** `<applicable / already handled / none>`

### Ranked opportunities

| Region | Measured cost | Proposed mechanism | Basis | Risk / effort |
| --- | --- | --- | --- | --- |
| ... | ... | ... | measured / upper bound / conditional / unknown | ... |

## Candidate C1: <short name>

- **Baseline compared against:** `r0` (`<packages>`)
- **Hypothesis:** `<mechanism and why it beats alternatives for remaining budget>`
- **Prediction:** `<expected change in launches, traffic, counters, or ms/tok>`
- **Refutation:** `<observation that would disprove the hypothesis>`

### Revisions

- **r1** (parent `r0`, worktree/patch `<path>`, settings `<flags>`): `<what changed>`
  - Local calibration & microbench: `<baseline vs r1, NCU/SASS summary, artifact paths>`
  - Note: `<keep / tweak / redesign and why>`

### Model evaluation

- **Export:** `<command>`, output `<packages>`, `.pt2` / pass check (`replaced > 0`, unaffected kernels checked)
- **Fast screen:** `<local tests + quick timing>`
- **Validation:** `<path>` — required failures / diagnostics / native parity (`pass` / `fail` / `not run`)
- **Paired benchmark:** `<path>` — baseline vs candidate per workload (`decode_ms_per_token`, prefill, memory)
- **Candidate profile:** `<nsys / ncu paths>` — predicted vs observed kernel behavior

### Decision

- **Outcome:** `keep` / `reject` / `revise`
- **Acceptance:** local `passed|failed|unverified`, model `passed|failed|unverified`
- **Promoted as baseline:** `no` / `yes (accepted)` / `yes (exploratory — failed HF check)`
- **Assessment:** `<predicted vs observed, explanation, remaining unknowns>`
- **Next action:** `<what to try next or why to stop>`
```
