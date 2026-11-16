# Experiment loop details

This expands the seven-step
[experiment loop](../SKILL.md#iterate-on-fusion-and-kernel-optimization) in
`SKILL.md`. Paths refer to the selected xinfer checkout, which may differ from
the installed plugin directory. Read its `AGENTS.md`, model guide,
`environments/README.md` and relevant sections of `docs/development-tools.md`;
these own the current commands and numerical policy. Inspect command `--help`
when the checkout differs.

## Before the loop

Complete the baseline profiling and brief work-elimination review in `SKILL.md`
before choosing candidates. Use [profiling.md](profiling.md) for captures and
evidence reuse.

Create or resume `investigation.json` using [record format](records/format.md).
Record the checkpoint, original baseline revision and workload/environment
references. Link existing benchmark reports for package/binary/workload hashes
and telemetry; source snapshots and settings identify each implementation
revision.

Use package metadata for compiled context capacity and per-call token bounds.
Prompt length plus forced decode steps must fit that capacity. A long prompt can
span multiple compiled prefill calls; these limits are different. Select context
positions that exercise the hypothesis.

On resumption, use the record helper's `show` overview to find decisions and
unfinished evidence, then inspect relevant entries and actual jobs. Verify that
the evidence still matches the source, artifacts, workload and environment. Apply
the attempt definition and remaining budget in
[Establish the task](../SKILL.md#establish-the-task).

## 1. Form the hypothesis

Identify the measured hotspot or execution behavior the candidate addresses.
Use the [fusion guidance](../SKILL.md#iterate-on-fusion-and-kernel-optimization) or
[kernel optimization](kernel-optimization-loop.md) notes as appropriate.

Before implementing, compare the few credible opportunities in a short list or
table. Note each region's measured cost, proposed mechanism, expected native
benefit and confidence, experiment effort/readiness and numerical/integration
risks. Distinguish the basis of each benefit claim:

- **Measured evidence:** link the result and state its scope and conditions;
  a local improvement still needs native confirmation.
- **Upper bound:** state the maximum removable cost within an explicit scope
  and its assumptions. This is a ceiling, not an expected saving.
- **Conditional estimate:** state the assumed regional improvement and how it
  translates to native latency; the calculation does not establish attainability.
- **Unknown headroom:** the mechanism is plausible but its reducible cost is
  unmeasured. Choose a cheap probe that can resolve the uncertainty.

Hotspot duration and NCU stall percentages alone do not establish attainable
savings. Account for overlap, added work and regressions elsewhere when
translating regional savings into native latency. Record why the selected
candidate is worth the remaining budget, including the value of resolving
unknown headroom. A user-specified target can focus the comparison on
interventions within it.

Before editing, add a candidate linked to its baseline revision and motivating
evidence. Record its origin and source or prior experiment: an existing option,
a reused approach or a new implementation. Record its readiness, such as a
scratch prototype, validated experiment or maintained implementation. In
`hypothesis`, explain the mechanism and why this candidate is worth trying. In
`prediction`, state the expected change in launches, traffic, kernel behavior or
native latency and what would refute it. Use qualitative expectations unless a
numeric estimate has a stated basis.

## 2. Implement the change

Inspect the relevant graph, generated code or kernel before choosing the edit.
Keep the baseline intact. Use ordinary worktrees or scratch copies for candidate
edits, with separate package/build outputs where needed. Reuse matching existing
environments and checkpoint files. Choose a focused change, but combine changes
when necessary to test the hypothesis and explain that dependency; use ablations
to distinguish their effects. Add a revision with its actual parent(s), source
snapshot/patch and settings; inner-loop revisions share the outer candidate.
Preserve the exported interface and runtime contract. Commits and publishing
require the user's instruction.

When rewriting an existing kernel or implementing a custom fusion kernel, enter
the [kernel inner loop](kernel-optimization-loop.md#kernel-inner-loop)
inside this step. First calibrate the local baseline against the corresponding
native operation or region. The design loop chooses or revises the architecture;
the implementation loop builds, tunes and tests it, including targeted interaction
experiments. Feed results back to the design loop when tuning stalls or
measurements suggest a structural limit. Both loops share the remaining budget
and existing candidate, revision, evidence and decision records. Return to step 3
with the selected implementation. Ordinary compiler-option comparisons can remain
in the outer loop.

## 3. Export

From the candidate checkout, build required native operators and export using
the model/environment guides, selected GPU and matching environment. Apply the
hypothesis-specific settings, preserve the baseline's compiled capacity and use
a fresh output directory. Inspect the log and generated implementation to confirm
the intended change reached the package. Record the command, exit status, log and
exact export identity using the [record guide](records/format.md).

## 4. Validate

Run affected local/operator checks from the model guide, then numerical/native
validation with `python -m tools.validation.validate` in the matching model
environment.

Validate the baseline too if matching evidence is unavailable. Use each
variant's matching source when evaluating its adapted eager stage. Record
validator exit status, required failures, diagnostics, runtime/native errors and
changes in actual numerical errors. Apply the
[acceptance guidance](../SKILL.md#judge-and-record-each-iteration) to
distinguish exploratory timing from accepted improvements.

## 5. Benchmark

Use `python -m tools.benchmark compare` for repeated, unprofiled paired native
measurements with identical fixed-token workloads, GPU and native executable.
Choose workload lengths, warmups and repetitions for the task. Run measurements
serially under suitable idle conditions; busy-GPU runs establish functionality
only. Report unsupported package comparisons as a limitation.

Use `decode_ms_per_token` as the default objective; it includes argmax and host
reads. Compare per-case distributions and report context length, prefill and memory
alongside it. Use A/A controls or additional samples when noise obscures the effect.

## 6. Profile and analyze

After the paired benchmark, profile the candidate's native execution following
[profiling.md](profiling.md), including its capture-reuse conditions. Compare
the candidate's kernels, launch behavior and remaining costs with the baseline
profile and the predicted mechanism. Combine this analysis with validation and
unprofiled timing to explain gains, regressions or uncertainty, and identify the
remaining bottleneck. For kernel changes, verify that inner-loop NCU evidence
matches the packaged implementation as described in the
[kernel inner loop](kernel-optimization-loop.md#kernel-inner-loop).

If unrelated recompilation changes could explain the measured gain or
regression, inspect the affected generated code and launch settings. Use a
controlled re-export or ablation as needed to isolate the intended
transformation.

### Measurement disagreements

Trigger a follow-up when measurements imply conflicting conclusions that could
change candidate selection or the claimed mechanism. First check comparable
implementations, timing scopes and conditions. If the conflict remains, run one
targeted check with a time limit within the remaining budget to distinguish the
leading explanations. Record whether it resolved the disagreement in the linked
feedback decision; otherwise retain the uncertainty and decide using the evidence
appropriate to the objective. Controlled, unprofiled native measurements establish
model performance. Improved counters alone do not imply lower latency.

## 7. Choose the next target

Use correctness, paired measurements and profile findings to keep, reject or
investigate the candidate, then choose the next experiment within the remaining
budget. Update the brief comparison as results or a new baseline change the
priorities. If an earlier step fails, record completed steps, failures and
unmeasured outcomes before choosing another attempt.

Promote an improvement to an accepted baseline only after all required checks
pass and controlled paired native measurements show a gain. Carry its
measurements, profile and numerical status into the next iteration. An
exploratory comparison baseline with required HF numerical failures retains its
failed acceptance status. Preserve the original baseline and failed attempts,
and update the candidate's readiness separately from model acceptance.

Add a decision linking the assessed revision and its evidence. If it selects a
new comparison baseline, record `next_baseline`; later candidates link that
revision and decision. Keep detailed results in their existing reports.

### Compare prediction with observation

After every assessed revision and outer attempt, including compile/validation
failures, compare the candidate's prediction (or the revision's updated local
prediction) with the observed result. Combine the comparison, explanation,
remaining unknowns and decision rationale in `assessment`; put the follow-up
in `next_action`.
Record current `readiness` separately from local and model `acceptance`. See
[record format](records/format.md) for the fields and links; short explanations
suffice.

Mark unmeasured outcomes explicitly. If the result differs from the prediction,
distinguish an unapplied change, invalid/noisy measurement, incorrect mechanism
and numerical failure using the available evidence. Reprofile when needed.
Update the benefit estimate and its evidence basis before choosing the next step.
Explain what new evidence would justify retrying an ineffective approach. Use
this feedback on resumption rather than repeating the same guess.
