# Experiment loop details

This expands the seven-step [experiment loop](../SKILL.md#iterate-on-fusion-and-kernel-optimization)
in `SKILL.md`. Paths refer to the selected xinfer checkout, which may differ from
the installed plugin directory. Read its `AGENTS.md`, model guide,
`environments/README.md` and relevant sections of `docs/development-tools.md`;
these own the current commands and numerical policy. Inspect command `--help`
when the checkout differs.

## Before the loop

Establish baseline measurements and the matching native profile described in
[profiling.md](profiling.md) before choosing candidates. Reuse matching evidence;
if it is missing or stale, profile first. The brief work-elimination review stays
in `SKILL.md`; use the measured costs to focus fusion and kernel experiments.

Record the checkpoint and baseline package, source revision plus local patch,
Python environment, native executable/library, GPU UUID and workload. Existing
benchmark reports already record package/binary/workload hashes and telemetry;
reference those files instead of creating another provenance system.

Use package metadata for compiled context capacity and per-call token bounds.
Prompt length plus forced decode steps must fit that capacity. A long prompt can
span multiple compiled prefill calls; these limits are different. Select context
positions that exercise the hypothesis rather than assuming one short prompt
represents decode everywhere.

On resumption, read the notes, commands and results, and check for still-running
work before launching another job. Verify that the evidence still matches the
source, artifacts, workload and environment. Work within the remaining candidate
and time budget established in `SKILL.md`. Local kernel revisions belong to
their outer candidate and share the same clock; changing the target or
model-level hypothesis starts another outer attempt. Reserve time for export,
full validation and native measurement.

## 1. Form the hypothesis

Identify the measured hotspot or execution behavior the candidate addresses.
Use the [symptom index](optimization-index.md) for relevant technique notes;
hardware-specific recipes require a match to the actual target and toolchain.

Before editing, record the candidate's origin and source or prior experiment:
an existing option, a reused approach or a new implementation. Record its
readiness, such as a scratch prototype, validated experiment or maintained
implementation. Predict the mechanism and expected change in launches, traffic,
kernel behavior or native latency, and what would refute the hypothesis. Use
qualitative expectations unless a numeric estimate has a stated basis.

## 2. Implement the change

Inspect the relevant graph, generated code or kernel before choosing the edit.
Keep the baseline intact. Use ordinary worktrees or scratch copies for candidate
edits, with separate package/build outputs where needed. Reuse matching existing
environments and checkpoint files. Choose a focused change, but combine changes
when necessary to test the hypothesis and explain that dependency; use ablations
to distinguish their effects. Preserve the exported interface and runtime
contract. Commits and publishing require the user's instruction.

When rewriting an existing kernel or implementing a custom fusion kernel, enter
the [kernel inner loop](techniques/kernel-optimization.md#kernel-inner-loop)
inside this step. Use Triton, CUDA C++ or targeted PTX as the measured bottleneck
and target support warrant. Iterate on local correctness, microbenchmarks and
required baseline/candidate NCU analysis before returning to step 3. Ordinary
compiler-option comparisons can remain in the outer loop. New operator or
packaging boundaries need an early export/AOTI and C++ execution proof before
extensive tuning; local kernel revisions do not each require a full-model export.

## 3. Export

Set paths from the selected checkout and environment. In these examples, `PY`,
`MODEL_DIR`, `CANDIDATE`, `BASELINE`, `BENCHMARK`, `VALIDATOR` and result paths are
absolute execution-host paths. `EXPORT_MODULE` is the module named by the model
guide, `CAPACITY` is the baseline's chosen compiled capacity, and `GPU_UUID`
selects the device. Use fresh result directories and an existing log parent.

From the candidate checkout, build required native operators using the existing
environment guide, then export with the hypothesis-specific options supported by
that exporter. Preserve its exit status and log:

```bash
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
"$PY" -m "$EXPORT_MODULE" --model-dir "$MODEL_DIR" \
  --max-context-length "$CAPACITY" --output "$CANDIDATE" > "$EXPORT_LOG" 2>&1
```

Inspect the export log and generated implementation to confirm the intended
change reached the package. A source edit or flag alone does not establish that
the measured artifact contains the change.

## 4. Validate

Run affected local/operator checks as described by the model guide, then the
project's numerical/native validation:

```bash
"$PY" -m tools.validation.validate --model-dir "$MODEL_DIR" \
  --packages "$CANDIDATE" --native "$VALIDATOR" --output "$VALIDATION"
```

Validate the baseline too if matching evidence is unavailable. Use each
variant's matching source when evaluating its adapted eager stage. Preserve
validator exit status, required failures, diagnostics and changes in actual
errors; do not reduce validation to failure counts or token agreement.

Known HF failures permit exploratory timing while retaining the failed
acceptance result. Nonfinite outputs, invalid state or native/Python disagreement
make a candidate unsuitable for a performance recommendation. Apply the project
validators' policy; numerical findings and native parity remain separate from
performance findings.

## 5. Benchmark

Use controlled, unprofiled paired native measurements. The shared comparison
uses the same native executable for both package bundles:

```bash
"$PY" -m tools.benchmark compare --model-dir "$MODEL_DIR" \
  --baseline-packages "$BASELINE" --candidate-packages "$CANDIDATE" \
  --native "$BENCHMARK" --gpu "$GPU_UUID" --output "$COMPARISON"
```

Choose `--lengths`, `--decode-steps`, `--warmup`, `--repeats` and `--rounds` for the
task. The controller alternates process order and checks idle conditions. Run
GPU measurements sequentially. `--allow-busy` is for functional exploration;
its provisional timings do not demonstrate an improvement.

Use `decode_ms_per_token` as the default objective: native fixed-token decode
includes argmax and host reads. Read per-case distributions, context length,
prefill and memory alongside it. Compare repeated rounds; use an A/A run or
additional samples when noise obscures the effect. Kernel microbenchmarks guide
implementation choices; native measurements establish the inference effect.

The comparator requires compatible package contracts and logical weights. If a
packing/kernel experiment needs different mappings or native libraries, inspect
that mismatch and record what comparison support is missing. Do not bypass the
check or silently measure both bundles with the wrong operator library.

## 6. Profile and analyze

After the paired benchmark, profile the candidate's native execution following
[profiling.md](profiling.md). Reuse a capture only if it matches that candidate,
environment and workload. Compare its kernels, launch behavior and remaining
costs with the baseline profile and the predicted mechanism. Combine this
analysis with validation and unprofiled timing to explain gains, regressions or
uncertainty, and identify the remaining bottleneck. Profiler timings are separate
evidence from the unprofiled paired comparison. The original baseline capture
does not describe changed candidate code.

Kernel analysis requires NCU evidence. Reuse inner-loop captures only when the
exported kernel implementation, launch configuration and measured conditions
match; otherwise capture the packaged kernel. Local kernel timings and profiles
do not replace the paired C++ benchmark or native execution profile.

## 7. Choose the next target

Use correctness, paired measurements and profile findings to keep, reject or
investigate the candidate, then choose the next experiment within the remaining
budget. If an earlier step fails, record the failure and available evidence
before choosing another attempt; do not claim measurements or profiles from
steps that did not run.

When a validated improvement becomes the new baseline, carry its measurements,
profile and numerical status into the next iteration. Preserve the original
baseline and failed attempts. Update readiness after the attempt; implementation
maturity remains separate from model acceptance.

Keep a short note with the hypothesis, source revision/patch, exact commands,
artifact and result paths, performance, numerical changes, profile observations,
decision and next action. On stopping, summarize the reviewable change, measured
outcome, numerical status, limitations and what to try next.
Report a faster exploratory candidate separately from model acceptance when
required numerical checks remain unresolved.

### Compare prediction with observation

After every attempt, including compile/validation failures, compare the recorded
prediction with the observed result. A short note is enough:

```text
Candidate: origin/reference; initial readiness.
Before: profile/observation; proposed change; expected effect; refuting evidence.
After: implementation applied; current readiness; performance/correctness; result paths.
Feedback: expected versus observed; explanation or unknowns; decision/next action.
```

Record unmeasured effects as unknown, not as improvements. If the result differs
from the prediction, distinguish an unapplied change, invalid/noisy measurement,
incorrect mechanism and numerical failure using the available evidence. Reprofile
when needed. Explain what new evidence would justify retrying an ineffective
approach. Use this feedback on resumption rather than repeating the same guess.
