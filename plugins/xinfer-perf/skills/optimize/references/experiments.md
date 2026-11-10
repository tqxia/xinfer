# Compile, validate and measure a candidate

Read this when moving from a hypothesis to an experiment. Paths mentioned here
refer to the selected xinfer checkout, which may differ from the installed
plugin directory. Read its `AGENTS.md`, model guide, `environments/README.md` and
relevant sections of `docs/development-tools.md`; these own the current commands
and numerical policy. Inspect command `--help` when the checkout differs.

## Establish comparable inputs

Record the checkpoint and baseline package, source revision plus local patch,
Python environment, native executable/library, GPU UUID and workload. Existing
benchmark reports already record package/binary/workload hashes and telemetry;
reference those files instead of creating another provenance system.

Use package metadata for compiled context capacity and per-call token bounds.
Prompt length plus forced decode steps must fit that capacity. A long prompt can
span multiple compiled prefill calls; these limits are different. Select context
positions that exercise the hypothesis rather than assuming one short prompt
represents decode everywhere.

Keep the baseline intact. Use ordinary worktrees or scratch copies for candidate
edits, with separate package/build outputs where needed. Reuse matching existing
environments and checkpoint files. Choose a focused change, but combine changes
when necessary to test the hypothesis and explain that dependency.

## Use the existing interfaces

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
"$PY" -m tools.validation.validate --model-dir "$MODEL_DIR" \
  --packages "$CANDIDATE" --native "$VALIDATOR" --output "$VALIDATION"
```

Run relevant local/operator tests as described by the model guide. Validate the
baseline too if matching evidence is unavailable. Use each variant's matching
source when evaluating its adapted eager stage. Preserve required failures and
their error metrics, including existing HF failures; do not suppress nonzero
validation exits or reduce validation to token agreement. Numerical findings and
native execution/parity are separate from performance findings.

The shared comparison uses the same native executable for both package bundles:

```bash
"$PY" -m tools.benchmark compare --model-dir "$MODEL_DIR" \
  --baseline-packages "$BASELINE" --candidate-packages "$CANDIDATE" \
  --native "$BENCHMARK" --gpu "$GPU_UUID" --output "$COMPARISON"
```

Choose `--lengths`, `--decode-steps`, `--warmup`, `--repeats` and `--rounds` for the
task. The controller alternates process order and checks idle conditions. Run
GPU measurements sequentially. `--allow-busy` is for functional exploration;
its provisional timings do not demonstrate an improvement.

The comparator requires compatible package contracts and logical weights. If a
packing/kernel experiment needs different mappings or native libraries, inspect
that mismatch and record what comparison support is missing. Do not bypass the
check or silently measure both bundles with the wrong operator library.

## Interpret and continue

Use `decode_ms_per_token` as the default objective: native fixed-token decode
includes argmax and host reads. Read per-case distributions, prefill and memory
alongside it. Compare repeated rounds; use an A/A run or additional samples when
noise obscures the effect. Profiled and unprofiled timings are separate evidence.

Inspect export logs/generated code to confirm the intended change was applied.
Record the hypothesis, patch, exact commands, measurements, numerical changes,
conclusion and next action in a short note. Retain failed attempts. On resumption,
read that note and check for still-running work before launching another job.
Report a faster exploratory candidate separately from model acceptance when
required numerical checks remain unresolved.
