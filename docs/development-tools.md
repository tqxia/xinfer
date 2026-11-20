# Development tools

Run commands from the repository root with the matching model interpreter.
Select `Q3_PY`, `Q35_PY` and the native build through the
[environment guide](../environments/README.md). Build the attention library before
export or standalone attention tests. Full-model tools require complete exported
bundles; use fresh output paths. The [model guides](../README.md#getting-started)
cover model-specific export settings and GPU tests.

This guide defines the shared commands and acceptance policy.
[Validation status](validation-status.md) records results for particular builds;
benchmark reports remain independent of accuracy validation.

## Formatting

Use Python 3.12 to install the pinned formatters in an isolated environment:

```bash
python3.12 -m venv work/format-venv
work/format-venv/bin/python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` pins Ruff 0.17.3 and clang-format 23.1.3. `ruff.toml`
formats Python with four-space indentation, double quotes and 100-column lines;
`.clang-format` uses LLVM style with two-space indentation and the same column
limit. Import and include order are retained. Run these commands from the
repository root to format tracked source files:

```bash
git ls-files -z '*.py' | xargs -0 work/format-venv/bin/ruff format
git ls-files -z '*.cpp' '*.h' | xargs -0 work/format-venv/bin/clang-format -i
```

Check formatting without changing files:

```bash
git ls-files -z '*.py' | xargs -0 work/format-venv/bin/ruff format --check
git ls-files -z '*.cpp' '*.h' | xargs -0 work/format-venv/bin/clang-format --dry-run --Werror
```

## CPU unit tests

Tests follow the code they cover. Preparation, benchmark and validation tooling
tests live under `tools/<area>/tests/`; shared tooling and model-specific package
ABI checks are tested under `tools/tests/`. Compiler metadata/export, model loading,
adapters and kernel tests remain under `compiler/*/tests/`; environment tests
remain under `environments/tests/`.

These CPU checks need no checkpoints, prebuilt model packages, project native
binaries or GPU. Most use temporary fixtures and mocks; `test_weights` compiles
tiny CPU AOTI packages and requires a working C++ toolchain.

```bash
CUDA_VISIBLE_DEVICES='' "$Q3_PY" -m unittest \
  compiler.common.tests.test_context compiler.common.tests.test_metadata \
  compiler.common.tests.test_export compiler.common.tests.test_weights \
  environments.tests.test_check environments.tests.test_setup \
  environments.tests.test_entrypoints \
  compiler.qwen3.tests.test_export \
  tools.tests.test_common tools.tests.test_packages \
  tools.prepare.tests.test_common tools.prepare.tests.test_qwen3 \
  tools.benchmark.tests.test_benchmark \
  tools.validation.tests.test_common tools.validation.tests.test_qwen3_runtime \
  tools.validation.tests.test_qwen3 tools.validation.tests.test_validation \
  tools.validation.tests.test_validation_cli \
  tools.validation.tests.test_validation_chain \
  tools.validation.tests.test_validation_runtime \
  tools.validation.tests.test_validation_native \
  tools.validation.tests.test_validation_modes tools.validation.tests.test_context
CUDA_VISIBLE_DEVICES='' "$Q35_PY" -m unittest \
  tools.prepare.tests.test_qwen35 tools.validation.tests.test_qwen35 \
  tools.validation.tests.test_qwen35_runtime compiler.qwen35.tests.test_model \
  compiler.qwen35.tests.test_text_cache compiler.qwen35.tests.test_export
```

Use each model's environment for its tests: Qwen3.5 model hooks require its
Transformers version. The commands deliberately separate them rather than
discovering all tooling tests in the Qwen3 environment. GPU operator and
kernel-AOTI tests are listed in the model guides; full-model accuracy checks
remain separate commands below.
The `test_export` modules also contain GPU export integration tests, skipped by
the CPU commands above. Run them with CUDA visible and the model's native
attention library built to check the production export path's structural
contracts; they use random model weights and do not compile AOTI packages.

## Checkpoint preparation and tokenization

`python -m tools.prepare` owns checkpoint download, manifest generation,
verification and offline tokenization for both model families. Its family modules
own pinned revisions and required download files; compiler configuration retains
model paths and execution settings. Run download with
the selected model's Python environment before enabling `HF_HUB_OFFLINE` or
`TRANSFORMERS_OFFLINE`:

```bash
"$Q3_PY" -m tools.prepare download --model Qwen/Qwen3-0.6B
"$Q35_PY" -m tools.prepare download --model Qwen/Qwen3.5-0.8B
"$Q35_PY" -m tools.prepare download --model Qwen/Qwen3.5-4B
```

These use fixed revisions and default to `work/models/qwen3-0.6b`,
`work/qwen35/0.8b/text` and `work/qwen35/4b/text`. `--output` selects another
prepared directory; `--cache-dir` selects a Hub cache. Qwen3 downloads the
required HF files and writes their manifest. Qwen3.5 additionally extracts and
renames language-model tensors into `text.safetensors`. Preparation runs on CPU
without CUDA or model inference. An existing output path is refused, and the
prepared directory is published only after successful completion.

Both formats use `model_manifest.json`. Existing prepared directories can be
copied or reused, and their manifest contents and compiled packages remain
unchanged. Checkpoint integrity verification happens during download/preparation
or through explicit `tools.prepare verify`; model loading, export and reference
generation do not repeat it. Verification and token commands infer the model
family from that manifest and use only local files:

```bash
"$Q3_PY" -m tools.prepare verify --model-dir work/models/qwen3-0.6b
"$Q3_PY" -m tools.prepare encode --model-dir work/models/qwen3-0.6b \
  --prompt "What is the capital of France?" --output work/prompt.ids
"$Q3_PY" -m tools.prepare decode --model-dir work/models/qwen3-0.6b \
  --input work/generated.ids
```

Use the same commands with the matching Qwen3.5 environment and prepared text
directory. Tokenization keeps the official non-thinking chat template; encoding
checks the encoded token count against the model's declared context ceiling.
The native package's selected capacity and requested
output budget are checked again during generation. Encoding refuses an existing
output file. Model modules retain checkpoint loading and
execution; the full-model acquisition and token CLIs live in `tools/prepare/`.

## Shared native executables

Both model builds provide `xinfer_aoti` (greedy generation), `xinfer_benchmark`
(fixed-token measurements) and `xinfer_validate` (logit/state traces and runtime
checks). Metadata selects the model ABI; native-path options override each tool's
`build/<model>/` default. The common runner owns loading, state, positions,
prompt chunking and dispatch. The [package contract](package-metadata.md)
defines integrity checks, state ownership and compatibility.

Generation stops at EOS or its output budget and refuses existing output/trace
files and path aliases. Prompt plus `--max-output-tokens` must fit compiled
capacity; prompts are chunked into at most 512-token calls. Benchmark workloads
consume all supplied tokens, including EOS. Validation records every logit and
selected state checkpoints, with coverage described by its case manifest.

With `XINFER_BUILD_VALIDATION=ON`, these native CPU contracts need neither model
packages nor a GPU:

```bash
CUDA_VISIBLE_DEVICES='' build/qwen3/xinfer_state_contract
CUDA_VISIBLE_DEVICES='' build/qwen3/xinfer_generation_contract
```

See [runtime installation](../environments/README.md#install-and-relocate-the-runtime)
for deployment and relocation. Native `load_ms` includes artifact preflight,
weight loading and loader setup; whole-process `process_ms` in benchmark
`runs.json` is the comparable startup measure for older reports that excluded
some integrity checks. First-use and steady-state timings remain separate.

## Python validation tools

One command runs the numerical chain for either family, selected from metadata:

```bash
"$Q35_PY" -m tools.validation.validate \
  --packages artifacts/qwen35/0.8b/text --output artifacts/qwen35/0.8b/validation-new
```

`--packages` and a fresh `--output` are required. Checkpoint and native validator
paths default from package identity; `--model-dir` and `--native` override them.
HF references are generated during the run, without saved-reference preparation.
Qwen3 additionally cross-checks manual HF greedy decoding against `HF.generate()`.

The chain is **HF → adapted/optimized eager → Python AOTI → C++ AOTI**.
Eager includes the model's adapters and explicit graph passes, such as linear
packing and decode projection replacement. Inductor code-generation settings
apply in the compiled stage. All stages receive the same teacher-forced tokens
and partitions; comparisons cover adjacent stages plus a direct AOTI/HF anchor.
Native/Python execution of the same packages must be bitwise.

Teacher-forced state snapshots are sparse, memory-mapped files under each
Python stage's output directory. This retains all selected tensors, including
unused batch rows, without keeping the entire trace in RAM. Qwen3.5 runtime
validation uses the same storage under `python_aoti/`.

Long-context cases retain every logit row and selected state checkpoints:
populated KV prefixes and complete functional states. Untouched future KV is
checked separately. Eager validation selects the bundled prefill attention
implementation; native/package execution uses standard AOTI loading. See
[attention loading](flash-attention.md#build-and-deployment).

For the dynamic batch matrix, use the matching model's `validate_batch` entry:

```bash
"$Q35_PY" -m tools.validation.qwen35.validate_batch \
  --packages artifacts/qwen35/0.8b/text --model-dir work/qwen35/0.8b/text \
  --native build/qwen35/xinfer_validate --output artifacts/qwen35/0.8b/batch-validation
```

Qwen3 uses `tools.validation.qwen3.validate_batch`. Both reuse shared fixtures,
comparison policy and C++ parity handling. Reports cover every B=1..8 and batch
transitions, with independent per-request BF16 HF references and all allocated
state rows included in native parity. Padding, row isolation, inactive/resumed
rows, reset, full capacity and overflow remain required. Qwen3.5 adds the 97/98
DeltaNet dispatch boundary and compares packed eager graphs with AOTI. Numerical
failures remain in per-group reports and produce a nonzero exit after native
checks; a passing native comparison does not resolve them.

The runtime-contract tools separately exercise shipping CLIs and deployment:

```bash
"$Q3_PY" -m tools.validation.qwen3.validate_runtime \
  --packages artifacts/qwen3-flash --output artifacts/qwen3-runtime-new
"$Q35_PY" -m tools.validation.qwen35.validate_runtime \
  --packages artifacts/qwen35/0.8b/text --output artifacts/qwen35/0.8b/runtime-new
```

Qwen3's `--native` selects the generation binary. Qwen3.5's `--binary` and `--cli`
select the validation and generation binaries. Both tools default to matching
model build directories and prepared checkpoints. Inspect complete reports even
when a command exits nonzero; do not assume every failure is a known numerical
issue. Package and native-binary identities accompany runtime/numerical reports.

The shared pipeline lives in `tools/validation/common/`; family `model.py` hooks
supply loading, eager transformations, fixtures and the independent BF16 baseline.
Prompts are shared in `common/prompts.py`. Component/kernel tests are separate
required checks and are not invoked by full-model validation. The model guides
list their coverage and the Qwen3.5 rounding diagnostic.

### Shared numerical policy

Qwen3 and Qwen3.5 use the same full-model policy, recorded in new reports as
`numerical_policy.name: shared_bf16` and `numerical_policy.version: 1`.
Classification depends on the comparison, not the model or fixture origin.

| Check | Acceptance rule |
| --- | --- |
| Matched-input logits: eager/HF, Python AOTI/eager and Python AOTI/HF | Required: both worst-row relative-L2 and peak-relative errors must be at most 5% |
| HF logits replayed on the compiled greedy continuation | Required: the same logit bounds |
| Full-model KV, convolution and recurrent-state numerical differences across HF/eager/AOTI | Diagnostic, with all measured errors retained |
| HF-relative teacher top-token and independent greedy-sequence differences | Diagnostic for every fixture |
| C++/Python execution of the same AOTI packages | Required: bitwise logits and selected states, exact generated tokens and positions |
| Shapes, finite values, state ownership, cache write regions/masking, reset/reuse and overflow | Required, independently of numerical differences |

For each comparison row, relative L2 is `norm(actual-reference) / norm(reference)`;
peak-relative error is `max(abs(actual-reference)) / max(abs(reference))`.
Both denominators have a `1e-12` floor, and the limit is inclusive. These are
row-normalized metrics, not an elementwise 5% tolerance. `max_abs` is reported
without a separate full-model acceptance threshold. State row definitions remain
model-specific. The existing 5% bounds are unchanged.

Teacher top-token records retain reference margins and selected-token gaps;
greedy records retain the first divergence and both sequences. Small logit
changes can change argmax, while independent generation subsequently follows
different inputs. The HF replay keeps logit acceptance on matching inputs.
Diagnostic state/token differences remain evidence to investigate; a numerical
pass alone does not establish equivalent task quality or deployment readiness.

Reports retain canonical stages, all tensor `comparisons`, required `failures`
and separate `diagnostics`. The diagnostic list contains state rows exceeding
the existing 5% threshold and non-native token/greedy mismatch records; all state
metrics remain in `comparisons`, including those within the threshold.
Token/greedy mismatch records are also indexed in `top1_differences` and
`greedy_differences`; these are additional views, so count each record once.
Structural violations abort validation and are recorded as errors. These checks
are outside the tensor-comparison count.

Full-model validation retains the unmodified BF16 HF reference and does not load
a separate FP32 HF control model. Focused kernel/block tests run separately
and keep their own output and carried-state tolerances. DeltaNet comparisons
use matching input dtypes: BF16 kernels against HF's BF16 tensor operations,
including their algorithm-defined FP32 accumulation and recurrent state.
FP32 DeltaNet support tests use FP32 references; BF16 acceptance does not require
an upcast FP32 control. Historical reports retain their original policies,
failures and labels. Earlier unified Qwen3.5 reports required state bounds and
end-to-end token agreement. Changing their classification does not fix the numerical
differences or establish a fresh validation result.
Performance experiments may separately record a predeclared baseline-relative
continuation criterion; it does not replace the project validation result or
task-quality acceptance.

Model validation modules require Python assertions. They refuse `-O`, `-OO` and
`PYTHONOPTIMIZE` before loading models or creating reports, because their graph,
state and runtime checks use assertions. Shared numerical/file comparisons and
subprocess status/error checks use explicit validation and remain active in
optimized interpreters. Existing reports are unchanged; rerun validation if a
report was produced with assertions disabled.

## Benchmark HF and native AOTI

`python -m tools.benchmark` selects the model from package metadata. Both families
support `run`, `compare` and `check-idle`. The harness prepares its own fixed
capital-of-France prompt/continuation tokens and does not consume validation
reports. Use the matching model environment:

```bash
"$Q3_PY" -m tools.benchmark run \
  --packages artifacts/qwen3-flash --output artifacts/benchmark-new
```

`run` compares unchanged HF with its dynamic cache (`hf`) against one compiled
pair (`cpp_aoti`). `--model-dir` and `--native` override the inferred checkpoint
and `build/<model>/xinfer_benchmark`. Each worker receives identical prompts and
forced decode tokens, so differing predictions do not shorten the workload.
HF reset discards its cache and first prefill includes allocation. HF receives
the whole prompt in one call; native prefill includes every compiled chunk.
`prefill_execution` records that difference. Numerical validation instead matches
partitions. Earlier masked-cache or different-workload speedups need fresh
comparisons; compare results within the same model and target.

Defaults are prompt lengths 25/128/512, 64 forced decode steps, two warmups,
five measured requests and three process rounds. Override with `--lengths`,
`--decode-steps`, `--warmup`, `--repeats` and `--rounds`. Prompt plus forced decode
tokens must fit package capacity. Use lengths 511/512/513/8191/8192 when measuring
chunk boundaries and long prompts, after checking available device memory.

All commands honor `CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER` by default.
`--gpu` selects an explicit physical `nvidia-smi` index or GPU UUID. Workers,
idle checks and telemetry use that same UUID as logical CUDA device zero.
The standalone idle check needs no model or packages:

```bash
"$Q3_PY" -m tools.benchmark check-idle --output artifacts/idle.json
```

It samples GPU/memory activity once per second, requiring five consecutive
readings at or below 10% and failing after roughly 30 seconds without that window.
Workers also check idle conditions before and after execution. `--allow-busy`
permits functional harness checks and marks timings provisional. Idle checks
alone do not prove healthy clocks or exclusive access; inspect telemetry and
repeat controlled measurements before making performance claims.

Reports use summary schema 1 and `model_type`; HF profiles record
`eager_baseline: hf_dynamic_cache`. `environment.json` records GPU identity and
package, binary and workload checksums. Summaries include reset, full-prompt
prefill, reset-plus-prefill (`first_prediction_ms`), decode/per-token and whole-
request distributions. Loading and first use are separate from warmed results.
Repeated predictions, counts, consumed positions and thread settings are checked;
`prediction_agreement` between different profiles is diagnostic.

Memory reports distinguish allocator peaks from sampled total-device use and
its delta from process startup. Device samples may include other GPU users and
miss transient allocations. Do not add allocator and device totals together.
Detailed environment capture, rotated process order and worker orchestration
are shared under `tools/benchmark/`, with small model hooks.

## Compare two compiled profiles

```bash
"$Q3_PY" -m tools.benchmark compare \
  --baseline-packages artifacts/aoti-baseline \
  --candidate-packages artifacts/aoti-candidate \
  --baseline-label baseline --candidate-label candidate \
  --output artifacts/comparison-new
```

Both pairs need compatible checkpoints, toolchains, tensor/numerical contracts
and logical weight identities/source mappings. General tuning options remain in
export logs. `--native` selects another common binary; labels default to
`baseline` and `candidate`. Four process rounds alternate execution order by
default, with the same warmup/repeat settings as `run`.

Inspect per-workload/per-round results as well as pooled distributions. Use an
identical-bundle A/A comparison when needed to characterize observed variation;
it is not a universal noise bound. Separate profiler timing from unprofiled
native measurements. Run accuracy validation independently on the same bundles.
