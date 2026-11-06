# Development tools

Run these commands from the repository root with the matching Python/LibTorch
environment and previously exported packages. Use fresh output directories.
Select `Q3_PY`, `Q35_PY` and `CMAKE` with the
[target environment guide](../environments/README.md), and set
`BUILD="$PWD/build/qwen3"` for the Qwen3 commands below. Each model's Python
tools default to executables and the attention library in `build/<model>/`;
existing path options and `XINFER_ATTENTION_OPS_LIBRARY` override those defaults.

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

## CPU benchmark tests

Benchmark unit tests live with the tooling and use temporary fixtures and mocks,
without checkpoint downloads, compiled packages, native binaries or a GPU:

```bash
CUDA_VISIBLE_DEVICES='' "$Q3_PY" -m unittest tools.benchmark.tests.test_benchmark
```

Use the matching model environments for preparation and model tests. Numerical
validation remains in the model-specific compiler test modules below.

## Checkpoint preparation

Use `python -m tools.prepare` for pinned checkpoint download, explicit offline
verification and token encoding/decoding for either model family. Preparation
publishes a new directory only after its files have been verified. Both use
`model_manifest.json`; retained Qwen3.5 text checkpoints can rename their former
`manifest.json` without changing its contents or recompiling packages.
Preparation tests live in `tools/prepare/tests`; checkpoint loading tests remain
with their compiler model. See the model guides for complete commands.

## Shared native executables

Both models build the same `xinfer_aoti`, `xinfer_benchmark` and `xinfer_validate`
targets in separate `build/qwen3/` and `build/qwen35/` directories. Each executable
reads the package metadata to select the model-specific contract. Both entry
JSON files require the shared
[`schema_version: 3` format](package-metadata.md), `runtime_abi_version: 1`, and an
explicit `model_type` of `qwen3` or `qwen35`. Schema 1, schema 2 and unversioned
sidecars must be converted or re-exported; model selection reads the versioned
metadata directly.

The common `Runner<Spec>` owns package loaders, shared weights, device checks,
token validation, positions and prefill/decode dispatch. Its common state policy
uses metadata update rules and ABI-derived output positions to retain KV buffers
and replace functional conv/recurrent outputs. Reset reuses KV storage and zeros
replacement state. Returned tensors are checked for shape, dtype, device,
contiguity and required KV aliases before replacement state is adopted. The
generation CLI refuses existing output/trace files and path aliases. Native benchmarks use
the same report header and timing boundaries for both models. The validation
binary retains the existing model-specific trace layouts so current Python
numerical checks and historical output comparisons remain usable.

With `XINFER_BUILD_VALIDATION=ON`, the `xinfer_state_contract` target checks the
common state policy on CPU without model packages or a GPU:

```bash
CUDA_VISIBLE_DEVICES= build/qwen3/xinfer_state_contract
```

`Spec` validates metadata and the model ABI. Each native runtime construction
performs fresh file preflight: it hashes prefill and decode once each against the
sidecars and embedded weight manifest, hashes the weight archive once, and
parses its manifest once. All integrity checks complete before GPU weight
allocation. These checks repeat for every runtime construction.

Native `load_ms` includes file preflight, weight loading and loader setup.
Earlier reports excluded the file checks performed during `Spec` construction.
Use whole-process elapsed time (`process_ms` in benchmark `runs.json`) to compare
total startup across that boundary change; first-request and steady-state
timings remain separate.

The [common runtime record](experiments/common-runtime.md) preserves the initial
runner refactor, its former model-specific state policies and validation evidence.
See [runtime installation](../environments/README.md#install-and-relocate-the-runtime)
for the CMake `Runtime` component, installed layout and prefix relocation.

## Python validation support

Numerical helpers live in `compiler/common/tests/validation.py`, alongside their
tests. Shared metric names are `max_abs`, `relative_l2` and `peak_relative`.
Full-model reports use `numerical_policy.limits` and a `failures` array for required
bounded-error comparisons. These replace the older Qwen3 `max_relative_l2`,
`max_peak_relative` and `numerical_failures` names and Qwen3.5's standalone `limit`.
Model validators retain tensor grouping and acceptance rules: Qwen3 bounds HF
logit errors and reports HF cache/token differences separately, while Qwen3.5's
end-to-end gate also bounds state errors and requires HF top-1 and greedy-token
agreement. The shared full-model
relative-error limit is 5%; focused kernel validators retain their own limits.
Native/Python AOTI comparisons continue to require bitwise equality.

Model validation modules require Python assertions. They refuse `-O`, `-OO` and
`PYTHONOPTIMIZE` before loading models or creating reports, because their graph,
state and runtime checks use assertions. Shared numerical/file comparisons and
subprocess status/error checks use explicit validation and remain active in
optimized interpreters. Existing reports are unchanged; rerun validation if a
report was produced with assertions disabled.

## Benchmark HF and native AOTI

The shared benchmark tool lives under `tools/benchmark/`. Its common module
owns environment and package records, rotated process order, GPU idle checks,
worker execution, measurements, failure reports and summaries. Small model hooks
provide checkpoint manifests, package checks, tokenization and HF model loading.
`python -m tools.benchmark` selects these hooks from package metadata and offers
`run`, `compare` and `check-idle` for both families. Workers use the common plain-HF
cached-forward helper. Benchmarking prepares its own fixed-token workload and
uses no validation reports or saved validation fixtures.

Benchmark `summary.json` uses `schema_version: 1` and `model_type`. `run` names its
profiles `hf` and `cpp_aoti`; `compare` uses the selected baseline/candidate labels.
Eager worker reports and summaries containing an eager profile record
`eager_baseline: hf_dynamic_cache`; each case stores
`prediction_agreement: {reference_profile: matching_count}`. Environment records
use `packages[profile][entry_point]` for metadata, size and checksum, and
`shared_weights[profile]` for archive identity checks. Package, binary and
workload checksums accompany the measurements. Accuracy reports remain in the
validation tools.

Benchmark summaries include reset, prefill, reset-plus-prefill (`first_prediction_ms`),
decode, per-token decode and whole-request distributions. Memory records include
the sampled device-memory peak and its delta from the process's initial reading,
as well as PyTorch allocator peaks. Device-memory samples can include other GPU
users. Repeated predictions, their counts, consumed positions and single-threaded
workers are checked before aggregation. Prediction agreement between profiles
remains diagnostic.

One command serves both model families, selected from package metadata. Run it
with the matching model's Python environment. `run` compares the unmodified HF
eager profile (`hf`) with a compiled package pair (`cpp_aoti`). Each process
receives identical prompts and forced decode tokens, so differing predictions
do not change the measured workload. Both models start with
`past_key_values=None`, then carry the cache returned by the unchanged HF model.
HF creates its dynamic cache and determines masks and positions. Normal PyTorch
SDPA backend selection remains enabled. Reset discards the cache; the first
prefill includes cache allocation. Initial one-token requests use the same path.

The benchmark tokenizes its own capital-of-France chat prompt and the fixed
"Paris is the capital of France. " continuation. It needs no validation fixtures
or reports. Earlier Qwen3 workloads used a saved generated continuation;
workload hashes identify that protocol change. Earlier masked-cache speedups
also need fresh measurements with the HF dynamic-cache baseline. Model-specific
Transformers versions and computation still differ, so compare measurements
within a model and target.

```bash
"$CMAKE" --build "$BUILD" --target xinfer_benchmark -j2
"$Q3_PY" -m tools.benchmark run \
  --packages artifacts/aoti \
  --native "$BUILD/xinfer_benchmark" \
  --output artifacts/benchmark-new
```

`--packages` and a fresh `--output` are required. The prepared model directory
and native executable default from package identity; `--model-dir` and `--native`
select another local checkpoint path or build. The native defaults are
`build/qwen3/xinfer_benchmark` and `build/qwen35/xinfer_benchmark`.
Deploy the shared attention library with the native executable and matching
LibTorch, following the runtime installation guide.

Defaults are prompt lengths 25, 128 and 512, 64 fixed decode steps, two warmups,
five measured requests per case and three process rounds. `compare` defaults to
four rounds. `--lengths`, `--decode-steps`, `--warmup`, `--repeats` and `--rounds`
override these counts. The package's `token_range` and `cache_length` define legal
prompt and context limits.

`run`, `compare` and `check-idle` default to CUDA logical device 0 under the current
`CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER` settings. `--gpu` accepts an
`nvidia-smi` physical index or GPU UUID and overrides inherited visibility.
Every worker uses that GPU as CUDA device 0. Idle checks, telemetry and recorded
GPU information use the same UUID, stored as `gpu_uuid` in `environment.json`.

The standalone idle check needs no model or packages:

```bash
"$Q3_PY" -m tools.benchmark check-idle --output artifacts/idle.json
```

It records GPU and memory activity once per second, accepting five consecutive
readings at or below 10%, and fails after roughly 30 seconds without that quiet
window. Every benchmark worker also has idle checks before and after execution.
`--allow-busy` permits functional harness runs and marks their timings
provisional.

## Compare two compiled profiles

`python -m tools.benchmark compare` supports both models. It runs two package
directories through the same native executable with identical fixed-token
workloads. The default four process rounds alternate baseline/candidate order,
with two warmups and five measured requests per prompt length. Loading and first
use are recorded separately from steady-state latency.

```bash
"$Q3_PY" -m tools.benchmark compare \
  --baseline-packages artifacts/aoti-baseline \
  --candidate-packages artifacts/aoti-candidate \
  --baseline-label baseline --candidate-label candidate \
  --native "$BUILD/xinfer_benchmark" \
  --output artifacts/comparison-new
```

Both pairs require compatible checkpoint, toolchain, tensor contracts and
numerical compilation settings, as well as matching logical weight identities
and source mappings. General compiler tuning options remain in export logs.
Labels name the measurements and default to `baseline` and `candidate`.

Results include request timings, latency distributions, prediction agreement,
memory samples, GPU telemetry and package/binary/workload hashes. Numerical
accuracy is checked separately with the validation tooling.

## Validate runtime correctness and report HF differences

```bash
"$Q3_PY" -m compiler.qwen3.tests.validate_cpp \
  --packages artifacts/aoti-candidate \
  --output artifacts/validation-candidate-new
```

The default validation report records HF cache errors and differing token IDs,
including top-two margins and the HF score gap to the compiled-selected token.
Those differences are accepted diagnostic results for the fused Flash profile;
the report distinguishes HF parity from runtime correctness. Logit differences
must still satisfy the existing 5% relative-L2 and peak-normalized limits.

C++/Python AOTI bitwise agreement, finite-value checks, cache writes and masking,
reset/reuse, stable storage, boundary rejection and native/Python generation
agreement remain strict. Independent HF eager and FP32 controls are retained.
This policy does not change model arithmetic or make HF numerical equivalence a
claim of the compiled profile. See [FlashAttention](flash-attention.md) for the
recorded differences and supported workload.
