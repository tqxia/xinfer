# Development tools

Run these commands from the repository root with the matching Python/LibTorch
environment. Full-model validation and benchmarks require exported packages;
use fresh output directories.
Select `Q3_PY`, `Q35_PY` and `CMAKE` with the
[target environment guide](../environments/README.md), and set
`BUILD="$PWD/build/qwen3"` for the Qwen3 commands below. Each model's Python
tools default to executables and the attention library in `build/<model>/`;
existing path options and `XINFER_ATTENTION_OPS_LIBRARY` override those defaults.
Shared logged subprocess execution and native-library discovery live in
`tools/common.py`.

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
git ls-files -z '*.cpp' '*.h' '*.cu' '*.cuh' | xargs -0 work/format-venv/bin/clang-format -i
```

Check formatting without changing files:

```bash
git ls-files -z '*.py' | xargs -0 work/format-venv/bin/ruff format --check
git ls-files -z '*.cpp' '*.h' '*.cu' '*.cuh' | xargs -0 work/format-venv/bin/clang-format --dry-run --Werror
```

## CPU unit tests

Tests follow the code they cover. Preparation, benchmark and validation tooling
tests live under `tools/<area>/tests/`. Compiler metadata/export, model loading,
adapters and kernel tests remain under `compiler/*/tests/`; environment tests
remain under `environments/tests/`.

These CPU checks use temporary fixtures and mocks, without checkpoint
downloads, compiled packages, native binaries or a GPU:

```bash
CUDA_VISIBLE_DEVICES='' "$Q3_PY" -m unittest \
  compiler.common.tests.test_context compiler.common.tests.test_metadata \
  compiler.qwen3.tests.test_packages compiler.qwen3.tests.test_export \
  tools.tests.test_common \
  tools.prepare.tests.test_common tools.prepare.tests.test_qwen3 \
  tools.benchmark.tests.test_benchmark \
  tools.validation.tests.test_common tools.validation.tests.test_qwen3_runtime \
  tools.validation.tests.test_qwen3 tools.validation.tests.test_validation \
  tools.validation.tests.test_validation_chain \
  tools.validation.tests.test_validation_runtime \
  tools.validation.tests.test_validation_native \
  tools.validation.tests.test_validation_modes tools.validation.tests.test_context \
  tools.validation.tests.test_hybrid
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

Both models build the same `xinfer_aoti`, `xinfer_benchmark` and `xinfer_validate`
targets in separate `build/qwen3/` and `build/qwen35/` directories. Each executable
reads the package metadata to select the model-specific contract. Both entry
JSON files require the shared
[`schema_version: 5` format](package-metadata.md), `runtime_abi_version: 1`, and an
explicit `model_type` of `qwen3` or `qwen35`. Schema 1 through schema 4 and
unversioned sidecars must be converted or re-exported; model selection reads the versioned
metadata directly.

The common `Runner<Spec>` owns package loaders, shared weights, device checks,
token validation, positions, prompt chunking and prefill/decode dispatch.
New exports choose `cache_length` with `--max-context-length`, defaulting to the
model's declared ceiling. Compiled prefill remains `token_range: [2,512]`.
The per-request `--max-output-tokens` budget shares context with the prompt;
there is no separate prompt-cap option.
A long request is split into calls of at most
512 tokens, with a final single token routed to decode. Request/ID/budget checks
cover the entire prompt before its first chunk executes. Ordinary L20 exports,
C++ execution and full 16,384-slot inference-memory probes completed for
Qwen3-0.6B, Qwen3.5-0.8B and Qwen3.5-4B, with bitwise native/Python AOTI
agreement. Qwen3 passed its existing HF numerical gates; Qwen3.5 HF acceptance
remains open. The new profile has not been validated on GB10. The common state
policy uses metadata update rules and ABI-derived output positions to retain KV buffers
and replace functional conv/recurrent outputs. Reset reuses KV storage and zeros
replacement state. Returned tensors are checked for shape, dtype, device,
contiguity and required KV aliases before replacement state is adopted. The
generation CLI refuses existing output/trace files and path aliases. Native benchmarks use
the same report header and timing boundaries for both models. The validation
binary records every logit and selected state checkpoints. Expanded-capacity
checks store the populated KV prefix and complete functional state; untouched
future KV is checked separately. The case manifest records checkpoint coverage.

With `XINFER_BUILD_VALIDATION=ON`, two targets check state ownership and complete
prompt partitioning/generation on CPU without model packages or a GPU:

```bash
CUDA_VISIBLE_DEVICES= build/qwen3/xinfer_state_contract
CUDA_VISIBLE_DEVICES= build/qwen3/xinfer_generation_contract
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

## Python validation tools

One metadata-selected command runs numerical validation for either model:

```bash
"$Q35_PY" -m tools.validation.validate \
  --packages artifacts/qwen35/0.8b/text --output artifacts/qwen35/0.8b/validation
```

`--packages` and a fresh `--output` are required. The model directory and native
`xinfer_validate` path default from package identity; existing `--model-dir` and
`--native` overrides select another checkpoint/build. Run from the repository
root with the matching model environment. Both models generate their HF references
during validation; neither needs a separate preparation command or saved-reference
input.

The shared chain is HF → eager → Python AOTI → C++ AOTI. The eager stage
applies adapters and explicit graph passes such as Qwen3.5 linear packing,
following the package metadata. Teacher forcing uses identical tokens and
partitions at every stage. Adjacent numerical comparisons distinguish the
combined eager transformations from compilation; a direct Python-AOTI/HF
comparison retains the end-result bound. Native/Python AOTI comparison is
bitwise. Long-context cases retain identical chunk partitions across the HF,
adapted/packed eager and AOTI stages, with the independent BF16 HF baseline and
selected state checkpoints. Runtime parity and open HF numerical acceptance
remain separate conclusions; existing HF limits are unchanged. Inductor code-generation options apply in the compiled stage.

`tools/validation/common/` owns shared execution, comparisons and native trace
handling. Each model's `model.py` supplies loading, the eager transformations,
fixtures and the independent BF16 HF reference. Shared prompt definitions in
`tools/validation/common/prompts.py` retain each model's texts and case names;
Qwen3.5 numerical and runtime-contract checks also reuse its long-prompt text.
Each HF stage generates its chat continuations and logits during the run, then
fixes identical teacher-forced inputs for later stages. Qwen3 also cross-checks
its manual HF greedy decoding against HF `generate()` before installing adapters.
The chain and runtime validator share the Python AOTI loader and explicit-state
runtime, including package preflight and ownership of shared weights.
`qwen3/validate_runtime.py` and `qwen35/validate_runtime.py` retain separate
CLI/deployment contract checks and tokenize their prompts from the local
checkpoint. These replace the former separate Python-AOTI,
model and end-to-end numerical entry points. Validation uses the shared
`compiler.common.attention.load_attention_ops` loader after checking package
metadata and resolving the attention library linked by the selected native
executable. A missing linked library is rejected before operator loading.
Validation tooling tests live in
`tools/validation/tests/`; compiler/model/operator tests remain under
`compiler/*/tests/`. DeltaNet numerical and behavior checks live in
`compiler/qwen35/tests/test_kernels.py`, with reusable fixtures and references
beside the tests. `compiler/qwen35/tests/test_kernels_aoti.py`
checks export, decomposition and exact direct-kernel/Python AOTI output/state
parity for both geometries and BF16/FP32. It compiles packages in temporary
directories, reuses them across test methods and skips when CUDA is unavailable.
There is no standalone kernel-validation CLI or kernel C++ runner; full-model
validation retains native coverage. Production compiler and runtime code stays
in its existing directories.

The retained Qwen3.5 rounding diagnostic runs with
`python -m tools.diagnostics.qwen35.probe_rounding` in the Qwen3.5 environment.
It uses the prepared text checkpoint to reproduce Q35-NUM-001; exit zero means
the issue was reproduced.

Shared attention operator checks and Qwen3 adapter checks run as unit tests in
`compiler/common/tests/test_attention.py` and `compiler/qwen3/tests/test_attention.py`.
The adapter tests construct an HF attention module with random weights and
require no checkpoint download. The model validation tools check exported
attention packages through Python AOTI and C++; there are no separate Qwen3
operator or adapter validation CLIs.

Shared tensor measurements live in `compiler/common/tests/tensors.py` and are
reused by `tools/validation/common/validation.py`; report and native-file helpers
remain in the validation tooling. Shared metric names are `max_abs`,
`relative_l2` and `peak_relative`.
Unified reports record canonical stage names, per-stage comparisons,
`numerical_policy`, required `failures` and separate `diagnostics`. The shared
full-model relative-L2 and peak-normalized limits remain 5%; focused kernel
unit tests retain their own limits. Qwen3 requires bounded logits and reports
HF cache/token/greedy drift. Qwen3.5 also requires state bounds; its original
end-to-end fixtures retain required Python-AOTI/HF top-token and greedy-token
agreement, while model-only token comparisons remain diagnostic. Full-model
validation does not load a separate FP32 HF control model. Standalone kernel
FP32 checks are unchanged. Native/Python AOTI comparisons remain
bitwise. Structural cache/write, poisoning, reset/reuse and overflow checks are
required independently of numerical differences. Historical reports retain
their original layouts and labels.

Optional MegaQwen correctness checks use `tools.validation.hybrid` with the same
live BF16 HF chain and numerical limits. Enable `XINFER_ENABLE_MEGAQWEN` and
`XINFER_BUILD_VALIDATION` in the Qwen3 native build to add
`qwen_attention_contract` and `qwen_hybrid_contract`. The former checks the actual
device attention helper against an independent dense reference; the latter
checks shared weights/KV, AOTI handoffs, positions, reset/reuse and stream
lifetime. The Python hybrid command runs that native contract helper as well
as the matched-input numerical chain. Its correctness timings are not benchmark
results. See [current build and validation commands](../compiler/qwen3/README.md#optional-megaqwen-decode).

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
decode, per-token decode and whole-request distributions. Prefill timing
covers the complete native prompt, including every compiled chunk and the final
prediction, rather than only the final chunk. Memory records include
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
HF receives the whole prompt in one call, whereas native AOTI uses chunks bounded
by the package prefill range. `prefill_execution` in the summary records this
partition difference; the benchmark is not a matched-partition numerical test.

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
override these counts. The package's `cache_length` bounds complete request
context, including both prompt and forced output tokens; `token_range`
bounds each compiled call. Use
`--lengths 511 512 513 8191 8192` to measure chunk boundaries and the complete
long prompt. Recheck idle conditions and available device memory before running
long-context workloads.

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
"$Q3_PY" -m tools.validation.validate \
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
agreement remain strict. The independent BF16 HF eager baseline is retained.
This policy does not change model arithmetic or make HF numerical equivalence a
claim of the compiled profile. See [FlashAttention](flash-attention.md) for the
recorded differences and supported workload.
