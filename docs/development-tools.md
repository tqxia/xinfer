# Benchmark comparisons and numerical diagnostics

Run these commands from the repository root with the matching Python/LibTorch
environment and previously exported packages. Use fresh output directories.

## Shared native executables

Both models use `xinfer_aoti`, `xinfer_benchmark` and `xinfer_validate` from one
native build. Each reads the package metadata to select the model-specific
contract and state policy. Both entry JSON files require the shared
[`schema_version: 1` format](package-metadata.md), with an explicit `model_type`
of `qwen3` or `qwen35`. Unversioned sidecars must be converted or re-exported;
model selection reads the versioned metadata directly.

The common runner owns package loaders, shared weights, device checks, token
validation, positions and prefill/decode dispatch. Model policies handle KV
buffer ownership and Qwen3.5's functional conv/recurrent outputs. The generation
CLI refuses existing output/trace files and path aliases. Native benchmarks use
the same report header and timing boundaries for both models. The validation
binary retains the existing model-specific trace layouts so current Python
numerical checks and historical output comparisons remain usable.

The [common runtime record](experiments/common-runtime.md) describes the runner,
state-policy boundaries, compatibility behavior and validation evidence.

## Shared Python benchmark and validation support

Both Python benchmark entry points use `compiler/common/benchmark.py` for
environment and package records, rotated process order, idle checks,
worker execution, failure reports and summaries. Model modules provide prompts,
checkpoint manifests, package checks and eager reset/step hooks. Qwen3 retains
paired compiled-profile comparisons; Qwen3.5 attaches its matching end-to-end
validation result.

Benchmark `summary.json` uses `schema_version: 1` and `model_type`. Its `profiles`
retain the CLI's profile labels; each case stores prediction agreement as
`prediction_agreement: {reference_profile: matching_count}`. Environment records
use `packages[profile][entry_point]` for metadata, size and checksum, and
`shared_weights[profile]` for the archive identity checks.

Benchmark environment records include package, binary and workload checksums.
Qwen3.5 runtime and end-to-end validation reports retain package and native
binary checksums.

Both summaries include reset, prefill, reset-plus-prefill (`first_prediction_ms`),
decode, per-token decode and whole-request distributions. Memory records include
the sampled device-memory peak and its delta from the process's initial reading,
as well as PyTorch allocator peaks. Device-memory samples can include other GPU
users. Repeated predictions, their counts, consumed positions and single-threaded
workers are checked before aggregation. Prediction agreement between profiles
remains diagnostic.

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

## Compare HF eager with compiled Qwen3

`run` compares the unmodified HF eager reference with one FlashAttention AOTI
package pair. The compiled profile is named `aoti`; there is no separate
eager-numerics compilation profile. Each process receives the same prompts and
forced decode tokens, so differences in predicted tokens do not change the
measured workload.

```bash
cmake -S . -B build -DQWEN_BUILD_BENCHMARK=ON
cmake --build build --target xinfer_benchmark -j2
.venv/bin/python -m compiler.qwen3.benchmark run \
  --packages artifacts/aoti \
  --native build/xinfer_benchmark \
  --output artifacts/benchmark-new
```

The Qwen3 build always links `libxinfer_attention_ops.so`. Deploy that library
with the native executable and matching LibTorch. Previously exported efficient
packages must be re-exported using the current exporter.

Both `run` and `compare` default to CUDA logical device 0 under the current
`CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER` settings. To select a physical GPU
explicitly, pass an `nvidia-smi` index or GPU UUID with `--gpu`; for example,
`--gpu 1` overrides the inherited visibility mask. Every worker uses that GPU as
CUDA device 0. Idle checks, telemetry and recorded GPU information use the same
UUID, stored as `gpu_uuid` in `environment.json`.

The [Qwen3.5 `run` and `check-idle` commands](../compiler/qwen35/README.md#benchmark)
follow the same GPU selection policy. Qwen3.5 full-model exports also use the
shared `libxinfer_attention_ops.so` for prefix FlashAttention; build it before
export and deploy it with the native executable. Re-export older Qwen3.5
efficient-attention packages for the current caller-owned KV contract.

## Compare two compiled profiles

`compiler/qwen3/benchmark.py compare` runs two package directories through the same
native benchmark executable. Each directory contains prefill and decode packages.
Both profiles receive identical prompts and forced decode tokens. Four default
process rounds alternate baseline/candidate order, with two warmups and five
measured requests per prompt length in each process. Loading and first use are
reported separately from steady-state latency.

```bash
.venv/bin/python -m compiler.qwen3.benchmark compare \
  --baseline-packages artifacts/aoti-baseline \
  --candidate-packages artifacts/aoti-candidate \
  --baseline-label baseline --candidate-label candidate \
  --native build/xinfer_benchmark \
  --output artifacts/comparison-new
```

Both package pairs must use the fixed FlashAttention operator and extent-input
ABI, with matching checkpoint, toolchain, tensor contracts and numerical
compilation settings. Each pair's prefill/decode metadata is checked as well.
Labels name the measurements; they do not select another execution backend.
`--reference` selects the directory containing the saved capital prompt and
continuation; it defaults to `artifacts/reference`.

Results include per-request timings, latency distributions, predicted-token
agreement, memory samples, GPU telemetry and package/binary/workload hashes.
Prediction agreement is diagnostic; use the correctness validator for numerical
acceptance. GPU-idle checks run before and after each worker. Use
`--allow-busy` only for harness checks; its timings are marked provisional.

## Validate runtime correctness and report HF differences

```bash
.venv/bin/python -m compiler.qwen3.tests.validate_cpp \
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
