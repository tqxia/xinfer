# Qwen3-0.6B AOTI / C++

Qwen3 has one supported compiled profile: fused BF16 AOTInductor with valid-prefix
FlashAttention and native GQA. The original HF model supplies the weights and
an independent BF16 eager reference. The runtime performs inference and generation
in C++ without Python.

The native executables are shared with Qwen3.5: `xinfer_aoti` for generation,
`xinfer_benchmark` for fixed-token timing and `xinfer_validate` for native
validation. Package metadata selects the model; model-specific checks and KV
state handling remain in `runtime/qwen3/runtime.h`.

The former eager-numerics compilation profile and efficient-attention export
backend are retired. Existing efficient packages must be re-exported. Known HF
cache/token differences are documented in [numerical history](../../docs/experiments/qwen3-numerics.md);
[the attention guide](../../docs/flash-attention.md) describes the fixed operator
and package ABI. Qwen3.5 retains its separate adapter, environment and policy.

## Environment

L20 and GB10 use the same compiler, attention adapter and C++ runtime. Start with
[the target environment guide](../../environments/README.md) to select the native
Torch/Triton/CUDA stack, create the isolated model environment, and build/export
with the `build/qwen3/` native directory and target-specific packages.
Export-time Torch and native LibTorch must come from the same selected installation.
The guide uses `environments/<target>/setup.sh` to install a fresh model venv
and `environments/check.sh` to check it. Setup is implemented in Bash; the
checker and environment tests also live under `environments/`.

This model's self-contained requirements files are
[`environments/l20/qwen3_requirements.txt`](../../environments/l20/qwen3_requirements.txt)
and [`environments/gb10/qwen3_requirements.txt`](../../environments/gb10/qwen3_requirements.txt).
Each specifies the PyTorch index and all Torch, Triton, build-tool and HF pins,
including Transformers 4.51.3. Setup and checking read the same selected file;
its Torch version can be changed independently of Qwen3.5.
The Flash adapter enforces the Transformers version before importing HF private
implementation code or modifying the model. Target profiles contain hardware checks.

Use the tested Torch 2.13 target requirements for new shared-weight exports. Torch
2.7 and packages with embedded constants are unsupported by the current runtime;
keep historical packages with their original binaries for rollback. Examples
below use the L20 paths; select the corresponding paths from the target guide
for GB10.

```bash
Q3_PY="$PWD/work/l20/qwen3-torch213-venv/bin/python"
BUILD="$PWD/build/qwen3"
```

## Prepare checkpoint and offline inputs

Run commands from the repository root with the matching model environment.
The shared preparation tool downloads the pinned Qwen3 checkpoint and writes
`model_manifest.json` with file sizes and SHA-256 checksums:

```bash
Q3_MODEL="$PWD/work/models/qwen3-0.6b"
"$Q3_PY" -m tools.prepare download --model Qwen/Qwen3-0.6B --output "$Q3_MODEL"
```

The output defaults to `work/models/qwen3-0.6b`; `--output` selects another path.
The checkpoint revision is `c1899de289a04d12100db370d81485cdf75e47ca`.
Download needs no CUDA or model inference. It refuses an existing output path
and publishes the prepared directory only after every required file is ready.
`--cache-dir` selects a Hub cache; otherwise the Hub's configured cache is used.
Run download before enabling offline mode.

A complete verified checkpoint can instead be copied or symlinked into the
default directory. Existing checkpoints and manifests remain usable without
rewriting or recompiling packages. Export, benchmark and validation commands
accept `--model-dir` to select another local directory.

Download performs checkpoint integrity checks. `tools.prepare verify` provides
an explicit check for copied or retained directories. Model loading, export and
validation do not repeat checkpoint file verification. Verification and
tokenization infer the model family from the prepared manifest and use only local
files. Token output must be a new file.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
# Optional explicit check for copied or retained checkpoints.
"$Q3_PY" -m tools.prepare verify --model-dir "$Q3_MODEL"
"$Q3_PY" -m tools.prepare encode --model-dir "$Q3_MODEL" \
  --prompt 'What is the capital of France? Answer in one short sentence.' \
  --output work/prompt.ids
```

Validation generates HF references during the run from the factual, arithmetic
and Chinese prompts in `tools/validation/common/prompts.py`. It needs no separate
reference-preparation command or saved reference directory. Thinking is disabled,
generation is greedy and both checkpoint EOS IDs are honored. The HF stage also
checks its manual greedy decoding against HF `generate()` before the Flash
adapter is installed.

## Build before export

The Qwen3 build always produces and links `libxinfer_attention_ops.so`; there is
no attention build switch. Both Python export and native inference require this
library and matching LibTorch/CUDA. The first CMake configuration fetches the
pinned nlohmann/json headers; later builds reuse the cached dependency.

Follow the target guide's native build command, including its CUDA runtime
selection on L20. Python tools default to the native executables and attention
library in the repository's `build/qwen3/`. Existing `--native` options select
another executable. To override the Python attention library:

```bash
export XINFER_ATTENTION_OPS_LIBRARY="$BUILD/libxinfer_attention_ops.so"
```

The library override is optional; the default is
`build/qwen3/libxinfer_attention_ops.so`. Native binaries link the library directly
and must be deployed with it and their matching LibTorch/CUDA libraries. A `.pt2`
package does not bundle this external operator. Native linkage must remain free
of `libpython` and `libtorch_python`.
The [installation guide](../../environments/README.md#install-and-relocate-the-runtime)
covers installing and moving the native prefix, and selecting the installed
attention library for Python.

Both models use `compiler/common/attention.py` for operator loading and export
shape registration. Qwen3-specific package checks live in `packages.py`.

## Export the canonical package pair

Choose a new or empty directory. One invocation exports and compiles both entry
points; the exporter refuses to replace a populated output directory.
These examples intentionally leave historical `artifacts/aoti/` packages alone.

```bash
export TORCHINDUCTOR_COMPILE_THREADS=2
export TORCHINDUCTOR_CACHE_DIR="$PWD/work/inductor-qwen3-flash"
"$Q3_PY" -m compiler.qwen3.export --output artifacts/qwen3-flash
```

The command accepts `--model-dir` and `--max-context-length`. When omitted,
capacity defaults to the selected model's declared context ceiling (40,960 for
the pinned checkpoint), and that capacity is actually allocated for K/V state.
The chosen capacity must be in 512..`max_position_embeddings` from the selected
checkpoint config (40,960 for the pinned checkpoint); invalid values are rejected
before GPU model loading. There is no numerical-mode or attention
selector. Export retains `emulate_precision_casts=True` and disables TF32;
ordinary Inductor fusion remains enabled. Use the same toolchain and tuning
cache for both entry points and record their actual compilation settings.
Changing the compiler or tuning cache can change rounding, so validate every
new pair rather than assuming cross-build bitwise reproducibility.

The directory contains weightless `prefill.pt2` and `decode.pt2`, matching JSON
metadata, exported graph text and `weights.pt`. The archive embeds its constant
mappings in `extra/weights.json`; there is no separate `weights.json` file. Both
entry metadata files use the shared `schema_version=5` format and record the
package and archive sizes and checksums. Common fields include `entry_point`,
`token_range`, `cache_length`, `torch_version`,
`cuda_compute_capability`, `vocab_size`,
`eos_token_ids` and ordered state descriptors. Each state gives its layer, kind,
shape, dtype and update policy; readers derive contiguous strides and names.
`runtime_abi_version=1` defines batch-one BF16 execution, input/output order,
logits shape, valid-prefix Flash attention and the shape-only CPU extent input.
Qwen3 needs no additional `model_specific` fields because its geometry is
available in the state descriptors. The runtime ABI replaces the former fixed
batch/dtype/attention/cache fields. Readers validate the actual attention-library
ABI, state contract, checkpoint, Torch version, GPU architecture, C++ ABI and
numerical settings. See the [ABI definition](../../docs/package-metadata.md#runtime-abi-1).

Export logs contain JSON events with dependency versions, Torch's CUDA version,
export strictness, adapter counts, graph counts and compilation timings. These
build and diagnostic details are not required deployment metadata. The full
resolved Inductor options are also logged; metadata retains the precision-cast
and two weightless-package options needed for the package contract.

Both entry points compile with `aot_inductor.package_constants_in_so=False` and
`always_keep_tensor_constants=True`. The common exporter saves each immutable
constant once and records exact per-package constant names. Python and C++ load
one CUDA weight pool, retain its owner until both loaders are destroyed, and
inject its tensors with `user_managed=True`. Loading verifies the metadata,
archive identity and every constant's shape/dtype; missing or mixed weights are
errors. Export stages both packages and graph dumps, writes the archive
once, and publishes the completed directory after both entries succeed. Both
metadata files contain the same final archive checksum. Deploy the five required
files (two `.pt2`, two `.json` and `weights.pt`) together with the native attention
library. Copying only the two `.pt2` files is insufficient.

### Package contract

Python signature: `(input_ids, cache_position, (K0, V0, ..., K27, V27), kv_extent)`.
The native loader receives 59 flat inputs: IDs, positions, 56 cache tensors, and
the extent tensor. Inputs are:

- CUDA int64 IDs `[1,T]` and contiguous absolute positions `[T]`.
- Caller-owned contiguous CUDA BF16 K/V buffers `[1,8,C,128]`, sharing the
  same stable storage between prefill and decode. Native allocations satisfy
  the 16-byte alignment requirement.
- A CPU uint8 extent vector of shape `[populated_length + 1]`. Only its shape is
  read; its data is ignored. This carries prefix length without a CUDA scalar
  synchronization or size-one specialization.
- One output: last-token CUDA BF16 logits `[1,151936]`.

Each compiled prefill call accepts 2–512 tokens; decode accepts one token,
including a one-token prompt at position zero. Default context capacity is
40,960 for the pinned model; `--max-context-length` can select a smaller export.
Shared generation
splits a long prompt into calls of at most 512 tokens, with a one-token remainder
sent to decode. `cache_length` records compiled capacity and `token_range` only
a compiled call's bounds. The prompt and requested output budget must fit their
combined context; there is no independent prompt cap. Earlier
positions must already be populated. Retained converted bundles keep their
original capacity; calling applications own any additional admission policy.
Larger state buffers require re-export, as described in
[package metadata](../../docs/package-metadata.md). The previously recorded
16,384-slot profile completed ordinary export and C++ execution on L20, with
bitwise native/Python AOTI agreement and a full-capacity inference-memory probe.
Qwen3 passed its existing HF numerical gates. Those results do not validate the
current 40,960-slot default. The recorded 16K profile has not been validated on GB10.
Runtime checks reject invalid IDs, nonpositive
or oversized budgets, incompatible metadata and cache overflow before model work.
The runtime also checks the actual CUDA device against the package's compute
capability before loading GPU weights. The common metadata reader validates
the schema version, model identity and all shared contracts. Native file
preflight verifies package and archive sizes and checksums before GPU weight
allocation. Missing or unsupported schema versions are rejected. Schema 1,
schema 2, schema 3, schema 4 and unversioned sidecars must be converted or the bundle re-exported; there are no
legacy field aliases. `model_type="qwen3"` selects the Qwen3 state policy after
common parsing.
This is an inference-only, unpadded, batch-one Qwen3-0.6B contract; arbitrary
masks, training and attention-weight requests are unsupported.

## Validate

The shared `tools.validation.validate` command reads the model type from package
metadata. Qwen3 hooks live in `tools/validation/qwen3/model.py`; the shipping-CLI
contract tool remains in that directory. The HF stage generates reference tokens
and logits during validation, before adapters or compilation are applied.
Model loading and tensor cache support remain in `compiler/qwen3/model.py`.
The runtime validator checks package metadata, resolves the attention library
linked by the selected native executable, and calls the shared `load_attention_ops`
loader directly. `compiler/qwen3/packages.py` retains the package ABI checks.
Preparation and shared validation/benchmark tooling tests live under
`tools/<area>/tests/`; see the shared
[CPU unit-test commands](../../docs/development-tools.md#cpu-unit-tests).
Shared operator tests and Qwen3 adapter tests live in
`compiler/common/tests/test_attention.py` and `compiler/qwen3/tests/test_attention.py`.
The focused tests construct an HF attention module with random weights; they do
not download a checkpoint or compile AOTI packages. They cover attention and
cache behavior, adapter rejection/reuse and exported graph contracts.
Run these commands from the repository root with the selected model environment.

```bash
"$Q3_PY" -m unittest \
  compiler.common.tests.test_attention compiler.qwen3.tests.test_attention
AOTI_RUNTIME_CHECK_INPUTS=1 "$Q3_PY" -m tools.validation.validate \
  --packages artifacts/qwen3-flash --native "$BUILD/xinfer_validate" \
  --output artifacts/qwen3-validation
"$Q3_PY" -m tools.validation.qwen3.validate_runtime \
  --model-dir "$Q3_MODEL" --packages artifacts/qwen3-flash \
  --native "$BUILD/xinfer_aoti" --output artifacts/qwen3-native-smoke
```

Use a fresh output directory for validation. The chain runs HF → eager →
Python AOTI → C++ AOTI on matched inputs. Eager applies the Flash adapter before
execution. The HF stage generates greedy continuations in memory, checks them
against HF `generate()`, and fixes the teacher-forced inputs used by later stages.
The chain compares adjacent stages and Python AOTI directly with the BF16 HF
baseline. Cache boundaries, future-slot poisoning, write regions,
storage reuse and overflow remain strict. The optional `--model-dir` selects a
prepared checkpoint; no saved-reference input is required. Validation separates
two conclusions:

- Runtime correctness requires finite results, bitwise native/Python AOTI
  agreement, correct cache writes/masking, stable storage, reset/reuse, legal
  boundaries/overflow rejection and matching native/Python generation.
- HF logit error remains bounded by the existing 5% row-relative L2 and
  peak-normalized error limits. HF cache differences and differing top-token or
  greedy-sequence choices are recorded diagnostics, with parity reported
  separately from the overall result. A successful run does not claim HF
  bitwise or token parity.

Logit comparisons use identical fed tokens, including when greedy trajectories
differ. The independent BF16 HF eager baseline remains in the reports. Focused
attention/operator checks retain their own numerical correctness requirements.
The original HF SDPA reference is not replaced by the compiled Flash adapter.
See [numerical history](../../docs/experiments/qwen3-numerics.md) for the recorded
14/56 cache differences and two top-token mismatches on L20.

## Generate in C++

The default decode backend is AOTI. The optional
[MegaQwen experiment](../../docs/experiments/qwen3-megaqwen.md) adds
`--decode-backend megaqwen` in builds with `XINFER_ENABLE_MEGAQWEN=ON`, sharing
the existing weights and KV storage. Its numerical acceptance remains open;
see the experiment's validation results and separate backend-comparison command.

```bash
"$BUILD/xinfer_aoti" artifacts/qwen3-flash work/prompt.ids work/generated.ids \
  --max-output-tokens 64 --trace work/native-logits.f32
"$Q3_PY" -m tools.prepare decode --model-dir "$Q3_MODEL" --input work/generated.ids
```

The native executable runs without an activated venv or interpreter. It resets
the cache for each request and stops at EOS or the requested limit. Output IDs
include EOS when reached; trace rows are raw little-endian float32. Input,
output and trace paths must differ. The prompt plus complete requested budget
must fit the package context capacity (40,960 by default), even if EOS
might arrive earlier. `--max-output-tokens` defaults to 64. The requested output
budget determines the remaining prompt allowance; there is no separate prompt cap.
Select the visible GPU with `CUDA_VISIBLE_DEVICES`.

## Benchmark and profile

```bash
"$Q3_PY" -m tools.benchmark run \
  --packages artifacts/qwen3-flash --native "$BUILD/xinfer_benchmark" \
  --output artifacts/qwen3-benchmark
```

The same metadata-selected command serves Qwen3 and Qwen3.5. `run` compares
ordinary HF eager (`hf`) with native AOTI (`cpp_aoti`) using identical forced
tokens. The benchmark tokenizes its own capital-of-France chat prompt and a fixed
"Paris is the capital of France. " continuation; saved validation fixtures are
not required. This changes the former Qwen3 workload, so compare workload hashes
when interpreting historical measurements.

The HF baseline uses the unchanged model with its own dynamic cache and
automatic masks/positions. Both models use normal PyTorch SDPA backend selection.
Each request starts with no cache; reset discards the cache and initial allocation
belongs to prefill. One-token initial requests use the same HF path. Reports
identify this baseline as `eager_baseline: hf_dynamic_cache`. Native prefill
timing includes all prompt chunks and the final prediction. HF receives the
whole prompt in one call; reports retain this partition difference. Numerical
validation uses matched partitions instead.

`compare` measures two compatible compiled package bundles for either model:

```bash
"$Q3_PY" -m tools.benchmark compare \
  --baseline-packages artifacts/aoti-baseline \
  --candidate-packages artifacts/aoti-candidate \
  --native "$BUILD/xinfer_benchmark" --output artifacts/qwen3-comparison
"$Q3_PY" -m tools.benchmark check-idle --output artifacts/qwen3-idle.json
```

`run` defaults to prompt lengths 25, 128 and 512, 64 fixed decode steps,
two warmups, five measured requests and three process rounds. `compare` defaults
to four rounds. `--lengths`, `--decode-steps`, `--warmup`, `--repeats` and
`--rounds` override those counts. Package metadata defines total context capacity
(`cache_length`); prompt and forced output tokens must fit that context for both
HF and native workers, while
`token_range` limits each compiled call. Use `--lengths 511 512 513 8191 8192`
to measure chunk boundaries and the complete long prompt.

All commands default to CUDA logical device 0, honoring `CUDA_VISIBLE_DEVICES`
and `CUDA_DEVICE_ORDER`. Use `--gpu` with an `nvidia-smi` physical GPU index or
GPU UUID to override that selection. Workers, idle checks, telemetry and GPU
records all use the selected UUID, recorded as `gpu_uuid` in `environment.json`.
The standalone idle check needs no model or package directory.

Reports preserve rotated process order, timings, package/binary/workload hashes,
memory and GPU telemetry. Accuracy checks and reports remain separate from
benchmarking. Use the [development tools guide](../../docs/development-tools.md)
for the timing protocol and paired comparisons. `--allow-busy` marks timings
provisional; controlled measurements require idle checks.
Earlier Qwen3 results used `QwenStep` with a masked 1,024-slot `TensorCache`;
rerun comparisons before quoting speedups for the dynamic-cache baseline.
Historical GB10 speedups do not establish a speedup for another toolchain/GPU.
