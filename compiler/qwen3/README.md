# Qwen3-0.6B AOTI / C++

Qwen3 has one supported compiled profile: fused BF16 AOTInductor with valid-prefix
FlashAttention and native GQA. The original HF model supplies the weights and
independent eager/FP32 references. The runtime performs inference and generation
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
into target-specific directories. Export-time Torch and native LibTorch must
come from the same selected installation.
The guide uses `environments/setup.sh` and `environments/check.sh` for setup and
checking; the Python implementations and tests also live under `environments/`.

`requirements.txt` pins this model's HF dependencies, including Transformers
4.51.3. Torch and Triton versions belong to the target profile. The existing
[`environments/l20/qwen3.lock`](../../environments/l20/qwen3.lock) records the
original L20 environment, including its CUDA packages. It is a historical full
snapshot, not the GB10 installation recipe.

Use the tested Torch 2.13 target profiles for new shared-weight exports. Torch
2.7 and packages with embedded constants are unsupported by the current runtime;
keep historical packages with their original binaries for rollback. Examples
below use the L20 paths; select the corresponding paths from the target guide
for GB10.

```bash
Q3_PY="$PWD/work/l20/qwen3-torch213-venv/bin/python"
BUILD="$PWD/work/l20/build-torch213"
```

## Prepare offline inputs

Run commands from the repository root. Qwen3 tools default to
`work/models/qwen3-0.6b`. Place the complete verified checkpoint there, including
`model_manifest.json`, or symlink an existing local checkpoint:

```bash
mkdir -p work/models
ln -s /path/to/qwen3-0.6b work/models/qwen3-0.6b
```

The checkpoint revision is `c1899de289a04d12100db370d81485cdf75e47ca`.
Helpers verify and read these files; they do not modify the checkpoint or
download another copy. Export, benchmark and model validation commands accept
`--model-dir` to select another directory. For preparation, put it before the
subcommand, for example:

```bash
"$Q3_PY" -m compiler.qwen3.prepare --model-dir /path/to/qwen3-0.6b verify
```

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
"$Q3_PY" -m compiler.qwen3.prepare verify
"$Q3_PY" -m compiler.qwen3.prepare encode \
  --prompt 'What is the capital of France? Answer in one short sentence.' \
  --output work/prompt.ids
# Run once in an absent output directory to generate independent HF fixtures.
"$Q3_PY" -m compiler.qwen3.tests.validate --output artifacts/reference
```

Reference generation checks manual HF DynamicCache decoding against HF
`generate()` and a fresh-cache repetition. It writes prompt/generated IDs,
little-endian float32 logits, text, metadata and a summary for factual,
arithmetic and Chinese fixtures. Thinking is disabled and greedy decoding is a
repeatable diagnostic. Both checkpoint EOS IDs are honored. Existing reference
fixtures remain usable; do not overwrite them during a compiler comparison.

## Build before export

The Qwen3 build always produces and links `libxinfer_attention_ops.so`; there is
no attention build switch. Both Python export and native inference require this
library and matching LibTorch/CUDA. The first CMake configuration fetches the
pinned nlohmann/json headers; later builds reuse the cached dependency.

Follow the target guide's native build command, including its CUDA runtime
selection on L20, then select that build for Python:

```bash
export XINFER_ATTENTION_OPS_LIBRARY="$BUILD/libxinfer_attention_ops.so"
```

Python defaults to `build/libxinfer_attention_ops.so`; the environment variable
selects another build directory. Native binaries link the library directly and
must be deployed with it and their matching LibTorch/CUDA libraries. A `.pt2`
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

The command accepts `--model-dir`. There is no numerical-mode or attention
selector. Export retains `emulate_precision_casts=True` and disables TF32;
ordinary Inductor fusion remains enabled. Use the same toolchain and tuning
cache for both entry points and record their actual compilation settings.
Changing the compiler or tuning cache can change rounding, so validate every
new pair rather than assuming cross-build bitwise reproducibility.

The directory contains weightless `prefill.pt2` and `decode.pt2`, matching JSON
metadata, exported graph text and `weights.pt`. The archive embeds its constant
mappings in `extra/weights.json`; there is no separate `weights.json` file. Both
entry metadata files use the shared `schema_version=1` format and record the
package and archive sizes and checksums. Common fields include `entry_point`,
`token_range`, `torch_version`, `torch_cuda_version`, `cuda_compute_capability`,
`vocab_size`, `eos_token_ids` and explicit state/input/output descriptors.
Qwen3 geometry and adapter details are nested under `model_specific`; source
and version information lives in `provenance`, with entry timing/graph results
in `diagnostics`. Metadata also records the fixed Flash backend, custom operator
ABI, extent/cache contract, checkpoint, Torch/CUDA versions, GPU architecture,
C++ ABI and actual numerical settings. Keep these fields even though the CLI no
longer selects a backend: they prevent incompatible packages being mixed.

Both entry points compile with `aot_inductor.package_constants_in_so=False` and
`always_keep_tensor_constants=True`. The common exporter saves each immutable
constant once and records exact per-package constant names. Python and C++ load
one CUDA weight pool, retain its owner until both loaders are destroyed, and
inject its tensors with `user_managed=True`. Loading verifies the metadata,
archive identity and every constant's shape/dtype; missing or mixed weights are
errors. Export stages both packages and their diagnostics, writes the archive
once, and publishes the completed directory after both entries succeed. Both
metadata files contain the same final archive checksum. Deploy the five required
files (two `.pt2`, two `.json` and `weights.pt`) together with the native attention
library. Copying only the two `.pt2` files is insufficient.

### Package contract

Python signature: `(input_ids, cache_position, (K0, V0, ..., K27, V27), kv_extent)`.
The native loader receives 59 flat inputs: IDs, positions, 56 cache tensors, and
the extent tensor. Inputs are:

- CUDA int64 IDs `[1,T]` and contiguous absolute positions `[T]`.
- Caller-owned contiguous CUDA BF16 K/V buffers `[1,8,1024,128]`, sharing the
  same stable storage between prefill and decode. Native allocations satisfy
  the 16-byte alignment requirement.
- A CPU uint8 extent vector of shape `[populated_length + 1]`. Only its shape is
  read; its data is ignored. This carries prefix length without a CUDA scalar
  synchronization or size-one specialization.
- One output: last-token CUDA BF16 logits `[1,151936]`.

Prefill accepts 2–512 tokens; decode accepts one token, including a one-token
prompt at position zero. Total context capacity is 1,024. Earlier positions
must already be populated. Runtime checks reject invalid IDs, nonpositive or
oversized budgets, incompatible metadata and cache overflow before model work.
The runtime also checks the actual CUDA device against the package's compute
capability before loading GPU weights. The common metadata reader requires
the schema version, package sizes and checksums, model identity and all shared
contracts before archive loading. Missing or unsupported schema versions are
rejected. Earlier unversioned sidecars must be converted or the bundle
re-exported; there are no legacy field aliases. `model_type="qwen3"` selects the
Qwen3 state policy after common parsing.
This is an inference-only, unpadded, batch-one Qwen3-0.6B contract; arbitrary
masks, training and attention-weight requests are unsupported.

## Validate

```bash
"$Q3_PY" -m compiler.qwen3.tests.custom_attention \
  --output artifacts/qwen3-attention-check
"$Q3_PY" -m compiler.qwen3.tests.attention_adapter \
  --output artifacts/qwen3-adapter-check
"$Q3_PY" -m compiler.qwen3.tests.validate_aoti --packages artifacts/qwen3-flash
AOTI_RUNTIME_CHECK_INPUTS=1 "$Q3_PY" -m compiler.qwen3.tests.validate_cpp \
  --packages artifacts/qwen3-flash --native "$BUILD/xinfer_validate" \
  --output artifacts/qwen3-validation
"$Q3_PY" -m compiler.qwen3.tests.smoke_cpp \
  --packages artifacts/qwen3-flash --native "$BUILD/xinfer_aoti" \
  --output artifacts/qwen3-native-smoke
```

Use fresh output directories. Validation separates two conclusions:

- Runtime correctness requires finite results, bitwise native/Python AOTI
  agreement, correct cache writes/masking, stable storage, reset/reuse, legal
  boundaries/overflow rejection and matching native/Python generation.
- HF logit error remains bounded by the existing 5% row-relative L2 and
  peak-normalized error limits. HF cache differences and differing top-token or
  greedy-sequence choices are recorded diagnostics, with parity reported
  separately from the overall result. A successful run does not claim HF
  bitwise or token parity.

Logit comparisons use identical fed tokens, including when greedy trajectories
differ. Independent HF eager and FP32 controls remain in the reports. Focused
attention/operator checks retain their own numerical correctness requirements.
The original HF SDPA reference is not replaced by the compiled Flash adapter.
See [numerical history](../../docs/experiments/qwen3-numerics.md) for the recorded
14/56 cache differences and two top-token mismatches on L20.

## Generate in C++

```bash
"$BUILD/xinfer_aoti" artifacts/qwen3-flash work/prompt.ids work/generated.ids \
  --max-new-tokens 64 --trace work/native-logits.f32
"$Q3_PY" -m compiler.qwen3.prepare decode --input work/generated.ids
```

The native executable runs without an activated venv or interpreter. It resets
the cache for each request and stops at EOS or the requested limit. Output IDs
include EOS when reached; trace rows are raw little-endian float32. Input,
output and trace paths must differ. The prompt plus complete requested budget
must fit 1,024 tokens even if EOS might arrive earlier. Select the visible GPU
with `CUDA_VISIBLE_DEVICES`.

## Benchmark and profile

```bash
"$Q3_PY" -m compiler.qwen3.benchmark run \
  --packages artifacts/qwen3-flash --native "$BUILD/xinfer_benchmark" \
  --output artifacts/qwen3-benchmark
```

`run` compares HF eager with one compiled profile using identical forced tokens.
`compare` compares two compiled artifact directories for optimization work.
Both commands default to the first CUDA-visible GPU, honoring
`CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER`. Use `--gpu` with an `nvidia-smi`
physical GPU index or GPU UUID to override that selection, for example `--gpu 1`
to select physical GPU 1. Workers, idle checks,
telemetry and GPU records all use the selected GPU; `environment.json` records
its `gpu_uuid`.

Both preserve rotated process order, warmups, package/binary/workload hashes,
memory and GPU telemetry. Use the [development tools guide](../../docs/development-tools.md)
for paired comparisons and numerical diagnostics. Shared-GPU smoke timings
are provisional; controlled performance measurements require the idle checks.
Historical GB10 speedups do not establish a speedup for another toolchain/GPU.
