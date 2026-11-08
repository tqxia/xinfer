# Qwen3-0.6B AOTI / C++

Qwen3 defaults to fused BF16 AOTInductor with valid-prefix FlashAttention and
native GQA. An optional MegaQwen decode profile keeps AOTI prefill and the same
shared weights and KV cache. The original HF model supplies the weights and
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
The preparation command uses checkpoint revision
`c1899de289a04d12100db370d81485cdf75e47ca`.
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

The compiler's `xinfer_attention_ops` target builds
`compiler/common/kernels/attention.cpp` and its C shim into
`libxinfer_attention_ops.so`. Build that target before export or standalone
operator tests. The default root build includes the library and runtime
executables; explicit runtime targets build independently. The exporter embeds
the library in both `.pt2` files. Native executables load it through those
packages and require matching LibTorch/CUDA. The first CMake configuration
fetches the pinned nlohmann/json headers; later builds reuse the cached
dependency. Export links the shared
attention C shim through the selected Linux GCC toolchain.

Follow the target guide's build command, including `xinfer_attention_ops` and
its CUDA runtime selection on L20. Python tools default to the native
executables and attention library in the repository's `build/qwen3/`. Existing
`--native` options select
another executable. To select the attention library for export or standalone
operator tests:

```bash
export XINFER_ATTENTION_OPS_LIBRARY="$BUILD/libxinfer_attention_ops.so"
```

The library override is optional; the default is
`build/qwen3/libxinfer_attention_ops.so`. Package inference and validation use
the bundled implementation, independently of this build-time selection. Native
binaries do not link or install a separate attention library and must remain
free of `libpython` and `libtorch_python`.
The [installation guide](../../environments/README.md#install-and-relocate-the-runtime)
covers installing and moving the native executables with matching LibTorch/CUDA.

Both models use `compiler/common/attention.py` for operator loading and export
shape registration, and `compiler/common/package.py` for bundling native
dependencies through Torch's packager. Torch's `custom_ops_to_c_shims` and
`custom_op_libs` settings generate direct C calls and normal library linkage.
Scoped `LIBRARY_PATH` and `LD_RUN_PATH='$ORIGIN'` set link-time lookup and the
wrapper's relative runtime search path. `package_aoti` explicitly embeds the
library; the standard AOTI loaders handle extraction and model loading. See
[attention packaging](../../docs/flash-attention.md#build-and-deployment) for
ownership and export-process details. Qwen3-specific package checks live in
`packages.py`. Re-export retained external-library packages; a metadata-only
conversion cannot add their missing implementation.

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
the default checkpoint), and that capacity is actually allocated for K/V state.
The chosen capacity must be in 512..`max_position_embeddings` from the selected
checkpoint config (40,960 for the default checkpoint); invalid values are rejected
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
The default Qwen3 profile uses `model_specific: {}` because its geometry is
available in the state descriptors. The optional MegaQwen decode profile adds
the fields documented below. The runtime ABI replaces the former fixed
batch/dtype/attention/cache fields. Readers validate the state contract,
checkpoint, Torch version, GPU architecture, C++ ABI and
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
files (two `.pt2`, two `.json` and `weights.pt`). Each `.pt2` contains the same
attention library; its bytes are covered by the package checksum. LibTorch/CUDA
remain environment dependencies. Copying only the two `.pt2` files is insufficient.

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
40,960 for the default checkpoint; `--max-context-length` can select a smaller export.
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
schema 2, schema 3, schema 4 and unversioned sidecars need migration or re-export;
external-library bundles always require re-export. There are no legacy field
aliases. `model_type="qwen3"` selects the Qwen3 state policy after
common parsing.
This is an inference-only, unpadded, batch-one Qwen3-0.6B contract; arbitrary
masks, training and attention-weight requests are unsupported.

## Validate

The shared `tools.validation.validate` command reads the model type from package
metadata. Qwen3 hooks live in `tools/validation/qwen3/model.py`; the shipping-CLI
contract tool remains in that directory. The HF stage generates reference tokens
and logits during validation, before adapters or compilation are applied.
Model loading and tensor cache support remain in `compiler/qwen3/model.py`.
The runtime validator checks package metadata and shared-weight integrity,
then loads the bundled attention implementation for eager execution and shape
registration. Python AOTI and C++ load the dependency through their compiled
wrappers. `compiler/qwen3/packages.py` retains the package ABI checks.
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

The native runtime executes both compiled entries through AOTI. Qwen3 export
selects default Flash decode or the [optional MegaQwen profile](#optional-megaqwen-decode);
the package records that choice. Both profiles share the same runtime, weight
ownership and KV storage contract.

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

## Optional MegaQwen decode

### Compiled decode profile

Build the compiler's `xinfer_megaqwen_ops` target with
`XINFER_ENABLE_MEGAQWEN=ON`, then export the optional profile. Its CUDA core,
operator and C shim live in `compiler/qwen3/kernels/megaqwen/`. The current
implementation supports Qwen3-0.6B only and is not bound to an exact Hugging Face
repository or revision. The flag controls compiler kernels; native runtimes load
the implementation from the package.

Configure the matching toolchain first using the
[environment guide](../../environments/README.md#build-native-runtimes-for-each-model).
The following reconfigures that Qwen3 build for the L20 SM89 target and adds
correctness drivers. Other GPU targets need separate validation.

```bash
"$CMAKE" -S . -B "$BUILD" \
  -DXINFER_ENABLE_MEGAQWEN=ON -DXINFER_MEGAQWEN_CUDA_ARCHITECTURES=89 \
  -DXINFER_BUILD_VALIDATION=ON -DXINFER_BUILD_BENCHMARK=OFF
"$CMAKE" --build "$BUILD" --target \
  xinfer_attention_ops xinfer_megaqwen_ops xinfer_aoti xinfer_validate \
  qwen_attention_contract qwen_megaqwen_op_contract qwen_megaqwen_package_contract -j2

export XINFER_MEGAQWEN_LIBRARY="$BUILD/libxinfer_megaqwen.so"
"$Q3_PY" -m compiler.qwen3.export --model-dir "$Q3_MODEL" \
  --decode-backend megaqwen --output artifacts/qwen3-megaqwen
"$BUILD/xinfer_aoti" artifacts/qwen3-megaqwen work/prompt.ids work/generated-megaqwen.ids \
  --max-output-tokens 64
```

`--decode-backend aoti` is the exporter default. Both profiles publish the same
five-file bundle and runtime ABI 1. In the MegaQwen profile, `prefill.pt2` uses
the existing AOTI Flash adapter for 2–512-token chunks; `decode.pt2` invokes the
shared `xinfer::megaqwen_decode` C++ operator for every single-token call,
including position zero. A one-token remainder of a chunked prompt also uses
MegaQwen. Python AOTI and C++ AOTI execute this same graph and operator.

Both entries borrow their canonical model constants from one `weights.pt` pool
and mutate the same caller-owned full-capacity KV buffers. Read-only RoPE tables
are external constants in that archive; temporary decode workspace is allocated
by the compiled graph. No GPU addresses or mutable workspace are serialized.
The profile is recorded in both sidecars as
`model_specific: {decode_backend: megaqwen}`.

The exporter embeds `libxinfer_megaqwen.so` in both `.pt2` files through
`compiler/common/package.py`, following the attention-library packaging path.
The decode wrapper calls the shared implementation through a direct C shim and
links that dependency with an `$ORIGIN` search path. Standard Python and C++ AOTI
loaders extract and load it with the wrapper. Deploy the five-file bundle with
matching LibTorch/CUDA. Both decode profiles use the same AOTI runtime path;
`--decode-backend` selects the implementation during export.

`XINFER_MEGAQWEN_LIBRARY` optionally selects a different build for export or
standalone operator checks; its default is `build/qwen3/libxinfer_megaqwen.so`.
Package inference uses the implementation carried by the package. The production
numerical chain selects packaged attention and MegaQwen for its eager stage.
Re-export retained MegaQwen packages that used an external library or serialized
`megaqwen_ops_abi_version`; sidecar edits cannot supply the bundled implementation.

Validate the actual production bundle with the ordinary BF16 HF → adapted
eager → Python AOTI → C++ AOTI chain:

```bash
"$Q3_PY" -m tools.validation.validate \
  --packages artifacts/qwen3-megaqwen --model-dir "$Q3_MODEL" \
  --native "$BUILD/xinfer_validate" --output artifacts/qwen3-megaqwen-validation
"$Q3_PY" -m tools.validation.qwen3.megaqwen_package \
  --packages artifacts/qwen3-megaqwen \
  --native "$BUILD/qwen_megaqwen_package_contract" \
  --output artifacts/qwen3-megaqwen-storage
```

The package contract command requires a validation-enabled build. It checks
caller-owned KV storage, full cache write regions, poisoned future state, reset,
ordered streams, retained outputs and asynchronous destruction, and requires
bitwise Python/C++ logits and selected KV bytes. Generated-wrapper inspection
classifies guarded alignment repairs and rejects unexpected copy code in the
MegaQwen decode graph. It does not instrument the operator or inspect its live
arguments. These checks do not replace the HF numerical chain.

Keep the default AOTI profile as an independent control. Numerical limits and
state contracts are unchanged; implementation parity does not resolve known HF
numerical differences. Every single-token invocation uses the compiled
MegaQwen implementation, including position zero.

The existing benchmark `run` command can execute either bundle; its raw native
results retain `backend_info.compiled_decode_backend`. The top-level profile name
remains `cpp_aoti`, identifying the executor. Paired package `compare` accepts
default AOTI and MegaQwen bundles with matching checkpoint, canonical weights
and numerical settings. Keep the package metadata and raw backend information
with any measurements.

### Kernel and focused validation

The MegaQwen kernel implements batch-one Qwen3-0.6B: 28 layers,
16 query heads, eight KV heads, head dimension 128 and vocabulary 151,936. Its
cache strides and RoPE tables use the exported capacity in 512–40,960.

The shared runtime validates the complete prompt and generation budget before
execution, then splits prefill into calls of at most 512 tokens. A final
single-token remainder uses the compiled decode graph. For example, a 513-token
prompt uses Flash prefill for its first 512 tokens and MegaQwen decode for its
last token. A 1,024-token prompt uses two Flash prefill calls before generated-token
decode.

Attention uses streaming online softmax with one, two or four KV partitions per
query head. The untuned policy selects one partition through 128 valid KV tokens,
two through 256, and four beyond, reduced when the resident cooperative grid
cannot fit the additional attention blocks. Partial softmax statistics and
unnormalized values use 33,280 bytes of FP32 scratch shared across layers and
ordered calls. One partition preserves the original short-context arithmetic;
split reductions can change floating-point accumulation order. These properties
do not establish numerical acceptance or a speedup.

Run the standalone attention contract in a fresh output directory:

```bash
mkdir -p artifacts/qwen3-megaqwen-checks
"$BUILD/qwen_attention_contract" artifacts/qwen3-megaqwen-checks/attention.json
```

The attention check uses an independent CPU dense-softmax reference and the
same device arithmetic in `compiler/qwen3/kernels/megaqwen/attention.cuh` as decode.
Its launch wrappers live in `runtime/qwen3/tests/attention.cu` with declarations
in the adjacent test header. The production-package validation commands above
retain the live BF16 HF chain, strict native/Python AOTI equality, shared storage,
write-region, reset/reuse,
handoff and stream checks. Required HF logit limits remain 5%; passing runtime
checks does not resolve HF numerical failures. The numerical validation CLI
returns 0 when required checks pass, 1 when completed checks fail and 2 when
validation is incomplete. Correctness timings are not benchmark results.

The [pre-schema-5 experiment record](../../docs/experiments/qwen3-megaqwen.md)
preserves earlier fixed-capacity results and commands. Evaluate each current
build and package separately; those recorded outcomes do not qualify expanded
capacities or a rebased implementation.

### Shared operator and AOTI prototype

The compiled production profile above wraps the same operator in the ordinary
four-argument model ABI. The explicit-input prototype below remains useful for
isolating operator and compiler storage behavior.

The compiler-built library registers `xinfer::megaqwen_decode`. Python export
and direct operator checks load its C++ implementation and FakeTensor contract
through `compiler.qwen3.megaqwen.load_megaqwen_ops`. Its AOTI C shim calls the
same implementation. The op accepts CUDA token IDs, 311 read-only canonical
weight tensors, 56 mutable KV tensors, read-only
RoPE tables, 12 mutable workspace tensors and the CPU shape-only extent. It
returns fresh BF16 logits. Weight buffers must have gradients disabled; this is
an inference-only interface. Workspace belongs to each runtime, with calls
ordered on the caller's CUDA streams.

The tooling-only `MegaQwenStep` in `tools/validation/megaqwen_op.py` exports a
thin graph containing this op. Every tensor is a runtime input: the prototype serializes no model weights, workspace or raw GPU addresses.
The CUDA launch constructs a small pointer table from live tensors and passes it
by value. It reads the token on the device and derives position from the extent
shape, without a GPU-to-host scalar read.

Build the validation drivers as above, then use a fresh output directory:

```bash
"$CMAKE" --build "$BUILD" --target qwen_megaqwen_op_contract -j2
"$Q3_PY" -m tools.validation.megaqwen_op \
  --packages artifacts/qwen3-flash \
  --op-library "$BUILD/libxinfer_megaqwen.so" \
  --native "$BUILD/qwen_megaqwen_op_contract" \
  --output artifacts/qwen3-megaqwen-op-001
```

This generates a bare `megaqwen_decode.pt2` verification artifact with the
MegaQwen shared library bundled through the same C-shim packaging helper. Every
model tensor remains an explicit input. `--op-library` selects the build used for
direct eager/opcheck calls and compilation. The probe uses the retained
package's AOTI prefill and selects direct-op or AOTI execution for single-token
calls. Production inference uses the compiled package's decode graph.

The probe checks direct Python op, Python AOTI, direct C++ op and C++ AOTI
results bitwise. It checks caller-owned storage, input immutability and
cache write regions without an operator recorder. Generated-wrapper inspection
and a warmed allocation bound provide copy checks. AOTI retains guarded
input-alignment fallbacks that may copy misaligned tensors; the probe requires
16-byte-aligned caller inputs, classifies those sites and rejects unexpected
copy code. These checks do not inspect live arguments inside the compiled op.
Compiler alignment guards remain enabled. Fixtures exercise
split thresholds, dynamic extent bounds, poisoned state/scratch, reset/reuse,
initial-call routing, return handoffs, retained outputs and ordered nondefault
streams. Registration checks use `torch.library.opcheck`. A direct position-zero
probe also exercises the exported extent's lower bound.

These checks establish implementation parity and storage behavior for this
prototype. They do not replace the independent live BF16 HF chain or resolve its
known numerical failures. The AOTI frontends load the implementation bundled in
the prototype package. Python export and direct eager/opcheck calls use the
selected build library. The native test driver first loads the prototype AOTI
package and uses its own dispatcher helper to invoke the registered operator.

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

`compare` measures two compatible compiled package bundles for either model.
For Qwen3, it accepts default AOTI and MegaQwen exports with the same selected
checkpoint, canonical model weights, context and numerical settings. Only the
MegaQwen profile's derived RoPE buffers may differ in the weight source mapping.
Both bundles execute through the same native AOTI runtime:

```bash
"$Q3_PY" -m tools.benchmark compare \
  --baseline-packages artifacts/qwen3-flash \
  --candidate-packages artifacts/qwen3-megaqwen \
  --baseline-label aoti --candidate-label megaqwen \
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
