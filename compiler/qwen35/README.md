# Qwen3.5 text-only AOTI / C++

The default target is **Qwen/Qwen3.5-0.8B**, pinned to
`2fc06364715b967f1860aea9cf38778875588b17`. Full-model 4B bring-up uses
**Qwen/Qwen3.5-4B**, pinned to `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,
through the same text preparation, export and runtime interfaces. Vision and MTP
are outside this experiment. Python prepares the model and tokenizes/decodes
offline; C++ loads the compiled packages and performs generation.

Use the explicit 4B paths below to select that checkpoint. Defaults remain 0.8B.
Kernel/block validation and full-model numerical acceptance are separate gates;
longer prompts remain a later milestone.

The [real 4B L20 run](../../docs/experiments/qwen35.md#full-4b-l20-validation-2026-11-04)
validates export, shared-weight loading and native/Python parity. Full HF/FP32
controls ran; the required numerical limits remain unmet.

Generation, fixed-token timing and native validation use the shared
`xinfer_aoti`, `xinfer_benchmark` and `xinfer_validate` executables. Package
metadata selects Qwen3.5 and its mixed-state policy; no model CLI flag is needed.

Prefill is always dynamic: each input of 2–512 tokens uses one prefill call.
A single token uses the decode package. Batch size is one and context capacity
is 1,024. Both sizes use BF16 weights/activations and FP32 RoPE frequencies.
State geometry comes from the selected checkpoint:

| Model | Layers | Conv/recurrent pairs | KV pairs | State tensors | KV shape |
| --- | --- | --- | --- | --- | --- |
| 0.8B | 24 | 18 | 6 | 48 | `[1,2,1024,256]` |
| 4B | 32 | 24 | 8 | 64 | `[1,4,1024,256]` |

Recurrent states use FP32. The caller owns the KV buffers, which are updated in place at the appended
positions. Returned KV tensors alias those buffers; conv/recurrent inputs remain
unchanged and their returned tensors replace the previous state.

Full-attention layers use native FlashAttention over the populated cache
prefix, with grouped KV heads and no explicit head expansion. The original HF
projections, Q/K normalization, RoPE and sigmoid output gate remain in the model.
DeltaNet layers retain their packaged Triton kernels.

One functional dispatcher selects fused recurrence for T<=97 and B3/C64
chunked FP32 prefill for longer requests. The threshold is provisional. Focused
validation compares it with HF recurrent/chunked operations and an independent
FP32 control; full-model and native checks cover generation and carried state.
HF/compiler numerical acceptance remains **open**. The experiment record retains
the earlier validation results, including the retired standalone-core control.
No timing result is claimed on the busy development GPU.

## Source layout

- `model.py`: prepared checkpoint loading, `TextStep` and explicit state ABI.
- `adapters/attention.py`: reuse HF attention weights and gating around the
  shared native prefix-FlashAttention operator.
- `adapters/delta.py`: reuse HF weights, projections and gated norm; call unified
  Triton preparation/recurrence for prefill and decode.
- `kernels/recurrent.py`: fused recurrence, parallel preparation, functional
  branch operators and shape-based `torch.cond` dispatch.
- `kernels/chunked.py`: C64 FP32 chunking with B3's 32-column U/W transform.
- `export.py`: strict export and AOTInductor packaging.
- `../../tools/prepare/`: shared checkpoint download/verification and offline token IDs.
- `tests/`: model/compiler, GPU operator and AOTI tests; `delta.py` contains
  synthetic DeltaNet fixtures and explicit-state HF wrappers.
- `../../tools/validation/validate.py`: the common numerical chain;
  `../../tools/validation/qwen35/` supplies HF/eager model hooks and runtime-contract
  checks; the common validation runner loads Python AOTI packages.
- `../../tools/diagnostics/qwen35/probe_rounding.py`: the retained Q35-NUM-001
  rounding reproducer.

Preparation, benchmark and validation tooling unit tests live under their
respective `tools/<area>/tests/` directories. Checkpoint loading/configuration
tests remain in `tests/test_model.py`. See the shared
[CPU unit-test commands](../../docs/development-tools.md#cpu-unit-tests).

`runtime/qwen35/runtime.h` owns the model-specific metadata checks. The runner,
state policy and shared preflight live in `runtime/common/`; the native
entry points are `runtime/main.cpp`, `runtime/benchmark.cpp` and
`runtime/validate.cpp`.

## Prefix attention and caller-owned KV

Both entry points accept the same state interface:

```text
step(input_ids, cache_position, state, kv_extent) -> last_logits, *next_state
```

`state` contains two tensors per layer in layer order, with shapes from the
table above. Updates use in-place `index_copy_`, writing only the
current token rows. Conv/recurrent pairs retain their functional update
contract: the DeltaNet adapter reads the caller's tensors directly without
cloning them and returns fresh tensors for the next call. KV outputs alias the
input buffers.

Inputs must be unpadded, batch-one contiguous appends to a populated prefix.
`kv_extent` is a CPU `uint8` tensor whose shape is `[valid_length + 1]`, including
the current append. Only its shape is read; no CUDA scalar is copied back to
discover the length. FlashAttention receives views of the valid K/V prefix and
handles grouped heads directly: eight query heads, two KV heads and head
dimension 256 for 0.8B. The shared native operator and its Python shape contract
derive GQA geometry from the inputs: query heads must be a positive multiple of
KV heads, with positive head dimensions divisible by eight and at most 256. This includes
4B's 16 query heads, four KV heads and head dimension 256. Its lower-right causal
alignment supports both decode and
prefill chunks appended after an existing prefix. Future capacity is neither
copied by cache updates nor expanded into attention temporaries.

Both entries use the same attention implementation and have no backend selector.
Rebuild the runtime and re-export packages made with the earlier functional KV and
`efficient_sdpa` contract. The runtimes reject that legacy metadata before
loading the packages. A backend change can alter floating-point results;
the existing numerical validators and limits still apply.

## DeltaNet tests

The unit tests exercise the production kernels directly. T<=97 uses the
existing fused recurrence; longer inputs use C64 chunking with the measured B3
32-column U/W transform. Threshold 97 comes from an isolated-kernel crossover
and still needs full-model performance evaluation on an idle target.

The condition uses `qkv.shape[1]` directly. Each branch owns fresh output/state
tensors, and only the long branch allocates chunk workspaces. Strict export
retains one dynamic prefill range, T=2..512. AOTI packages both branches and
generates a native shape comparison; T=1 uses a separate decode package.

Select `Q35_PY` using the [target environment guide](../../environments/README.md)
and set `BUILD="$PWD/build/qwen35"`.
Run the numerical and behavior tests without downloading a checkpoint or
compiling AOTI packages:

```bash
QWEN_PY="$Q35_PY"
"$QWEN_PY" -m unittest compiler.qwen35.tests.test_delta compiler.qwen35.tests.test_kernels
```

These GPU tests cover both model geometries, BF16 and FP32, zero and nonzero
state, gate extremes, convolution cancellation, partitioning, repeated requests,
dispatch boundaries, state carried across short/long calls, streams and layouts.
Kernel numerical limits remain 1% for BF16 and 0.01% for FP32, with the existing
1e-7 FP32 absolute-roundoff allowance. HF recurrent/chunked and independent FP32
controls remain part of the tests; a passing compiled check does not establish
numerical accuracy. The existing BF16-to-FP32 limit violations in both the
production and HF paths remain ordinary test failures.

Run the export and AOTI tests separately when changing kernels or the compiler:

```bash
"$QWEN_PY" -m unittest compiler.qwen35.tests.test_kernels_aoti
```

These GPU tests cover both geometries and compile four packages per geometry:
BF16/FP32 prefill and decode. They check symbolic dispatch before and after
lowering, dynamic lengths, carried/reset state and nondefault CUDA streams.
Direct kernel execution and Python AOTI must agree bitwise on core output,
convolution history and recurrent state. AOTI input guards exercise canonical
strides, including singleton dimensions. Packages use temporary directories
that are cleaned up automatically; there is no kernel-validation CLI, output
report directory or dedicated kernel C++ runner. Normal test discovery includes
these tests, and they skip when CUDA is unavailable. Full-model validation
retains C++ coverage.

## Dynamic full-model prefill and Triton GDN

`torch.cond` retains both GDN branches in the exported graph. Each branch uses
`torch.library.triton_op` and `wrap_triton` so AOTInductor can compile the kernels and
include their cubins in each `.pt2` package. The C++ loader needs the two packages,
their JSON metadata, `weights.pt`, `libxinfer_attention_ops.so` and matching
LibTorch/CUDA dependencies. It needs no Python or Triton installation at
inference time. GDN kernels remain inside the packages; the shared native
library supplies FlashAttention.

The separate cubin builder, CUDA-driver launcher and ATen chunk implementation
have been removed. Export has no backend or dynamic-prefill switch. Older fixed
or external-custom-op packages need to be re-exported for this runtime.

The GDN adapter retains the same functional tensor interface for both entry points:

```text
fused(qkv, b, a, conv, state, weight, A_log, dt_bias)
    -> core, next_conv, next_recurrent
```

Both geometries use 16 key heads (`Hk`), head dimensions `K=V=128` and a
bias-free width-four SiLU convolution. The value-head count (`Hv`) is 16 for
0.8B and 32 for 4B:

| Tensor | 0.8B | 4B |
| --- | --- | --- |
| QKV | `[1,T,6144]` | `[1,T,8192]` |
| A/B gates | `[1,T,16]` | `[1,T,32]` |
| Convolution state | `[1,6144,4]` | `[1,8192,4]` |
| Convolution weights | `[6144,4]` | `[8192,4]` |
| Decay parameters | `[16]` | `[32]` |
| Recurrent state | `[1,16,128,128]` | `[1,32,128,128]` |
| Core output | `[1,T,16,128]` | `[1,T,32,128]` |

Inputs and parameters use matching BF16 or FP32, with recurrent state always
FP32. The wrapper establishes contiguous inputs and allocates fresh outputs.
Gates, decay and recurrent state belong to each value head. Grouped value heads
reuse their corresponding Q/K head, matching HF's `repeat_interleave` order.

Decode keeps the existing fused kernel: one program owns one head and 32 value
columns and combines bias-free width-four convolution/history handling, SiLU,
beta/log-decay preparation, Q/K normalization, state update and output reduction.
Four value tiles share Q/K preparation. Only tile zero of the first value head
in each Q/K group writes that group's convolution history, avoiding duplicate
writes when 4B's two value heads share one key head.

Short prefill reuses fused recurrence. Long prefill prepares Q/K/V and gates in
parallel across tokens and value heads, materializing Q/K per value head for
the existing chunked scan. Its Q/K convolution history also has one writer per
group. It then runs
four chunk kernels: cumulative decay and triangular solve, U/W transforms,
state propagation between chunks, and parallel outputs. The fixed chunk size
is 64. It is an internal compute tile; dynamic T=2..512 still uses one prefill
package, with masked neutral tails rather than separate length buckets. The
U/W output-feature tile is 32; the dot reduction still covers all 64 chunk rows.
Workspaces, state arithmetic and matrix products use FP32 with explicit IEEE
dot precision. Convolution/SiLU and beta retain their input-dtype rounding.

Both paths retain the same eight-input/three-output operator interface and
require no extra flag. Input projections, z-gated normalization and output
projection stay outside. Focused validation uses HF recurrent and chunked
references; the runtime continues to load packaged Triton without Python.

## Prepare, export and build

L20 and GB10 use the same model adapter, GDN kernels and C++ runtime. Follow
[the target environment guide](../../environments/README.md) for the native
Torch/Triton/CUDA stack, isolated model environments, the `build/qwen35/`
native directory and target-specific packages. This model's self-contained
requirements files are [`environments/l20/qwen35_requirements.txt`](../../environments/l20/qwen35_requirements.txt)
and [`environments/gb10/qwen35_requirements.txt`](../../environments/gb10/qwen35_requirements.txt).
Each specifies the PyTorch index and all Torch, Triton, build-tool and HF pins,
including Transformers 5.20.2. Setup and checking read the same selected file;
its Torch version can be changed independently of Qwen3.
Target profiles contain hardware checks.
Use `environments/<target>/setup.sh` to install a fresh model venv, followed by
`environments/check.sh`, through that guide. Setup is implemented in Bash;
environment tooling and its tests live under `environments/`.
For full-model 4B work, use the
[offline L20 Docker testing environment](../../environments/l20/docker/README.md).
It provides `Q35_PY` and the matching native stack for the commands below.
Export-time Torch and native LibTorch must come from the same installation.
Optional `fla` and `causal_conv1d` must be absent so the surrounding HF operations
use the validated tensor implementation.

Use the Torch 2.13 target requirements for the current shared-weight format. Keep
older Torch 2.7 bundles with their matching runtime for historical comparisons.
Rebuild and re-export on the target GPU and toolchain; compiled SM89 packages
remain specific to their deployment stack. The shared target guide sets
`Q35_PY`, the matching CUDA runtime link flags and build tools. Use
`BUILD="$PWD/build/qwen35"` for this model.

Run from the repository root. One export invocation compiles both entry points
sequentially. Choose a new or empty `--output` directory for each experiment;
export refuses to replace a populated directory.

Build the native targets first using the target environment guide. Qwen3.5
Python tools default to native executables and the FlashAttention library in
the repository's `build/qwen35/`. To select another library for Python:

```bash
export XINFER_ATTENTION_OPS_LIBRARY="$BUILD/libxinfer_attention_ops.so"
```

The override is optional; the default is `build/qwen35/libxinfer_attention_ops.so`.
Native full-model targets link the library directly. Deploy it with the native
executable and matching LibTorch/CUDA; it is not embedded in the `.pt2` files.
The [installation guide](../../environments/README.md#install-and-relocate-the-runtime)
covers the installed native runtime, prefix relocation and selecting its
attention library for Python export or validation.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
QWEN_PY="$Q35_PY"
# Keep large compilation scratch files off a small system /tmp filesystem.
mkdir -p work/qwen35/tmp work/qwen35/inductor-cache
export TMPDIR="$PWD/work/qwen35/tmp"
export TORCHINDUCTOR_CACHE_DIR="$PWD/work/qwen35/inductor-cache"
# Prepare once before enabling offline mode, or copy a verified text directory.
"$QWEN_PY" -m tools.prepare download --model Qwen/Qwen3.5-0.8B
"$QWEN_PY" -m compiler.qwen35.export
```

For 4B, select its pinned checkpoint and keep its prepared weights and compiled
bundle separate from the 0.8B defaults:

```bash
"$QWEN_PY" -m tools.prepare download \
  --model Qwen/Qwen3.5-4B --output work/qwen35/4b/text
"$QWEN_PY" -m compiler.qwen35.export \
  --model-dir work/qwen35/4b/text --output artifacts/qwen35/4b/text
```

The shared preparation tool downloads pinned source shards, verifies their Hub
checksums, and extracts and renames only language-model tensors into
`text.safetensors`. Preparation runs on CPU without CUDA or inference. It refuses
an existing output path and publishes the prepared directory only after success;
reuse the complete verified directory on later runs. `--cache-dir` selects a Hub
cache; otherwise the Hub's configured cache is used. Run download before enabling
`HF_HUB_OFFLINE` or `TRANSFORMERS_OFFLINE`. Prepared text checkpoints use
`model_manifest.json`, the same filename as Qwen3. Rename retained checkpoints' former `manifest.json`
without changing its contents; existing compiled packages remain usable.
The manifest records the selected model ID and pinned
revision, checksums of source shards and all prepared files. Preparation and
explicit `tools.prepare verify` check checkpoint integrity. Model loading reads
the prepared files without repeating manifest or checksum verification. Both
supported checkpoints require tied embeddings and default RoPE.

Ensure `nvcc` is on `PATH`, or pass `-DCMAKE_CUDA_COMPILER=/path/to/nvcc`.
Export produces `prefill.pt2`, `decode.pt2`, their `.json` metadata and the shared
`weights.pt` archive under
`artifacts/qwen35/0.8b/text`. Both model families use `schema_version=3`, written
by `compiler/common/metadata.py` and read by the common Python/C++ metadata
readers. Common fields include `model_type="qwen35"`, `entry_point`, token ranges,
`runtime_abi_version=1`, Torch version, C++ ABI, GPU architecture, vocabulary,
EOS IDs and ordered tensor state descriptors. Each state records its layer, kind, shape, dtype and update
policy; readers derive names, contiguous strides, layer types and input/output
order from the supported model ABI. `checkpoint` contains the model ID, revision
and source manifest. The [runtime ABI](../../docs/package-metadata.md#runtime-abi-1)
defines batch-one BF16 execution, the Flash operator, shape-only extent input and
KV-only mutation policy; the former fixed descriptions are no longer separate
metadata fields. Readers query the loaded attention library's actual ABI.

`model_specific` contains the DeltaNet backend, short/long operators and
algorithms, threshold/chunk/tile/precision settings and a minimal packing
flag/version. Layer layout and geometry come from the state descriptors.
Export logs contain JSON events with dependency versions, Torch's CUDA version,
export strictness, graph counts, symbolic ranges, packing details and timings.
Full resolved Inductor tuning options are logged; metadata retains the
precision-cast and two weightless-package options. Build diagnostics are not
required deployment fields.

Runtimes require schema 3 and runtime ABI 1. Retained compatible Flash/shared-weight
bundles can migrate by rewriting their two JSON sidecars while keeping the compiled graphs
and weight archive unchanged. Schema 1, schema 2 and unversioned sidecars are
not read through legacy aliases.
The archive embeds its mappings in
`extra/weights.json`; no standalone weight manifest is deployed. Both entry
metadata files contain the same final archive size and checksum. The exporter
stages both entries, writes the archive once, and publishes the completed
directory only after both succeed. Keep all five required files
(two `.pt2`, two `.json` and `weights.pt`) together. The runtime rejects incompatible pairs and
missing or mismatched weights; old embedded-weight bundles need re-export.

Export runs linear packing before extracting constants, so the archive contains
the packed tensors used by the compiled graph. Shared and tied tensors are
deduplicated. Each package records its exact constant-name mapping, including
lifted constants, inside the archive. Compilation sets `aot_inductor.package_constants_in_so=False`
and `always_keep_tensor_constants=True`. Prefill and decode therefore carry no
private copy of the model weights.

Both runtimes load the archive to CUDA once and retain that owner for the full
loader lifetime. They inject each package's constants with
`load_constants(..., user_managed=True)` (the fourth C++ argument). The native
loader verifies that its actual constant pointers refer to the shared pool.
Treat weights as immutable while either loader is running. Both exports use one
loaded model and matching compiler settings. Inconsistent shared tensor content
is rejected before the bundle is published.

Export uses ordinary Inductor fusion and reductions, with
`emulate_precision_casts=True` and TF32 disabled. There is no `--eager-numerics`
profile. Floating-point results can differ from HF eager; the existing
full-model numerical issues and validation limits remain documented in the
[experiment record](../../docs/experiments/qwen35.md). Legacy packages marked
`eager_numerics=true` must be re-exported with the current exporter.

### Optional compiler optimizations

Pass `--enable-inductor-optimization` to enable the current Inductor optimization
settings. This generic flag currently sets `combo_kernels=True` and
`benchmark_combo_kernel=True`; future optimizations can extend the same flag.
Export the pair with matching settings into a fresh directory:

```bash
"$QWEN_PY" -m compiler.qwen35.export --enable-inductor-optimization \
  --output artifacts/qwen35/0.8b/text-optimized
```

Metadata records the `enable_inductor_optimization` flag and resolved
`inductor_configs`; the runtime rejects package pairs with different configs.
The flag is off by default and
requires no runtime switch. Rerun the unified numerical and runtime-contract checks with
`--packages artifacts/qwen35/0.8b/text-optimized`, since these optimizations can
change floating-point results.

## Generate and validate

The prepared manifest selects the model family for the shared offline token
commands. Token output must be a new file. For a copied or retained checkpoint,
run an explicit integrity check when needed; successful download already performs
verification:

```bash
"$QWEN_PY" -m tools.prepare verify --model-dir work/qwen35/0.8b/text

"$QWEN_PY" -m tools.prepare encode --model-dir work/qwen35/0.8b/text \
  --prompt "What is 17 + 25?" --output work/prompt.ids
"$BUILD/xinfer_aoti" artifacts/qwen35/0.8b/text \
  work/prompt.ids work/generated.ids
"$QWEN_PY" -m tools.prepare decode --model-dir work/qwen35/0.8b/text \
  --input work/generated.ids

"$QWEN_PY" -m unittest compiler.qwen35.tests.test_kernels_aoti
"$QWEN_PY" -m tools.validation.validate \
  --packages artifacts/qwen35/0.8b/text \
  --output artifacts/qwen35/0.8b/validation
"$QWEN_PY" -m tools.validation.qwen35.validate_runtime \
  --output artifacts/qwen35/0.8b/runtime-new
```

Kernel numerical and behavior checks run through `test_kernels`; the
`test_kernels_aoti` module checks exact direct-kernel/Python AOTI parity.
The unified numerical command selects Qwen3.5 from package metadata and runs
HF → eager → Python AOTI → C++ AOTI. The eager stage includes the adapters
and linear packing selected by package metadata. It checks adjacent stages and
Python AOTI directly against HF. Teacher forcing uses identical tokens and
partitions throughout. Independent FP32 HF controls
remain diagnostic. `validate_runtime` separately checks runtime/CLI contracts.
Inspect complete reports before treating a nonzero exit as an expected
numerical gate.

Use matching model, package and tokenization paths for the complete 4B model:

```bash
"$QWEN_PY" -m tools.prepare encode --model-dir work/qwen35/4b/text \
  --prompt "What is 17 + 25?" --output work/qwen35/4b/prompt.ids
"$BUILD/xinfer_aoti" artifacts/qwen35/4b/text \
  work/qwen35/4b/prompt.ids work/qwen35/4b/generated.ids
"$QWEN_PY" -m tools.prepare decode --model-dir work/qwen35/4b/text \
  --input work/qwen35/4b/generated.ids
"$QWEN_PY" -m tools.validation.validate \
  --model-dir work/qwen35/4b/text --packages artifacts/qwen35/4b/text \
  --output artifacts/qwen35/4b/validation
"$QWEN_PY" -m tools.validation.qwen35.validate_runtime \
  --model-dir work/qwen35/4b/text --packages artifacts/qwen35/4b/text \
  --output artifacts/qwen35/4b/runtime-new
```

Run the unified numerical and runtime-contract checks against the same prepared
checkpoint and package pair. Both reuse prompt definitions and long-prompt text
from `tools/validation/common/prompts.py`, preserving existing texts and case names.
Both model families generate HF references during numerical validation, without
saved-reference input files or a separate preparation command. Qwen3 additionally
checks its manual HF greedy decoding against HF `generate()`.
The numerical report preserves the model and
end-to-end fixture origins, intermediate-stage isolation, independent FP32
controls, native/Python bitwise comparisons, carried state and generation
checks. Logits and state retain both 5% error limits. HF token and greedy
agreement remains required for the original end-to-end fixtures; model-only
token differences remain diagnostic. Review the complete reports before
accepting a new package.

The kernel unit and AOTI test modules each cover both the 0.8B and 4B
geometries automatically, using synthetic fixtures without a model checkpoint.
Both retain T=1..512 and the 97-token dispatch threshold.
Kernel/block results do not establish full-model 4B support or acceptance of
longer inputs.

Chunking reorders arithmetic across sequence partitions. Core output and
recurrent-state partition comparisons use the existing BF16/FP32 numerical
limits, while convolution history remains exact. Repeated requests and
same-input direct-kernel/AOTI comparisons retain their bitwise requirements.

Use `--packages` and `--binary`/`--native`/`--cli` to select other artifact and
build directories. Validation output directories must be fresh.
Runtime and unified numerical reports include package and native binary checksums.
The AOTI unit tests check core output and both carried states for BF16/FP32
prefill/decode, dynamic lengths and mixed continuations. Broader numerical,
partitioning and gate-extreme checks live in `test_kernels`. These describe test
coverage, not full-model acceptance. Full-model native processes require the
shared attention library while remaining free of Python libraries. Full-model
checks retain all state tensors, HF/FP32 controls, dispatch counts, stable KV storage, appended-write
regions, future-prefix isolation and CLI rejection cases. Numerical validators exit
nonzero for acceptance failures and retain the reports.

Recorded reports from the initial unified implementation are under
`artifacts/qwen35/0.8b/unified-fused-{validation,model,runtime}/summary.json`;
the full-model packages are in `artifacts/qwen35/0.8b/text-unified-fused`.
Those reports predate removal of the standalone recurrent control. Historical
cleanup revalidation is recorded in
`artifacts/qwen35/0.8b/remove-recurrent-validation/summary.json`: all 396 native
calls / 1,188 tensors match Python AOTI bitwise; fusion checks pass. The fused
and HF paths each retain 125 FP32 accuracy failures, so overall acceptance is open.
All of these reports precede the chunked prefill PoC. Re-export and run the
current validators into fresh output directories for its validation.
The [experiment record](../../docs/experiments/qwen35.md#unified-deltanet-preparation-and-recurrence-2026-10-29)
records the code-generation workaround, numerical limits and temporary raw
fixture location.

## Projection weight packing

Qwen3.5 export always runs the generic pass in
`compiler/common/passes/pack_linears.py` after strict export and before
Inductor decomposition. It combines independent, bias-free linear
projections sharing the same input. Weights must be frozen parameters with
exclusive storage and no other graph consumers; shared/tied/trainable weights
are skipped. Packed parameters are created once during export. Replaced
parameters are removed, and there is no inference-time weight concatenation.

For 0.8B, 48 groups combine DeltaNet inputs, MLP gate/up and attention QKV,
reducing 187 linear calls to 97. Split outputs retain the original BF16 dtype
and contiguous layout for downstream views. Contiguous copies can cost time
in prefill; packing reduces launches rather than the number of weight bytes.
Packing is the default for both prefill and decode. It adds no model adapter or
native library dependency beyond the full-model FlashAttention path.

```bash
"$QWEN_PY" -m compiler.qwen35.export
"$QWEN_PY" -m tools.validation.qwen35.validate_runtime \
  --packages artifacts/qwen35/0.8b/text \
  --output artifacts/qwen35/0.8b/packing-runtime-new
"$QWEN_PY" -m tools.validation.validate \
  --packages artifacts/qwen35/0.8b/text \
  --output artifacts/qwen35/0.8b/packing-validation-new
```

Exports default to `artifacts/qwen35/0.8b/text`. Use a fresh `--output` directory
if older unpacked packages already occupy that path. The metadata's
`model_specific.linear_packing` field records whether packing is enabled and its
policy version; C++ and Python reject mismatched pairs. Logical groups and the
full packing report remain in export logs and the separate `prefill.packing.json` /
`decode.packing.json` files. Those reports map FX outputs and source parameter
names to each packed constant,
which helps profiling attribution.

For packed packages, the eager stage runs the exported graphs after adapters
and linear packing have both been applied. The `eager_vs_hf` and
`python_aoti_vs_eager` comparisons separate the combined eager transformation
from compilation. Changed matrix shapes can change floating-point reductions;
HF/FP32 controls and numerical limits remain in place. The export CLI has no
packing switch.

Hold packing constant across compiler-tuning arms. For a packing A/B experiment,
use an unpacked control checkout with the same selected Inductor configuration.

## Benchmark

```bash
"$QWEN_PY" -m tools.benchmark run \
  --packages artifacts/qwen35/0.8b/text \
  --output artifacts/qwen35/0.8b/benchmark-new
"$QWEN_PY" -m tools.benchmark run \
  --packages artifacts/qwen35/4b/text \
  --output artifacts/qwen35/4b/benchmark-new
```

Package metadata selects the model family and the prepared 0.8B or 4B checkpoint.
`--model-dir` selects another local path for that same checkpoint; `--native`
selects another `xinfer_benchmark` build. The native executable defaults to
`build/qwen35/xinfer_benchmark`. The same CLI and defaults serve Qwen3.

Both models support paired package comparison and a standalone idle check:

```bash
"$QWEN_PY" -m tools.benchmark compare \
  --baseline-packages artifacts/qwen35/4b/baseline \
  --candidate-packages artifacts/qwen35/4b/candidate \
  --output artifacts/qwen35/4b/comparison-new
"$QWEN_PY" -m tools.benchmark check-idle --output artifacts/qwen35-idle.json
```

`run` compares ordinary HF eager (`hf`) with native AOTI (`cpp_aoti`) using
identical fixed tokens. Both models tokenize the capital-of-France chat prompt
and a fixed "Paris is the capital of France. " continuation in the benchmark.
The unchanged HF model creates and carries its own dynamic cache and constructs
its masks and positions. Normal PyTorch SDPA backend selection remains enabled.
Qwen3.5 retains its environment checks. Reset discards the cache, so initial
allocation is counted in prefill; one-token initial requests follow the same
path. Reports record `eager_baseline: hf_dynamic_cache`. EOS is data during timing.

Defaults are prompt lengths 25, 128 and 512, 64 decode steps, two warmups,
five measured requests and three process rounds (`compare` uses four rounds).
The corresponding count options override these values. Accepted prompt lengths
and total context capacity come from package metadata.

All commands default to CUDA logical device 0, honoring `CUDA_VISIBLE_DEVICES`
and `CUDA_DEVICE_ORDER`. `--gpu` accepts an `nvidia-smi` physical index or GPU
UUID and overrides inherited visibility. Workers, idle checks, telemetry and GPU
records use the same UUID, stored as `gpu_uuid` in `environment.json`.
The standalone idle check loads no model and needs no package directory.

Reports contain reset/prefill/decode timings, memory, workload and telemetry,
with package, binary and workload checksums in `environment.json`. Accuracy
validation and its reports remain separate; benchmarks do not consume validation
reports. `--allow-busy` marks timings provisional. Controlled measurements require
an idle environment.
Earlier Qwen3.5 benchmarks used an HF `StaticCache` with an explicit 1,024-slot
mask; rerun comparisons for the dynamic-cache baseline. The matched static-cache
HF wrapper in the validation model hooks remains a numerical control.

[The Inductor tuning guide](../../docs/experiments/qwen35-inductor-tuning.md)
records the original generated-GEMV and GEMM-autotuning experiments. Use the
current benchmark command for new measurements.

## Numerical issues and historical evidence

- **Q35-NUM-001:** lost BF16 rounding before DeltaNet gated normalization.
  `python -m tools.diagnostics.qwen35.probe_rounding` reproduces the issue;
  exit zero means reproduced, not fixed.
- **Q35-NUM-002:** partition-sensitive greedy choices in earlier experiments.
  Triton full-model tests also record differences already present before
  Inductor; they cannot all be attributed to the lost rounding boundary.

[The experiment record](../../docs/experiments/qwen35.md) preserves earlier
fixed-prefill, ATen custom-op and manually packaged Triton results as history.
Their removed source files can be recovered from Git history or the recorded
source snapshots. The retired fixed-shape block exporter
(`diagnostics/export_block.py`) and original HF dynamic-export probe
(`diagnostics/probe_dynamic.py`) are archived in commit `0984678`; their
historical commands require that checkout or its source snapshot. The legacy
`qwen35_block` runner, `tests/validate_block.py` and fixed-shape sequence helpers
were subsequently retired; their last containing commit is `a63eb21`.
`TensorState` and `BlockStep` live in `compiler/qwen35/tests/delta.py` for
synthetic unit tests and the rounding diagnostic. The diagnostic loads the
prepared full-model text checkpoint; it does not need a separate block checkpoint.
Generated evidence remains ignored in `artifacts/qwen35/`; builds and caches
remain ignored in `work/`.
