# Qwen3.5 text-only AOTI / C++

The same pipeline handles Qwen3.5-0.8B, 4B and Qwen3.8-27B-FP8: Python prepares
and compiles the text model; the shared C++ runtime loads the bundle and generates tokens.
Vision and MTP are outside the supported scope. Defaults select 0.8B.

Use the [environment guide](../../environments/README.md) for setup/builds and
[development tools](../../docs/development-tools.md) for shared tooling.
[Validation status](../../docs/validation-status.md) distinguishes tested 0.8B
profiles from unresolved 4B and context-profile findings. Kernel support and
native parity alone do not establish full-model numerical acceptance.

## Prepare and export

Select `Q35_PY` using the environment guide and build `xinfer_attention_ops` plus
the native executables in `build/qwen35/`. The
[L20 Docker guide](../../environments/l20/docker/README.md) provides an offline
environment for full-model testing. Use sufficient GPU/host memory for ordinary
export and the independent validation stages.

```bash
"$Q35_PY" -m tools.prepare download --model Qwen/Qwen3.5-0.8B
"$Q35_PY" -m compiler.qwen35.export \
  --model-dir work/qwen35/0.8b/text --output artifacts/qwen35/0.8b/text
```

For 4B, keep the prepared checkpoint and compiled bundle separate:

```bash
"$Q35_PY" -m tools.prepare download --model Qwen/Qwen3.5-4B
"$Q35_PY" -m compiler.qwen35.export \
  --model-dir work/qwen35/4b/text --output artifacts/qwen35/4b/text
```

| Model | Pinned revision | Prepared directory |
| --- | --- | --- |
| Qwen/Qwen3.5-0.8B | `2fc06364715b967f1860aea9cf38778875588b17` | `work/qwen35/0.8b/text` |
| Qwen/Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `work/qwen35/4b/text` |
| Qwen/Qwen3.8-27B-FP8 | `c3a462cc3d38cf351f3b26cbd763469eeb74ef81` | `work/qwen35/27b-fp8/text` |

Preparation verifies source shards and extracts only language-model tensors into
`text.safetensors`, with `model_manifest.json`. Reuse prepared directories after
that one-time step. BF16 checkpoints require tied embeddings; FP8 preserves the
untied BF16 output head. All supported checkpoints use default RoPE.
Dependency pins live in the per-device requirements files.
See [preparation](../../docs/development-tools.md#checkpoint-preparation-and-tokenization)
for verification, offline mode and tokenization.

One invocation exports both entries to a new or empty directory. Omitted
`--max-context-length` allocates the selected config's full 262,144-position
ceiling. Select a smaller capacity explicitly, for example
`--max-context-length 16384`; valid capacities are 512 through that ceiling.
Both entries must be re-exported to change capacity.
Capacity is per row, and the runtime reserves eight rows even for B=1. At the
0.8B config's full ceiling, KV storage alone requires 24 GiB; select a smaller
capacity for L20, such as the 1,024-slot validation profile.

Export uses ordinary strict tracing, linear packing, BF16 cast emulation and
TF32 disabled. `configure()` sets one CPU thread and disables TF32; export leaves
SDPA backend selection unchanged and uses the explicit FlashAttention adapter.
`--enable-inductor-optimization` optionally enables
`combo_kernels=True` and `benchmark_combo_kernel=True`; it is off by default.
Resolved compiler options are recorded in export logs. Validate each new
bundle because these options and autotuning can alter floating-point results.
The [package contract](../../docs/package-metadata.md) defines the five deployment
files and shared-weight ownership. There is no packing, attention-backend or
eager-numerics selector.

## Qwen3.8-27B-FP8

Qwen3.8 uses the same dense Qwen3.5 text architecture and native model ABI.
This is an experimental integration: the GB10 bundle runs natively and passes
native parity, but its full-model HF logit comparisons currently exceed the
required limits. See [validation status](../../docs/validation-status.md).
Select the independent `qwen38` environment and `build/qwen38/` using the
[environment guide](../../environments/README.md), then prepare and export with
the shared compiler:

```bash
export XINFER_ATTENTION_OPS_LIBRARY="$Q38_BUILD/libxinfer_attention_ops.so"
"$Q38_PY" -m tools.prepare download --model Qwen/Qwen3.8-27B-FP8
"$Q38_PY" -m compiler.qwen35.export \
  --model-dir work/qwen35/27b-fp8/text --output artifacts/qwen38-fp8-c1024 \
  --max-context-length 1024
```

The text checkpoint has 400 E4M3 FP8 projections with BF16 128x128 weight
scales and dynamic per-token, per-128-value activation quantization. Other
parameters and activations remain BF16; recurrent state stays FP32. Preparation
retains the separate output head and `generation_config.json`, including both
EOS IDs. Vision and MTP weights are excluded.

HF eager uses Transformers' original `FP8Linear` with
[`kernels-community/finegrained-fp8`](https://huggingface.co/kernels/kernels-community/finegrained-fp8)
loaded with `version=4`. Online loading resolves the current `v4` branch;
offline loading uses its cached revision. Validate after refreshing the kernel
cache because version 4 can receive implementation updates. Export wraps the same upstream
Triton kernel in `torch.library.triton_op`; a fixed 32-row launch tile supports
symbolic batch/token dimensions. AOTI packages the kernels, so C++ inference
needs neither Python nor a Hub connection. Before offline export, fetch the
version branch explicitly so the cache includes both its files and `v4` reference:

```bash
export KERNELS_CACHE="$PWD/work/qwen38/kernel-cache"
"$Q38_PY" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    "kernels-community/finegrained-fp8", repo_type="kernel", revision="v4",
    allow_patterns=["build/torch-cuda/*"], cache_dir=os.environ["KERNELS_CACHE"],
)
PY
```

For a disconnected target, transfer this cache from a connected host and set
`KERNELS_CACHE` there before enabling `HF_HUB_OFFLINE=1`. FP8 operators are not
BF16 linear-packing candidates; remaining BF16 projections use the existing passes.

This model has 64 layers, 48 DeltaNet layers with 48 value heads, and 16 full
attention layers with 24 query heads and four KV heads. Its 128 state tensors
reserve eight rows as usual. Text weights occupy about 27.44 GiB; state at
1,024-token capacity adds about 1.65 GiB. The config's default 262,144-token
capacity requires about 129 GiB of state alone: select capacity explicitly.

Validation compares the unmodified **FP8 HF checkpoint** against adapted eager,
Python AOTI and C++ AOTI. It keeps the existing logit limits and strict native
parity; it does not measure quantization loss against unquantized model weights.
The reference protocol records FP8 storage and the kernel version. Run
`compiler.qwen35.tests.test_fp8_adapter` for checkpoint/projection AOTI coverage and
the 27B classes in `test_kernels` and `test_kernels_aoti` for DeltaNet coverage.
See [validation status](../../docs/validation-status.md) for tested full-model scope.

## Model interface and state

Both entries use the same explicit `ModelStep` interface:
`step(input_ids, cache_position, state, kv_lengths, query_lengths) -> last_logits, *next_state`.
One bundle supports dynamic B=1..8. Activations are BF16, with BF16 or FP8
projection storage as described above; RoPE frequencies and recurrent state
use FP32. `C` is the compiled context capacity.

| Model | Layers | Conv/recurrent pairs | KV pairs | State tensors | Allocated KV shape |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.8B | 24 | 18 | 6 | 48 | `[8,2,C,256]` |
| 4B | 32 | 24 | 8 | 64 | `[8,4,C,256]` |
| 27B-FP8 | 64 | 48 | 16 | 128 | `[8,4,C,256]` |

IDs and positions are CUDA int64 `[B,T]`; both length vectors are CPU int32
`[B]`. Each row appends `query_lengths[row]` real tokens, ending at
`kv_lengths[row]`. Inputs are right-padded. The runtime allocates all eight state
rows and passes active-prefix views. KV updates only real appended positions;
DeltaNet returns fresh convolution/recurrent state for active rows. Omitted rows
retain their state. Logits select each row's last real token; inactive rows
return zero logits and leave all state unchanged.

Prefill accepts padded widths 2–512; decode accepts width one. Scalar generation
uses B=1 and chunks longer prompts, with prompt and output sharing capacity.
The [dynamic batch contract](../../docs/package-metadata.md#dynamic-batch-interfaces)
defines admission, ownership and the compiled calling convention.
The text-only RoPE calculation matches HF bitwise while avoiding its redundant
cross-grid slice assignments and their singleton-batch export guard.

Linear packing remains enabled; [projection passes](#projection-passes) describes
the dynamic graph's GEMV limitation.
The [validation status](../../docs/validation-status.md) distinguishes functional
batch support from numerical acceptance and untested model/device profiles.

Full-attention layers preserve HF projections, Q/K normalization, RoPE and sigmoid
output gating around the shared read-only FlashAttention operator. Query/KV
heads are 8/2 for 0.8B and 16/4 for 4B, with head dimension 256. Attention reads
only the populated prefix and handles grouped heads without expansion. See the
[shared attention guide](../../docs/flash-attention.md) for operator constraints
and bundling; DeltaNet kernels are packaged by AOTInductor as well.

## DeltaNet kernels

One shape-based dispatcher selects fused recurrence for `T <= 97` and FP32
chunked prefill for longer calls. Both branches remain in `torch.cond` through
strict export and use `torch.library.triton_op` / `wrap_triton`, so AOTInductor
packages their kernels without a separate CUDA-driver launcher or Python runtime.
The threshold is provisional: an isolated-kernel crossover does not establish
an optimal full-model threshold on every device.

The operator is
`fused(qkv, b, a, conv, state, weight, A_log, dt_bias, query_lengths=None)`
returns `core, next_conv, next_recurrent`.

The standalone kernel supports dynamic B=1..8. Optional CUDA int32 `[B]`
`query_lengths` supplies each row's valid prefix in 0..T; omission means all T
inputs are real. Padding produces zero core outputs and does not advance
convolution or recurrent state. Dispatch uses the padded width T, so shorter
rows in a long batch also use the chunked path. Both state outputs are fresh;
inputs remain unchanged. The full-model adapters supply these per-row lengths.

All geometries use 16 key heads, key/value dimension 128 and width-four bias-free
SiLU convolution. Value-head count is 16 for 0.8B, 32 for 4B and 48 for 27B-FP8:

| Tensor | 0.8B | 4B | 27B-FP8 |
| --- | --- | --- | --- |
| QKV | `[B,T,6144]` | `[B,T,8192]` | `[B,T,10240]` |
| A/B gates | `[B,T,16]` | `[B,T,32]` | `[B,T,48]` |
| Convolution state | `[B,6144,4]` | `[B,8192,4]` | `[B,10240,4]` |
| Convolution weights | `[6144,4]` | `[8192,4]` | `[10240,4]` |
| Decay parameters | `[16]` | `[32]` | `[48]` |
| Recurrent state | `[B,16,128,128]` | `[B,32,128,128]` | `[B,48,128,128]` |
| Core output | `[B,T,16,128]` | `[B,T,32,128]` | `[B,T,48,128]` |

Inputs/parameters support matching BF16 or FP32, with recurrent state always
FP32. Gates and recurrent state belong to value heads; grouped value heads reuse
Q/K in HF `repeat_interleave` order. One writer per Q/K group updates convolution
history, avoiding duplicate writes for grouped value heads.

Fused recurrence combines convolution/history, SiLU, gate preparation, Q/K
normalization, state update and output reduction. Each program owns one value
head and 32 value columns. Long prefill prepares tokens/heads in parallel, then
runs cumulative decay/triangular solve, U/W transforms, inter-chunk state
propagation and parallel output kernels. Chunk size is 64; the U/W output tile
is 32 while its dot reduction spans all 64 rows. Masked neutral tails preserve
one dynamic package. Workspaces and matrix products use FP32 IEEE arithmetic;
convolution/SiLU and beta retain input-dtype rounding. Only the long branch
allocates chunk workspaces.

Input projections, gated normalization and output projection remain outside the
core. Kernel/block checks therefore remain necessary alongside full-model tests.

## Projection passes

Export applies `compiler/common/passes/pack_linears.py` after strict export and
before decomposition. It combines bias-free linears sharing one input when their
weights are frozen, exclusively stored and have no other consumers. Shared,
tied or trainable weights are skipped. Packed parameters are created once;
there is no runtime weight concatenation.

For 0.8B, 48 groups reduce 187 linear calls to 97 across DeltaNet inputs, MLP
gate/up and attention QKV. Outputs retain BF16 and contiguous layouts. Packing
reduces launches, not weight bytes; output copies can cost time during prefill.
The policy version is in metadata, while detailed source/group mappings live in
`prefill.packing.json` / `decode.packing.json` diagnostics.

The existing GEMV pass targets five profiled 0.8B singleton BF16 projection
geometries. It does not match dynamic-batch graphs, whose linear operations
remain eligible for ordinary Inductor lowering. Its shared Triton kernel's FP32
accumulation uses four rows/program for `K=1024`, one for `K>=2048`, four warps
and one stage. The vocabulary head and unsupported geometries keep the existing
linear implementation. Preserve
eligibility and alignment guards when extending the pass.

The [GEMV case study](../../docs/experiments/qwen35-gemv-2026-11-13.md) explains
the measured bottleneck, target-dependent gains and FP-fusion caveat. Full-model
validation's eager stage uses the same graph builder and explicit passes;
Inductor code-generation options apply in the compiled stage. Hold packing
constant when comparing compiler configurations.

## Generate and validate

The commands below use 0.8B. For 4B, substitute its model and package directories
consistently; validation infers the prepared model from package identity.

```bash
"$Q35_PY" -m tools.prepare encode --model-dir work/qwen35/0.8b/text \
  --prompt "What is the capital of France?" --output work/qwen35-prompt.ids
build/qwen35/xinfer_aoti artifacts/qwen35/0.8b/text \
  work/qwen35-prompt.ids work/qwen35-generated.ids --max-output-tokens 64
"$Q35_PY" -m tools.prepare decode --model-dir work/qwen35/0.8b/text \
  --input work/qwen35-generated.ids

"$Q35_PY" -m tools.validation.validate \
  --packages artifacts/qwen35/0.8b/text --output artifacts/qwen35/0.8b/validation-new
"$Q35_PY" -m tools.validation.qwen35.validate_batch \
  --packages artifacts/qwen35/0.8b/text --model-dir work/qwen35/0.8b/text \
  --output artifacts/qwen35/0.8b/batch-validation-new
"$Q35_PY" -m tools.validation.qwen35.validate_runtime \
  --packages artifacts/qwen35/0.8b/text --output artifacts/qwen35/0.8b/runtime-new
```

Use fresh output paths. Numerical validation and runtime/CLI checks are distinct;
see the [shared workflow and policy](../../docs/development-tools.md#python-validation-tools).
Benchmarks use the separate [shared tools](../../docs/development-tools.md#benchmark-hf-and-native-aoti).

## Focused tests

After the shared [CPU tests](../../docs/development-tools.md#cpu-unit-tests), run
these checks with CUDA visible and the native attention library built:

```bash
CUDA_VISIBLE_DEVICES='' "$Q35_PY" -m unittest compiler.common.tests.test_projection_gemv
"$Q35_PY" -m unittest compiler.common.tests.test_attention \
  compiler.common.tests.test_projection_gemv_cuda compiler.qwen35.tests.test_export
"$Q35_PY" -m unittest compiler.qwen35.tests.test_delta compiler.qwen35.tests.test_kernels \
  compiler.qwen35.tests.test_batch compiler.qwen35.tests.test_chunked
"$Q35_PY" -m unittest compiler.qwen35.tests.test_kernels_aoti
```

Synthetic kernel tests cover all three geometries, BF16/FP32, carried/reset state,
gate extremes, partitioning, dispatch boundaries, layouts and streams. Existing
kernel limits are 1% BF16 and 0.01% FP32, with a 1e-7 FP32 absolute-roundoff
allowance. Comparisons use matching input dtypes: BF16 kernel against HF BF16,
and FP32 kernel against HF FP32. BF16 acceptance has no upcast FP32 control;
recurrent state and internal FP32 accumulation remain as defined by the algorithm.
Output and carried-state bounds remain required independently of full-model results.

`test_export` checks the production graph and passes with a small random mixed
model; it needs no checkpoint or full-model AOTI compilation.
`test_kernels_aoti` compiles BF16/FP32 prefill/decode for all three geometries and
requires exact direct-kernel/Python AOTI core/state parity, including canonical
singleton strides and nondefault streams. Its batched 0.8B tests compile one
prefill/decode pair per dtype for B=1..8, check the 97/98 dispatch boundary,
carried/inactive rows, and reject B=0/9 with runtime input guards enabled.
Temporary packages are cleaned up. Standalone C++ batching evidence is recorded
in [validation status](../../docs/validation-status.md#current-validation);
the native proof harness remains scratch code.
GEMV's packaging tests have the separate FP32 fusion qualification in the case
study. Run affected component tests separately from full-model validation.

## Numerical considerations

**Q35-NUM-001** identified a lost BF16 rounding boundary before gated RMSNorm
in a Torch 2.7 layer-0 probe. The BF16 core output matched eager, but conversion
back to FP32 used unrounded producer values: all 2,048 converted elements differed,
and rounding the compiled values through BF16 restored equality. This persisted
with cast emulation and pattern matching disabled. Native reduction fallbacks
did not remove the output difference; an experimental native-cast workaround
failed to load and was not adopted. This identifies one contributor, not every
full-model discrepancy.

**Q35-NUM-002** showed full-prompt and partitioned HF selecting different greedy
tokens at a zero-margin tie. This cannot be attributed solely to compilation.
Use matched partitions for numerical attribution and retain independent greedy
sequences/margins as diagnostics. Do not infer FP32 accuracy from a matching
argmax. [Validation status](../../docs/validation-status.md) records the tested
profiles and remaining logit failures.
