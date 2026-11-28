# Qwen3.5 text-only AOTI / Native Runtime

This module owns the dense Qwen3.5 text architecture and the Qwen3.5-0.8B / 4B
export entry point. The [Qwen3.8 frontend](../qwen38/README.md) reuses its adapters,
state handling and export pipeline with its own loader and defaults. Python
prepares and compiles the text model; the shared native runtime loads the bundle
and generates tokens.
Vision and MTP are outside the supported scope. Defaults select 0.8B.

Use the [environment guide](../../environments/README.md) for setup/builds and
[development tools](../../docs/development-tools.md) for shared tooling.
[Validation status](../../docs/validation-status.md) distinguishes tested 0.8B
profiles from unresolved 4B and context-profile findings. Kernel support and
native parity alone do not establish full-model numerical acceptance.

## Prepare and export

Select `PY` using the environment guide and build `xinfer_attention_ops`,
`xinfer_deltanet_ops` and the native executables in `build/`. The
[L20 Docker guide](../../environments/l20/docker/README.md) provides a
containerized environment for full-model testing. Use sufficient GPU/host memory
for ordinary export and the independent validation stages.

```bash
"$PY" -m tools.prepare download --model Qwen/Qwen3.5-0.8B
"$PY" -m compiler.qwen35.export \
  --model-dir work/qwen35/0.8b/text --output artifacts/qwen35/0.8b/text
```

For 4B, keep the prepared checkpoint and compiled bundle separate:

```bash
"$PY" -m tools.prepare download --model Qwen/Qwen3.5-4B
"$PY" -m compiler.qwen35.export \
  --model-dir work/qwen35/4b/text --output artifacts/qwen35/4b/text
```

| Model | Pinned revision | Prepared directory |
| --- | --- | --- |
| Qwen/Qwen3.5-0.8B | `2fc06364715b967f1860aea9cf38778875588b17` | `work/qwen35/0.8b/text` |
| Qwen/Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `work/qwen35/4b/text` |

Preparation verifies source shards and extracts only language-model tensors into
`text.safetensors`, with `model_manifest.json`. Reuse prepared directories after
that one-time step. The Qwen3.5 loader accepts unquantized checkpoints with tied
embeddings and default RoPE. The [Qwen3.8 loader](../qwen38/README.md) owns FP8
checkpoint rules and preserves its untied BF16 output head.
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
SDPA backend selection unchanged and uses the explicit FlashAttention and
native DeltaNet adapters.
`--enable-inductor-optimization` optionally enables
`combo_kernels=True` and `benchmark_combo_kernel=True`; it is off by default.
Resolved compiler options are recorded in export logs. Validate each new
bundle because these options and autotuning can alter floating-point results.
The [package contract](../../docs/package-metadata.md) defines the deployment
bundle files and shared-weight ownership. There is no packing, attention-backend or
eager-numerics selector.

## Model interface and state

Both entries use the same explicit logits-only `ModelStep` interface:
`step(input_ids, cache_position, state, kv_lengths, query_lengths) -> last_logits`.
One bundle supports dynamic B=1..8. Activations and main BF16 weights are BF16,
with BF16 or FP8 projection storage according to the selected frontend;
checkpoint-native FP32 parameters (`linear_attn.A_log`, `linear_attn.norm.weight`),
RoPE frequencies and recurrent state use FP32. `C` is the compiled context capacity.

| Model | Layers | Conv/recurrent pairs | KV pairs | State tensors | Allocated KV shape |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.8B | 24 | 18 | 6 | 48 | `[8,2,C,256]` |
| 4B | 32 | 24 | 8 | 64 | `[8,4,C,256]` |
| 27B-FP8 | 64 | 48 | 16 | 128 | `[8,4,C,256]` |

IDs and positions are CUDA int64 `[B,T]`; both length vectors are CPU int32
`[B]`. Each row appends `query_lengths[row]` real tokens, ending at
`kv_lengths[row]`. Inputs are right-padded. The runtime allocates `1..8` state
rows (defaulting to 8; scalar CLI and benchmark allocate 1) and passes
active-prefix views (`narrow(0, 0, B)`). Both KV and DeltaNet convolution/recurrent
state are updated in place on caller-owned buffers; omitted tail rows and
inactive rows (`query_lengths[row] == 0`) remain unchanged. Logits select each
row's last real token; inactive rows return zero logits.

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
and bundling; Qwen3.5 bundles `libxinfer_deltanet_ops.so` alongside
`libxinfer_attention_ops.so` through the same AOTInductor C-shim packaging path.

## DeltaNet kernels

DeltaNet layers run through the native C++/CUDA custom operator
`xinfer::batched_deltanet` (`compiler/common/kernels/gated_delta_net/`, compiled into
`libxinfer_deltanet_ops.so` and invoked directly by AOTInductor through the C
shim `aoti_torch_cuda_batched_deltanet`). Moving dispatch into C++/CUDA removes
`torch.cond` from the exported graph and enables zero-copy in-place mutation of
`conv` and `state` so Qwen3.5 shares Qwen3's logits-only runtime ABI.

`compiler/common/adapters/gated_delta_net.py` connects the shared operator to
the Qwen HF module layout and explicit cache interface used by Qwen3.5 and
Qwen3.8. The Python entry point is
`batched_deltanet(qkv, b, a, conv, state, weight, A_log, dt_bias, cu_query=None, cuda_cu_query=None, *, query_lengths=None)`,
which mutates `conv` and `state` in place and returns `core` (`[B,T,Hv,128]`).

The standalone operator supports dynamic B=1..8. `cu_query`/`cuda_cu_query`
(or optional `query_lengths`) supplies each row's valid prefix in `0..T`;
omission means all `T` inputs are real. Padding produces zero core outputs and
does not advance convolution or recurrent state. Active-prefix views `[:B]`
leave omitted tail rows untouched. The full-model adapter shares `HybridCache`'s
cumulative length tensors across layers.

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

Inputs/parameters support matching BF16 or FP32, with decay parameters (`A_log`,
`dt_bias`) also accepted in FP32 and recurrent state always FP32. Gates and
recurrent state belong to value heads; grouped value heads reuse Q/K in HF
`repeat_interleave` order.

Execution dispatches natively on the active stream using CPU `cu_query` lengths
without device synchronization:
- **Short BF16 rows (`1..15`), SM < 8.0 GPUs, and FP32 inputs** run
  `deltanet_fused_kernel`, reading `conv` read-only and performing fused
  width-4 causal convolution + SiLU (retaining input-dtype rounding for
  convolution/SiLU and beta), gate preparation, Q/K L2 normalization, exact
  FP32 state recurrence, and output reduction while updating `state` in place
  (each `(32, 4)` thread block owns one `(batch, value_head, v_tile=32)` slice
  with zero local spills).
- **BF16 rows of length `>= 16` on SM 8.0+ GPUs** run
  `deltanet_chunked_bf16_kernel` (`chunk_size = 32`, zero global workspace,
  256 threads / 8 warps per `(batch, value_head)` CTA, zero local spills),
  keeping `S^T` (`[128, 128]` FP32) resident in warp registers across chunks
  while executing `BF16x3` split-precision Tensor Core MMAs (`m16n8k16` +
  `ldmatrix`), parallel `uint4` staging, register-cached `bf16x2` conv/norm,
  and 4-warp block-recursive exact FP32 triangular inversion. Mixed ragged
  batches dispatch short (`< 16`) and long (`>= 16`) rows independently so
  every row matches scalar `B=1` execution bit-for-bit.
- A lightweight trailing `update_conv_inplace_kernel` then shifts the final 4
  taps into `conv` after all blocks have finished reading it, eliminating
  cross-block read-after-write hazards for grouped Q/K channels.

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
Packing counts are printed during export, while detailed source/group mappings live in
`prefill.packing.json` / `decode.packing.json` diagnostics.

The GEMV pass targets five profiled 0.8B BF16 decoder geometries for dynamic
small-batch (`B=1..8`, `T=1`) decode graphs; prefill, the vocabulary head and
unsupported geometries remain on ordinary Inductor lowering. Its shared Triton
kernel loads each weight tile once across `B` rows and performs identical
per-row FP32 reductions using four rows/program for `K=1024`, one for
`K>=2048`, four warps and one stage. Preserve eligibility and alignment guards
when extending the pass.

The [GEMV case study](../../docs/experiments/qwen35-gemv-2026-11-13.md) explains
the measured bottleneck, target-dependent gains and FP-fusion caveat. Full-model
validation's eager stage uses the same graph builder and explicit passes;
Inductor code-generation options apply in the compiled stage. Hold packing
constant when comparing compiler configurations.

## Generate and validate

The commands below use 0.8B. For 4B, substitute its model and package directories
consistently; validation infers the prepared model from package identity.

```bash
"$PY" -m tools.prepare encode --model-dir work/qwen35/0.8b/text \
  --prompt "What is the capital of France?" --output work/qwen35-prompt.ids
build/xinfer_aoti artifacts/qwen35/0.8b/text \
  work/qwen35-prompt.ids work/qwen35-generated.ids --max-output-tokens 64
"$PY" -m tools.prepare decode --model-dir work/qwen35/0.8b/text \
  --input work/qwen35-generated.ids

"$PY" -m tools.validation.validate \
  --packages artifacts/qwen35/0.8b/text --output artifacts/qwen35/0.8b/validation-new
"$PY" -m tools.validation.qwen35.validate_batch \
  --packages artifacts/qwen35/0.8b/text --model-dir work/qwen35/0.8b/text \
  --output artifacts/qwen35/0.8b/batch-validation-new
"$PY" -m tools.validation.qwen35.validate_runtime \
  --packages artifacts/qwen35/0.8b/text --output artifacts/qwen35/0.8b/runtime-new
```

Use fresh output paths. Numerical validation and runtime/CLI checks are distinct;
see the [shared workflow and policy](../../docs/development-tools.md#python-validation-tools).
Benchmarks use the separate [shared tools](../../docs/development-tools.md#benchmark-hf-and-native-aoti).

## Focused tests

After the shared [CPU tests](../../docs/development-tools.md#cpu-unit-tests), run
these checks with CUDA visible and both `xinfer_attention_ops` and
`xinfer_deltanet_ops` built:

```bash
CUDA_VISIBLE_DEVICES='' "$PY" -m unittest compiler.common.tests.test_projection_gemv
"$PY" -m unittest compiler.common.tests.test_attention \
  compiler.common.tests.test_projection_gemv_cuda compiler.qwen35.tests.test_export
"$PY" -m unittest compiler.qwen35.tests.test_delta compiler.qwen35.tests.test_model \
  compiler.common.tests.gated_delta_net.test_recurrent \
  compiler.common.tests.gated_delta_net.test_batch
"$PY" -m unittest compiler.common.tests.gated_delta_net.test_aoti
```

Kernel tests and tensor-level HF references live in
`compiler/common/tests/gated_delta_net/` so Qwen3.5 and Qwen3.8 share one suite
covering all three value-head geometries (`16`, `32`, `48`).
Synthetic kernel tests cover BF16/FP32, carried/reset state, gate extremes,
partitioning, in-place buffer preservation, layouts and streams. Existing
kernel limits are 1% BF16 and 0.01% FP32, with a 1e-7 FP32 absolute-roundoff
allowance. Comparisons use matching input dtypes: BF16 kernel against HF BF16,
and FP32 kernel against HF FP32. BF16 acceptance has no upcast FP32 control;
recurrent state and internal FP32 accumulation remain as defined by the algorithm.
Output and carried-state bounds remain required independently of full-model results.

`test_export` checks the production graph (including zero `torch.cond` nodes and
in-place state mutation) and passes with a small random mixed model; it needs no
checkpoint or full-model AOTI compilation.
`test_aoti` compiles BF16/FP32 prefill/decode packages with bundled
`libxinfer_deltanet_ops.so` for all three geometries and requires exact
direct-kernel/Python AOTI core/state parity, including canonical singleton
strides, nondefault streams and active-prefix `B=1..8` views over 8-row state
buffers without modifying omitted tail rows. Temporary packages are cleaned up.
Standalone C++ batching evidence is recorded in
[validation status](../../docs/validation-status.md#current-validation); the
native proof harness remains scratch code.
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
