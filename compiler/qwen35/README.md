# Qwen3.5 text-only AOTI / C++

The same pipeline handles Qwen3.5-0.8B and 4B: Python prepares and compiles the
text model; the shared C++ runtime loads the bundle and generates tokens.
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

Preparation verifies source shards and extracts only language-model tensors into
`text.safetensors`, with `model_manifest.json`. Reuse prepared directories after
that one-time step. Both checkpoints require tied embeddings and default RoPE.
The model checks its pinned Transformers version and requires `fla` and
`causal_conv1d` to be absent so HF reference operations use the tested tensor path.
See [preparation](../../docs/development-tools.md#checkpoint-preparation-and-tokenization)
for verification, offline mode and tokenization.

One invocation exports both entries to a new or empty directory. Omitted
`--max-context-length` allocates the selected config's full 262,144-position
ceiling. Select a smaller capacity explicitly, for example
`--max-context-length 16384`; valid capacities are 512 through that ceiling.
Both entries must be re-exported to change capacity.

Export uses ordinary strict tracing, linear packing, BF16 cast emulation and
TF32 disabled. `--enable-inductor-optimization` optionally enables
`combo_kernels=True` and `benchmark_combo_kernel=True`; it is off by default.
The flag and resolved options are recorded in export logs. Validate each new
bundle because these options and autotuning can alter floating-point results.
The [package contract](../../docs/package-metadata.md) defines the five deployment
files and shared-weight ownership. There is no packing, attention-backend or
eager-numerics selector.

## Model interface and state

Both entries use the same explicit interface:
`step(input_ids, cache_position, state, kv_extent) -> last_logits, *next_state`.
Batch size is one, weights/activations are BF16, and RoPE frequencies and recurrent
state use FP32. `C` below is the compiled context capacity.

| Model | Layers | Conv/recurrent pairs | KV pairs | State tensors | KV shape |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.8B | 24 | 18 | 6 | 48 | `[1,2,C,256]` |
| 4B | 32 | 24 | 8 | 64 | `[1,4,C,256]` |

KV inputs are caller-owned buffers updated only at appended positions. Returned
KV tensors alias those buffers. Conv/recurrent inputs remain unchanged; fresh
returned tensors replace the previous state. There is no per-token full-cache
copy. The common state policy handles both update rules; the model runtime header
checks the supported ABI.

Inputs are unpadded contiguous appends. Prefill accepts 2–512 tokens per call;
decode accepts one, including at position zero. Shared generation chunks longer
prompts, with prompt and output sharing the bundle capacity. The shape-only CPU
`kv_extent` identifies the populated prefix without reading a CUDA scalar.
See [runtime ABI 1](../../docs/package-metadata.md#runtime-abi-1).

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
`fused(qkv, b, a, conv, state, weight, A_log, dt_bias) -> core, next_conv, next_recurrent`.
Both geometries use 16 key heads, key/value dimension 128 and width-four bias-free
SiLU convolution. Value-head count is 16 for 0.8B and 32 for 4B:

| Tensor | 0.8B | 4B |
| --- | --- | --- |
| QKV | `[1,T,6144]` | `[1,T,8192]` |
| A/B gates | `[1,T,16]` | `[1,T,32]` |
| Convolution state | `[1,6144,4]` | `[1,8192,4]` |
| Convolution weights | `[6144,4]` | `[8192,4]` |
| Decay parameters | `[16]` | `[32]` |
| Recurrent state | `[1,16,128,128]` | `[1,32,128,128]` |
| Core output | `[1,T,16,128]` | `[1,T,32,128]` |

Inputs/parameters support matching BF16 or FP32, with recurrent state always
FP32. Gates and recurrent state belong to value heads; grouped value heads reuse
Q/K in HF `repeat_interleave` order. One writer per Q/K group updates convolution
history, avoiding duplicate writes for 4B's grouped heads.

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
export logs and `prefill.packing.json` / `decode.packing.json` diagnostics.

Decode then replaces the five profiled 0.8B singleton BF16 projection geometries
with a shared Triton GEMV operator. FP32 accumulation uses four rows/program for
`K=1024`, one for `K>=2048`, four warps and one stage. The vocabulary head and
unsupported geometries keep the existing linear implementation. Preserve
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
"$Q35_PY" -m unittest compiler.qwen35.tests.test_delta compiler.qwen35.tests.test_kernels
"$Q35_PY" -m unittest compiler.qwen35.tests.test_kernels_aoti
```

Synthetic kernel tests cover both geometries, BF16/FP32, carried/reset state,
gate extremes, partitioning, dispatch boundaries, layouts and streams. Existing
kernel limits are 1% BF16 and 0.01% FP32, with a 1e-7 FP32 absolute-roundoff
allowance. HF and independent FP32 controls remain required; known failures are
not waived by a full-model pass.

`test_export` checks the production graph and passes with a small random mixed
model; it needs no checkpoint or full-model AOTI compilation.
`test_kernels_aoti` compiles BF16/FP32 prefill/decode for both geometries and
requires exact direct-kernel/Python AOTI core/state parity, including canonical
singleton strides and nondefault streams. Temporary packages are cleaned up.
There is no kernel C++ runner; full-model validation supplies native coverage.
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
full-model discrepancy. Reproduce against the selected stack with:

```bash
"$Q35_PY" -m tools.diagnostics.qwen35.probe_rounding \
  --model-dir work/qwen35/0.8b/text --output artifacts/qwen35/rounding-probe-new
```

Exit zero means reproduced, not fixed. The probe exposes intermediate outputs
only in its diagnostic graph and uses the prepared full text checkpoint.

**Q35-NUM-002** showed full-prompt and partitioned HF selecting different greedy
tokens at a zero-margin tie. This cannot be attributed solely to compilation.
Use matched partitions for numerical attribution and retain independent greedy
sequences/margins as diagnostics. Do not infer FP32 accuracy from a matching
argmax. [Validation status](../../docs/validation-status.md) records the tested
profiles, remaining logit failures and historical sources.
