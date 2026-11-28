# Qwen3.5 text-only AOTI / Native Runtime

Owns the dense Qwen3.5 text architecture and Qwen3.5-0.8B / 4B export entry
point (defaults select 0.8B; vision and MTP are excluded).
[Qwen3.8](../qwen38/README.md) reuses these adapters, state handling and export
pipeline with its own FP8 loader and defaults. See
[environments](../../environments/README.md),
[development tools](../../docs/development-tools.md) and
[validation status](../../docs/validation-status.md).

## Prepare and export

Build `xinfer_attention_ops`, `xinfer_deltanet_ops` and the native executables in
`build/` first (or use the [L20 Docker workflow](../../environments/l20/docker/README.md)):

```bash
"$PY" -m tools.prepare download --model Qwen/Qwen3.5-0.8B
"$PY" -m compiler.qwen35.export   --model-dir work/qwen35/0.8b/text --output artifacts/qwen35/0.8b/text

# For 4B, keep checkpoint and bundle directories separate:
"$PY" -m tools.prepare download --model Qwen/Qwen3.5-4B
"$PY" -m compiler.qwen35.export   --model-dir work/qwen35/4b/text --output artifacts/qwen35/4b/text
```

| Model | Pinned revision | Prepared directory |
| --- | --- | --- |
| Qwen/Qwen3.5-0.8B | `2fc06364715b967f1860aea9cf38778875588b17` | `work/qwen35/0.8b/text` |
| Qwen/Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `work/qwen35/4b/text` |

Preparation extracts text-only weights into `text.safetensors` with
`model_manifest.json` (unquantized checkpoints with tied embeddings and default
RoPE; see [preparation](../../docs/development-tools.md#checkpoint-preparation-and-tokenization)).

One invocation compiles both entries into a fresh directory.
`--max-context-length` accepts `512` through the model's 262,144 ceiling
(omitting it allocates the full ceiling — e.g., 24 GiB of KV state alone for 8
rows on 0.8B — so pass an explicit capacity such as `--max-context-length 16384`
or `1024`). Capacity is per row; the package ceiling is 8 rows while `Session`
allocates `1..8` rows (`1` for scalar `xinfer_aoti`/`xinfer_benchmark`).

Export uses strict tracing, linear packing, profiled small-batch decode GEMV
replacement, BF16 cast emulation and TF32 disabled.
`--enable-inductor-optimization` optionally enables `combo_kernels=True` and
`benchmark_combo_kernel=True` (off by default). See
[package metadata](../../docs/package-metadata.md) for bundle files and
shared-weight ownership.

## Model interface and state

Both entries expose the unified logits-only in-place interface
`step(input_ids, cache_position, state, kv_lengths, query_lengths) -> last_logits`
for dynamic `B=1..8`. Main weights and activations are BF16 (or FP8 projections
for Qwen3.8); checkpoint-native FP32 parameters (`linear_attn.A_log`,
`linear_attn.norm.weight`), RoPE frequencies and recurrent state remain FP32.

| Model | Layers | Conv/recurrent pairs | KV pairs | State tensors | Package KV shape |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.8B | 24 | 18 | 6 | 48 | `[8,2,C,256]` |
| 4B | 32 | 24 | 8 | 64 | `[8,4,C,256]` |
| 27B-FP8 | 64 | 48 | 16 | 128 | `[8,4,C,256]` |

See [dynamic batch interfaces](../../docs/package-metadata.md#dynamic-batch-interfaces)
for lengths, padding, active views and state ownership. Text-only RoPE matches
HF bitwise without HF's cross-grid slice assignments that introduce a `B >= 2`
guard. Linear packing and profiled small-batch decode GEMV replacement are
detailed in [projection passes](#projection-passes).

Full-attention layers preserve HF projections, Q/K norm, RoPE and sigmoid output
gating around `xinfer::batched_flash_attention` (query/KV heads: `8/2` for 0.8B,
`16/4` for 4B; head dim `256`). Both `libxinfer_attention_ops.so` and
`libxinfer_deltanet_ops.so` are embedded in each `.pt2` via AOTInductor C shims;
see [FlashAttention](../../docs/flash-attention.md).

## DeltaNet kernels

DeltaNet layers invoke the native C++/CUDA custom operator
`xinfer::batched_deltanet` (`compiler/common/kernels/gated_delta_net/`,
`libxinfer_deltanet_ops.so`, C shim `aoti_torch_cuda_batched_deltanet`). Native
dispatch removes `torch.cond` from the exported graph and mutates `conv` and
`state` in place.

`compiler/common/adapters/gated_delta_net.py` connects the operator to Qwen3.5
and Qwen3.8 via `batched_deltanet(qkv, b, a, conv, state, weight, A_log, dt_bias, cu_query=None, cuda_cu_query=None, *, query_lengths=None)`,
mutating `conv` and `state` in place and returning `core` (`[B,T,Hv,128]`).
Padded tokens (`>= query_lengths[row]`) yield zero outputs without advancing
state; omitted tail rows remain untouched.

All geometries use 16 key heads, head dim 128 and width-4 bias-free SiLU
convolution (`Hv` = 16 for 0.8B, 32 for 4B, 48 for 27B-FP8):

| Tensor | 0.8B | 4B | 27B-FP8 |
| --- | --- | --- | --- |
| QKV | `[B,T,6144]` | `[B,T,8192]` | `[B,T,10240]` |
| A/B gates | `[B,T,16]` | `[B,T,32]` | `[B,T,48]` |
| Convolution state | `[B,6144,4]` | `[B,8192,4]` | `[B,10240,4]` |
| Convolution weights | `[6144,4]` | `[8192,4]` | `[10240,4]` |
| Decay parameters | `[16]` | `[32]` | `[48]` |
| Recurrent state | `[B,16,128,128]` | `[B,32,128,128]` | `[B,48,128,128]` |
| Core output | `[B,T,16,128]` | `[B,T,32,128]` | `[B,T,48,128]` |

Inputs accept matching BF16 or FP32 (`A_log`/`dt_bias` also accepted in FP32;
recurrent state is always FP32). Dispatch runs on the active CUDA stream from
CPU `cu_query` with no device sync:

- **Short BF16 rows (`1..15`), SM < 8.0, and FP32 inputs:** `deltanet_fused_kernel`
  (fused width-4 causal conv + SiLU, gates, Q/K L2 norm, exact FP32 recurrence,
  `(32, 4)` thread block per `(batch, Hv, v_tile=32)` with zero local spills).
- **BF16 rows `>= 16` on SM 8.0+:** `deltanet_chunked_bf16_kernel` (`chunk_size=32`,
  zero global workspace, 256 threads / 8 warps per `(batch, Hv)` CTA, `S^T` in
  registers, `BF16x3` split-precision Tensor Core MMAs `m16n8k16` + `ldmatrix`,
  4-warp block-recursive exact FP32 triangular inversion). Ragged batches
  dispatch short and long rows independently for bitwise scalar parity.
- **Trailing conv shift:** `update_conv_inplace_kernel` shifts the final 4 taps
  into `conv` after all blocks finish reading it, avoiding cross-block RAW
  hazards on grouped Q/K channels.

## Projection passes

- **Linear packing (`compiler/common/passes/pack_linears.py`):** combines
  bias-free frozen linears sharing an input (skipping tied/shared/trainable
  weights). On 0.8B, 48 groups reduce 187 linear calls to 97 across DeltaNet
  inputs, MLP gate/up and attention QKV. Packing maps are written to
  `{prefill,decode}.packing.json`.
- **Decode GEMV (`compiler/common/passes/projection_gemv.py`):** replaces five
  profiled 0.8B BF16 decoder linear geometries on dynamic small-batch
  (`B=1..8`, `T=1`) decode graphs with a packaged Triton kernel (4 rows/program
  for `K=1024`, 1 for `K>=2048`, 4 warps, 1 stage, FP32 accumulation). Prefill,
  the vocabulary head and unprofiled geometries stay on standard Inductor
  lowering. Because eager Triton flags such as `enable_fp_fusion` may not
  propagate identically through AOTI, verify compiled `.pt2` numerics after any
  kernel change.

## Generate and validate

Commands below use 0.8B (substitute 4B paths consistently):

```bash
"$PY" -m tools.prepare encode --model-dir work/qwen35/0.8b/text   --prompt "What is the capital of France?" --output work/qwen35-prompt.ids
build/xinfer_aoti artifacts/qwen35/0.8b/text   work/qwen35-prompt.ids work/qwen35-generated.ids --max-output-tokens 64
"$PY" -m tools.prepare decode --model-dir work/qwen35/0.8b/text   --input work/qwen35-generated.ids

build/xinfer_serve artifacts/qwen35/0.8b/text --port 8000

"$PY" -m tools.validation.validate   --packages artifacts/qwen35/0.8b/text --output artifacts/qwen35/0.8b/validation-new
"$PY" -m tools.validation.qwen35.validate_batch   --packages artifacts/qwen35/0.8b/text --model-dir work/qwen35/0.8b/text   --output artifacts/qwen35/0.8b/batch-validation-new
"$PY" -m tools.validation.qwen35.validate_runtime   --packages artifacts/qwen35/0.8b/text --output artifacts/qwen35/0.8b/runtime-new
```

See [development tools](../../docs/development-tools.md#python-validation-tools)
for validation policy and
[benchmarks](../../docs/development-tools.md#benchmark-hf-and-native-aoti).

## Focused tests

After [CPU unit tests](../../docs/development-tools.md#cpu-unit-tests), run GPU
checks with both `xinfer_attention_ops` and `xinfer_deltanet_ops` built:

```bash
CUDA_VISIBLE_DEVICES='' "$PY" -m unittest compiler.common.tests.test_projection_gemv
"$PY" -m unittest compiler.common.tests.test_attention   compiler.common.tests.test_projection_gemv_cuda compiler.qwen35.tests.test_export
"$PY" -m unittest compiler.qwen35.tests.test_delta compiler.qwen35.tests.test_model   compiler.common.tests.gated_delta_net.test_recurrent   compiler.common.tests.gated_delta_net.test_batch
"$PY" -m unittest compiler.common.tests.gated_delta_net.test_aoti
```

`compiler/common/tests/gated_delta_net/` covers all three value-head geometries
(`16`, `32`, `48`) with matching input dtypes (`<= 1%` BF16, `<= 0.01%` FP32
with `1e-7` absolute floor, no upcast FP32 control for BF16). `test_export`
verifies production graph structure (zero `torch.cond`, in-place state) with
random weights; `test_aoti` verifies direct-kernel vs Python AOTI parity across
all three geometries and active-prefix `B=1..8` views.

## Numerical considerations

- **Q35-NUM-001:** a Torch 2.7 layer-0 probe isolated a lost BF16 rounding
  boundary before gated RMSNorm (FP32 conversion read unrounded producer values;
  explicit BF16 rounding restored equality). This persisted with cast emulation
  and pattern matching disabled and remains to be rechecked on the current
  toolchain.
- **Q35-NUM-002:** full-prompt and partitioned HF can select different greedy
  tokens at zero-margin ties; use matched partitions for compiler attribution
  and retain independent greedy trajectories as diagnostics. See
  [validation status](../../docs/validation-status.md).
