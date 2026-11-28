# Shared valid-prefix FlashAttention

All supported models use native batched variable-length FlashAttention under
`compiler/common/kernels/flash_attention/`. Model adapters preserve HF
projections, normalization and RoPE (plus output gating in Qwen3.5/Qwen3.8).

## Compilation and runtime contract

Adapters update KV caches in place via `xinfer::append_kv_batched` (a
`torch.library.triton_op` packaged directly by AOTI; only real tokens are
written), then call read-only `xinfer::batched_flash_attention` over the
populated prefix. The operator returns a fresh contiguous output without
mutating or aliasing cache inputs, calling ATen FlashAttention with native GQA
and lower-right causality (no `flash-attn` pip package required; unused cache
slots are not read).

- **Geometry & bounds:** derived from input tensors. Query heads must be a
  positive multiple of KV heads; head dimension must be a multiple of 8 in
  `8..256`. Exports use BF16, 1–512 padded tokens per call, dynamic `B=1..8`,
  and `backed_size_oblivious=True`. Model-specific details live in the
  [Qwen3](../compiler/qwen3/README.md#model-interface) and
  [Qwen3.5](../compiler/qwen35/README.md#model-interface-and-state) guides.
- **Restrictions:** adapters reject arbitrary masks, training, attention-weight
  outputs and unsupported cache/layer types. Load HF weights before installing
  adapters; adapters are not general HF `generate()` replacements.
- **Batched execution:** `xinfer::batched_flash_attention` packs real queries,
  invokes variable-length FlashAttention once per batch, and scatters results
  into zero-padded output. Cumulative query/KV lengths are derived once per step
  from CPU int32 `[B]` lengths with no CUDA readback. Each request's cache is
  viewed as a single block when capacity is divisible by 256; otherwise valid
  prefixes are packed. The wrapper fixes `num_splits=1`, normalizes singleton
  block-table strides, and uses the general variable-length path for mixed
  inactive/decode rows (avoiding ATen's one-query GQA shortcut assumption).
  See [dynamic batch interfaces](package-metadata.md#dynamic-batch-interfaces)
  for state allocation and views.

## Build and deployment

Build the compiler-owned `xinfer_attention_ops` target before export or
standalone attention tests using the
[environment build commands](../environments/README.md#build-compiler-kernels-and-native-runtimes).
This writes `build/libxinfer_attention_ops.so` (override with
`XINFER_ATTENTION_OPS_LIBRARY`). Runtime executables build independently.

Exporters pass C shim declarations from
`compiler/common/kernels/flash_attention/attention.py` into
`compiler/common/compile.py` (`aot_inductor.custom_ops_to_c_shims` and
`custom_op_libs`). On Linux GCC, packaging temporarily sets `LIBRARY_PATH` and
`LD_RUN_PATH='$ORIGIN'` (restored afterward; run concurrent exports in
separate processes) and embeds `libxinfer_attention_ops.so` in each `.pt2`.

Standard Python and native AOTI loaders extract the library beside the wrapper
so the linker loads it via `$ORIGIN`. Native executables neither link nor
install a separate attention library. Deploy the complete
[deployment bundle](package-metadata.md) with matching LibTorch/CUDA.

Eager validation extracts the prefill package's library before installing
adapters (or reuses an already registered operator); runtime validators rely on
standard AOTI loading. Package checksums cover embedded libraries; independent
prefill/decode attention implementations remain future work because both entries
currently share one SONAME, C-shim symbol and Torch operator registration.

## Verification

With `libxinfer_attention_ops.so` built, run the operator and model adapter
tests listed in the [model guides](../README.md#getting-started). Full-model
validation separately checks `HF → adapted/optimized eager → Python AOTI →
native AOTI (cpp_aoti)` under the
[shared numerical policy](development-tools.md#shared-numerical-policy); see
[validation status](validation-status.md).
