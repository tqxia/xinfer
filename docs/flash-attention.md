# Shared valid-prefix FlashAttention

Qwen3 and Qwen3.5 full-attention layers use native FlashAttention operators
implemented under `compiler/common/kernels/flash_attention/`. Both model families use batched
variable-length attention.
Model-specific adapters preserve HF projections, normalization and RoPE;
Qwen3.5 also preserves its output gate. Neither model has an attention-backend
selector or a separate eager-numerics export profile.

## Compilation and runtime contract

Adapters own cache updates. They append K/V in place, then call the read-only
operator with the populated prefix. The operator returns fresh contiguous
output and does not mutate or alias cache inputs. It calls PyTorch's ATen
FlashAttention with native grouped-query attention and lower-right causality,
covering initial prefill, appended chunks and decode. No `flash-attn` pip package
is required, and unused cache capacity is not read by attention.

Geometry comes from input tensors. Query heads must be a positive multiple of
KV heads; head dimensions must be positive, divisible by eight and at most 256.
Exports use BF16 and 1–512 padded tokens per call; one bundle supports dynamic
batches of 1 through 8. Model-specific restrictions remain in the [Qwen3](../compiler/qwen3/README.md#model-interface)
and [Qwen3.5](../compiler/qwen35/README.md#model-interface-and-state) guides.

The Python fake implementation supplies export shape checks;
[package metadata](package-metadata.md#dynamic-batch-interfaces) defines the
per-row calling convention. Both exports enable `backed_size_oblivious` to
retain singleton batches.

Adapters reject arbitrary masks, training, attention-weight outputs and unsupported cache/layer
types. Load HF weights before installing adapters: wrapping reuses parameter
objects but changes module paths. These adapters are not general HF training or
`generate()` replacements; validation retains an unmodified HF baseline.

Both adapters use `xinfer::append_kv_batched`, a `torch.library.triton_op`, for one
masked KV-write launch across all rows and heads. Only real tokens are written;
padded positions are not loaded. Its Triton code is packaged by AOTI without a
separate kernel library.

`xinfer::batched_flash_attention` packs real queries, calls ATen's variable-length
FlashAttention once for the batch and scatters results into zero-padded output.
CPU cumulative query/KV lengths and their CUDA copy are derived from CPU int32
length tensors `[B]` once per step and shared across layers. No CUDA length readback is needed.
Each request's contiguous cache is viewed as one block when capacity is divisible
by 256; other capacities pack valid K/V prefixes for the same batched call.
This fixed block view is internal to attention; runtime cache ownership and
[the model interfaces](package-metadata.md#dynamic-batch-interfaces) define the
eight-row allocation and active prefix views.

The wrapper fixes `num_splits=1`, normalizes the singleton block-table stride and
uses the general variable-length path for mixed inactive/decode rows. The pinned
ATen one-query GQA shortcut otherwise assumes a query for every row. Tests cover
these boundaries, query padding, unrelated rows and unused cache slots against
FP32 math. Query packing/scattering still has per-row overhead; performance has
not been measured for this implementation.

## Build and deployment

Build the compiler-owned `xinfer_attention_ops` CMake target before export or
standalone attention tests, using the selected model's LibTorch/CUDA stack and
[environment build commands](../environments/README.md#build-compiler-kernels-and-native-runtimes-for-each-model).
It writes `libxinfer_attention_ops.so` to `build/`.
`XINFER_ATTENTION_OPS_LIBRARY` selects another build-time library.
Runtime executable targets build independently.

Both exporters pass the shared C shim declaration/mapping from
`compiler/common/kernels/flash_attention/attention.py` to
`compiler/common/compile.py`. Torch's `aot_inductor.custom_ops_to_c_shims` emits
direct C calls; `aot_inductor.custom_op_libs` adds native linkage. The shim borrows
input tensor handles, calls the operator and transfers the output handle to
AOTI. Torch's error bridge propagates native exceptions.

On the Linux GCC toolchain, packaging temporarily sets `LIBRARY_PATH` for
link-time discovery and `LD_RUN_PATH='$ORIGIN'` for lookup beside the wrapper.
Both variables are restored afterward. Because they are process-wide, independent
concurrent exports must use separate processes; each export CLI compiles its
entries sequentially. `package_aoti` includes the native library in each `.pt2`
during normal packaging.

Standard Python and native AOTI loaders extract the library beside the wrapper;
the system linker loads it and registers the operator. Native executables
neither link nor install a separate attention library. Deploy the complete
[deployment bundle](package-metadata.md) with compatible LibTorch/CUDA. External-
library packages require re-export; editing JSON or copying a library beside
an old package does not meet the bundled contract.

Eager validation extracts the prefill package's implementation before installing
adapters, or reuses an already registered eager operator. Runtime-contract
validators use the standard package loaders without an eager preload. Whole-
package checksums cover native dependencies; loaders do not compare attention
library hashes or payload bytes between entries.

Independent prefill/decode implementations remain future work. The current
library shares a SONAME, C-shim symbol and process-wide Torch operator registration.
Selecting distinct implementations needs separate linkage identities or dispatch
within one library; removing a byte-equality guard alone does not provide it.

## Verification

With the native library built, run the shared operator tests and the affected
model's adapter/export checks. The [model guides](../README.md#getting-started)
list those commands. Operator checks use synthetic tensors; Qwen3 adapter tests
use random HF attention weights. Full-model validation separately checks
HF → adapted/optimized eager → Python AOTI → C++ AOTI, including state ownership,
write regions, masking and boundaries.

Follow the [shared numerical policy](development-tools.md#shared-numerical-policy).
Changing attention implementation or compiler tuning can change rounding; native
parity does not prove HF equivalence. [Validation status](validation-status.md)
records tested targets/capacities, known differences and remaining coverage.
