# Shared valid-prefix FlashAttention

Qwen3 and Qwen3.5 full-attention layers use the same native operator,
`xinfer::flash_attention`, implemented under `compiler/common/kernels/`.
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
The current exports use batch-one BF16 and 1–512 tokens per call. Model-specific
restrictions remain in the [Qwen3](../compiler/qwen3/README.md#model-interface)
and [Qwen3.5](../compiler/qwen35/README.md#model-interface-and-state) guides.

A CPU uint8 extent tensor has shape `[valid_length + 1]`; its contents are ignored.
This conveys prefix length without reading a CUDA scalar. The extra element
avoids size-one specialization. Prefix slicing stays inside the opaque operator,
and export requires `backed_size_oblivious=False`. The Python fake implementation
supplies export shape checks; [runtime ABI 1](package-metadata.md#runtime-abi-1)
defines the full model calling convention.

Inputs must be unpadded contiguous appends to an existing prefix. Adapters reject
arbitrary masks, training, attention-weight outputs and unsupported cache/layer
types. Load HF weights before installing adapters: wrapping reuses parameter
objects but changes module paths. These adapters are not general HF training or
`generate()` replacements; validation retains an unmodified HF baseline.

## Build and deployment

Build the compiler-owned `xinfer_attention_ops` CMake target before export or
standalone attention tests, using the selected model's LibTorch/CUDA stack and
[environment build commands](../environments/README.md#build-compiler-kernels-and-native-runtimes-for-each-model).
It writes `libxinfer_attention_ops.so` to `build/qwen3/` or `build/qwen35/`.
`XINFER_ATTENTION_OPS_LIBRARY` selects another build-time library.
Runtime executable targets build independently.

Both exporters pass the shared C shim declaration/mapping from
`compiler/common/attention.py` through the shared-weight compiler to
`compiler/common/package.py`. Torch's `aot_inductor.custom_ops_to_c_shims` emits
direct C calls; `aot_inductor.custom_op_libs` adds native linkage. The shim borrows
input tensor handles, calls the operator and transfers the output handle to
AOTI. Torch's error bridge propagates native exceptions.

On the Linux GCC toolchain, packaging temporarily sets `LIBRARY_PATH` for
link-time discovery and `LD_RUN_PATH='$ORIGIN'` for lookup beside the wrapper.
Both variables are restored afterward. Because they are process-wide, independent
concurrent exports must use separate processes; each export CLI compiles its
entries sequentially. `package_aoti` includes the native library in each `.pt2`
during normal packaging.

Standard Python/C++ AOTI loaders extract the library beside the wrapper; the
system linker loads it and registers the operator. Native executables neither
link nor install a separate attention library. Deploy the complete
[five-file bundle](package-metadata.md) with compatible LibTorch/CUDA. External-
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
