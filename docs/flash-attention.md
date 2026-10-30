# Qwen3 valid-prefix FlashAttention

The existing Flash module adapter is the sole supported Qwen3 export backend.
It targets batch-one BF16 Qwen3-0.6B, retains eight KV heads and attends only to
the populated cache prefix. There is no backend selector or alternate compiled
eager-numerics profile. Independent HF eager/FP32 references remain unmodified.

## Compilation and runtime contract

1. Load the original HF weights, then wrap each decoder layer's `self_attn`
   child with the existing `FlashAttention` adapter.
2. `FlashStep` calls the original embedding, rotary embedding, decoder layers,
   final norm and LM head with explicit caller-owned cache state. Each attention
   adapter reuses the original projections and Q/K normalization, updates K/V,
   then calls `xinfer::flash_attention` with native GQA.
3. Verify 28 custom attention calls and all 56 cache mutations in the exported
   graph, then compile the bounded dynamic prefill and single-token decode.
4. The read-only native operator calls PyTorch's ATen FlashAttention with
   lower-right causality. The same implementation handles initial prefill,
   appended chunks and decode. There is no `flash-attn` pip-package dependency.

The fixed cache layout is `[1,8,1024,128]`. The query has 16 heads, dimension 128,
with 1–512 tokens. A CPU uint8 extent tensor of shape `[valid_length + 1]` passes
populated prefix length through metadata; its data is ignored. The extra one
avoids input size-one specialization. Prefix slicing stays inside the opaque
operator and export requires `backed_size_oblivious=False`. The native flat
input list has 59 tensors: IDs, positions, 56 K/V tensors, then the extent.

The operator returns fresh contiguous output and never mutates/aliases its
cache inputs. Cache updates belong to the adapter. The adapter rejects arbitrary
masks, training, attention-weight requests and unsupported cache/layer types.
This is not a general replacement for HF training or `generate()`. Load weights
before installing adapters; wrapping reuses parameter objects but changes their
module paths by adding an `original` component.

## Build and deployment

Qwen3 always builds and links `libxinfer_attention_ops.so`. Build before export,
using matching LibTorch/CUDA and the toolchain commands in the
[Qwen3 guide](../compiler/qwen3/README.md#build-before-export). Python resolves
`build/libxinfer_attention_ops.so`, or `XINFER_ATTENTION_OPS_LIBRARY` for a separate
build directory. Native targets link it directly; `.pt2` does not bundle it.
Deploy the library alongside the binaries and their matching dependencies.

The runtime validates fixed `attention_backend: flash`, cache/extent settings,
operator ABI version 1, checkpoint/toolchain/architecture, tensor metadata and
actual numerical settings before loading packages. These metadata fields remain
important without a CLI selector. Old efficient packages are incompatible and
must be re-exported; legacy eager-numerics packages are also rejected. Existing
ordinary Flash packages retain the same native ABI and can be checked against
the current metadata contract.

```bash
# First build with the matching toolchain from the Qwen3 guide.
.venv/bin/python -m compiler.qwen3.tests.custom_attention \
  --output artifacts/attention-check-new
.venv/bin/python -m compiler.qwen3.tests.attention_adapter \
  --output artifacts/adapter-check-new
.venv/bin/python -m compiler.qwen3.export decode --output artifacts/qwen3-flash
.venv/bin/python -m compiler.qwen3.export prefill --output artifacts/qwen3-flash
AOTI_RUNTIME_CHECK_INPUTS=1 .venv/bin/python -m compiler.qwen3.tests.validate_cpp \
  --packages artifacts/qwen3-flash --native build/qwen_validate \
  --output artifacts/flash-validation-new
```

## Numerical policy and evidence

Finite results, C++/Python AOTI bitwise agreement, cache write regions, poisoning,
masking, reset/reuse, storage and boundary checks remain hard requirements. HF
logits retain the existing 5% relative-L2 and peak-normalized limits. HF cache
and top-token/greedy-sequence differences are accepted diagnostics and are
reported separately from runtime correctness and HF parity. HF logit comparisons
use identical fed tokens even when independent greedy trajectories differ.

The retained L20/Torch 2.7 evidence covers 26 cases/185 rows with exact native/
Python AOTI logits and caches. HF comparison reports 14/56 cache deviations and
183/185 top-token agreement; a separately tuned build had 20/56 cache deviations
and the same token mismatches. Focused adapter checks pass at maximum relative
L2 0.289% / peak-normalized error 0.747%, and nine direct/compiled attention cases
pass. These historical results motivate the explicit policy and do not replace
validation of a fresh artifact. Full details and artifact paths are in
[numerical history](experiments/qwen3-numerics.md).

Historical GB10 measurements (Torch 2.13/CUDA 13) reported approximately
1.27–1.32x decode and 1.38–1.49x prefill speedups against fused efficient
attention. They are not a speedup claim for L20 or a different toolchain.
Use the [paired benchmark](development-tools.md) for controlled measurements.
