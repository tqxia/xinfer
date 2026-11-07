# Qwen3 valid-prefix FlashAttention

The existing Flash module adapter is the sole supported Qwen3 export backend.
It targets batch-one BF16 Qwen3-0.6B, retains eight KV heads and attends only to
the populated cache prefix. There is no backend selector or alternate compiled
eager-numerics profile. The independent BF16 HF eager reference remains unmodified.

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

Qwen3's cache layout is `[1,8,C,128]`, with `C` defaulting to the pinned model's
40,960-position ceiling; runtime capacity is derived from the
package state descriptors and the native operator's K/V tensors. The query has
16 heads, dimension 128, with 1–512 tokens per compiled call. The shared runner
splits complete prompts into calls of at most 512 tokens, subject to compiled
context capacity and the requested output budget; a one-token remainder
uses decode. A CPU uint8 extent tensor of
shape `[valid_length + 1]` passes populated prefix length through metadata; its data is ignored. The extra one
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
`build/qwen3/libxinfer_attention_ops.so` for Qwen3 and
`build/qwen35/libxinfer_attention_ops.so` for Qwen3.5 by default;
`XINFER_ATTENTION_OPS_LIBRARY` optionally selects another build or installed
library. Native targets link it directly; `.pt2` does not bundle it.
Deploy the library alongside the binaries and their matching dependencies.

`runtime_abi_version: 1` defines batch-one BF16 prefix Flash attention,
full-capacity caller-owned caches and the shape-only CPU extent input. The
former fixed backend/cache fields are replaced by this versioned contract.
Python and C++ query the loaded attention library's actual ABI 1 and retain
state, checkpoint/toolchain/architecture and numerical checks. See the
[ABI definition](package-metadata.md#runtime-abi-1).
Old efficient packages and legacy eager-numerics packages require re-export.
Compatible ordinary Flash packages retain the same compiled ABI, but their JSON
sidecars must use the [schema 5 metadata contract](package-metadata.md). Verified
schema 1 through schema 4 and unversioned sidecars can be converted in a
separate bundle directory without changing compatible compiled graphs or
weights. A conversion retains the compiled `cache_length`, sets schema 5 and
removes any former `max_prompt_length` field. Calling applications own any
additional input policy; conversion does not expand the graph.
Re-export for a chosen larger capacity and rebuild the attention library for
generalized capacity support. The previously recorded 16,384-slot Qwen3 profile
completed ordinary export and C++ execution on L20, with bitwise native/Python
AOTI agreement, passing existing HF numerical gates and a full-capacity
inference-memory probe. Those results do not validate the current 40,960-slot
default. The recorded 16K profile has not been validated on GB10.

```bash
# First select Q3_PY and build with the matching target toolchain.
BUILD="$PWD/build/qwen3"
"$Q3_PY" -m unittest \
  compiler.common.tests.test_attention compiler.qwen3.tests.test_attention
"$Q3_PY" -m compiler.qwen3.export --output artifacts/qwen3-flash
AOTI_RUNTIME_CHECK_INPUTS=1 "$Q3_PY" -m tools.validation.validate \
  --packages artifacts/qwen3-flash --native "$BUILD/xinfer_validate" \
  --output artifacts/flash-validation-new
```

The focused unit tests cover the shared operator and Qwen3 adapter, using an HF
attention module with random weights. They require the built attention library
but no checkpoint download or AOTI compilation. The separate attention validation
CLIs have been removed. `tools.validation.validate` runs the shared HF → eager
→ Python AOTI → C++ AOTI chain. Eager includes the Flash adapter; the chain covers
real-model cache and numerical checks. `validate_runtime` separately checks the
shipping CLI and deployment contract.

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
