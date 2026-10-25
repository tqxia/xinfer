# Read-only attention custom operator on GB10

The optional `--attention-backend flash` profile implements:

```text
QwenStep with ordinary full-cache SDPA
  -> torch.export
  -> replace the known attention subgraphs with xinfer::flash_attention
  -> re-export to rebuild signatures and tensor metadata
  -> AOTInductor package
  -> standalone C++ runtime
```

This is the only Flash export path. It uses the same PyTorch FlashAttention
kernel family as the removed direct-ATen exporter; the custom operator changes
the compiler boundary while preserving the attention algorithm. The operator is
read-only: cache updates remain in the exported graph. New custom-op exports use
ordinary contiguous caches with no padding and leave `backed_size_oblivious`
disabled. The 1,024-token capacity, fixed-size mutation inputs and CPU `[L+1]`
extent input remain unchanged.

The [initial padded custom-op report](../benchmarks/gb10-custom-attention-2026-10-25/REPORT.md)
records bitwise parity with direct Flash, effectively unchanged decode latency,
and 1.1–1.6% higher prefill medians. The existing numerical limitation remains.
The [unpadded comparison](../benchmarks/gb10-custom-attention-unpadded-2026-10-25/REPORT.md)
validates removal of the two export workarounds and compares with that padded
custom-op baseline. It retains bitwise parity; measured latency changes are
within 0.14% for decode and 0.36% for prefill.

## Code walkthrough

- [Export integration](../compiler/export.py): export the efficient-attention
  wrapper, run the replacement pass, re-export, validate the resulting
  `ExportedProgram`, then compile it. Save before/after graphs and rewrite counts.
- [Graph pass](../compiler/attention_rewrite.py): require the known causal-mask,
  GQA expansion and absolute-position cache-update patterns. Replace 28 SDPA
  calls, bypass the 8-to-16-head expansion, and preserve all 56 cache writes.
  Reject unknown masks and unexpected attention patterns. Dead-code elimination
  preserves mutation and other side-effect nodes.
- [Operator registration and fake kernel](../compiler/attention_ops.py): load
  the C++ library and describe a fresh, contiguous BF16 output with Q's shape.
  The fake implementation does not construct prefix views or call Flash.
- [Native implementation](../runtime/attention_ops.cpp): read `L` from
  `kv_extent.size(0) - 1`, create K/V prefix views, call native ATen Flash with
  lower-right causality, and return an output with normalized contiguous strides.
  The CPU extent's contents are never read. No input is mutated or returned as
  an alias.
- [Native contract](../runtime/runtime.h): check that the package pair agrees
  on the custom operator's name, ABI version and read-only contract. The runtime
  still passes the original fixed-size cache buffers and a view of its persistent
  1,025-byte CPU extent allocation. Cache allocation follows the package's
  declared padding and strides, including older padded packages.

The operator is intentionally restricted to this inference contract: batch one,
BF16, 16 query heads, 8 KV heads, head dimension 128, full cache shape
`[1,8,1024,128]`, and 1–512 new query tokens. The valid length must be at least
the query length and at most 1,024. Training is unsupported.

## Layout registration matters

The first compiled version inserted two full-cache repacking kernels per layer,
despite the C++ implementation itself using views. The final registration uses
`at::Tag::flexible_layout` so Inductor can pass cache buffers directly, including
the earlier padded layout.

The native implementation honors that tag: if a caller supplies a non-unit
feature stride, it makes the affected query or valid-prefix tensor contiguous.
The normal Qwen cache and query layouts have unit feature stride, so that
fallback is not needed. The operator tests include a non-unit-stride case.

Native Flash can also return different strides for singleton dimensions in
decode. The wrapper normalizes the output strides to its fake kernel's contract.
For the validated Qwen shapes, native Flash output is already logically
contiguous; this normalization does not require an output-data copy.

The custom operator is dispatched through AOTInductor's native proxy executor.
This creates a compiler boundary and adds dispatch work. The historical
comparisons above measured that overhead against the former direct-ATen path.

## Why the padding and size-oblivious setting are unnecessary here

Export and AOTInductor see fixed `[1,8,1024,128]` cache inputs and the opaque
custom op. The fake implementation's output shape depends on Q, not the valid
cache length. Prefix slicing and native FlashAttention execute inside the C++
implementation using concrete runtime lengths.

Consequently, the compiler never specializes on whether the internal prefix view
becomes contiguous at `L=1024`, or on its size-one boundary at `L=1`. The native
operator can handle both layouts and lengths. This removes the need for padding
and `backed_size_oblivious` in the Flash export path. The removed direct-ATen
exporter exposed those intermediate views to the compiler and needed both
workarounds.

New custom packages record `cache_padding: 0`, cache strides
`[1048576,131072,128,1]` and `export_options.backed_size_oblivious: false`.
The earlier padded strides were `[1049600,131200,128,1]`. Native requested tensor
storage decreases by 98 KiB across 56 caches; allocator reservations need not
decrease by that exact amount. Logical cache capacity remains 112 MiB in BF16.

For older packages without `cache_padding`, the runtime uses their original
contract: one spare position for legacy custom-op packages, zero for efficient
SDPA. Custom packages may use either supported layout; metadata must agree
across the prefill/decode pair. The operator ABI and implementation are unchanged.

The CPU extent is still `[L+1]`, with declared dynamic size 2–1025. Removing that
encoding is outside this change; fixed cache inputs and extent metadata continue
to avoid a dynamically sized mutation input.

## Build, export and validate

Run from the repository root with the existing GB10 venv and checkpoint:

```bash
./scripts/build-gb10.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/work/inductor-custom"
export TORCHINDUCTOR_COMPILE_THREADS=2

# Use absent output paths for tests and package generation.
.venv/bin/python compiler/tests/custom_attention.py \
  --output artifacts/custom-attention-unit-new
.venv/bin/python compiler/export.py decode --attention-backend flash \
  --output artifacts/aoti-flash-new
.venv/bin/python compiler/export.py prefill --attention-backend flash \
  --output artifacts/aoti-flash-new
AOTI_RUNTIME_CHECK_INPUTS=1 .venv/bin/python compiler/validate_cpp.py \
  --packages artifacts/aoti-flash-new \
  --output artifacts/three-path-flash-new
```

The isolated test checks exact agreement with native Flash, an independent FP32
masked-attention reference, input immutability, future-slot poisoning, extent-data
independence, `torch.library.opcheck`, invalid lengths, rejection of unknown masks,
and strict compiled boundary cases including cache length one and full capacity.

Full-model validation also checks native/Python AOTI parity, greedy fixtures,
cache update regions, stable storage, reset/reuse, overflow and chunked prefill.
The existing Flash profile's BF16 numerical limitation is separate: its unchanged
5% whole-cache acceptance criterion can cause the full validator to exit nonzero
after saving its complete report. Do not interpret structural or bitwise parity
with Flash as improved numerical acceptance against HF BF16.

For native CLI smoke checks, populate the package's `reference/` directory with
its validated Python AOTI fixture logits, as expected by `compiler/tests/smoke_cpp.py`.

## Loading and deployment

The build produces `build/libxinfer_attention_ops.so`. It contains native operator
registration and links LibTorch, without Python or `libtorch_python`.
The native executables link this library and reference its ABI-version function,
ensuring the linker retains registration before AOTI package loading.

Python consumers must register the operator before loading a custom package:

```python
from attention_ops import load_package_attention_ops

load_package_attention_ops(metadata)  # decode.json, using compiler/ on sys.path
model = torch._inductor.aoti_load_package("artifacts/aoti-flash/decode.pt2")
```

The native operator library is a deployment dependency; it is not embedded in
the `.pt2` archive. Keep the executables, operator library and matching LibTorch
build together. The standard build configures paths for running on this GB10 host.

## Migration from the two Flash exporters

`--attention-backend flash` now selects the custom operator and graph pass.
The `flash-custom` CLI choice, direct HF attention adapter, mask monkeypatch,
model-side prefix slicing, and padded export allocator have been removed.
There is no second Flash exporter. The general no-flag export default remains
`efficient` because Flash's full-model numerical acceptance is still unresolved.

Older compiled custom packages retain the `flash-custom` metadata label; the
Python and C++ loaders accept that label with the same custom-op ABI contract.
This compatibility does not restore the removed exporter. Direct-ATen packages
have no `custom_attention` metadata and fail with a re-export instruction before
loading their weights. Rebuild them with the commands above.

On this checkout, `artifacts/aoti-flash` contains freshly compiled canonical
custom-op packages. The former direct packages were moved to
`artifacts/aoti-flash-direct-legacy`. The [consolidation validation
report](../benchmarks/gb10-flash-canonical-2026-10-25/REPORT.md) records package,
operator, graph, loader and baseline regression checks.

## Compare with the previous custom packages

```bash
.venv/bin/python compiler/benchmark.py compare \
  --baseline-packages artifacts/aoti-flash-custom-unpadded \
  --candidate-packages artifacts/aoti-flash \
  --baseline-label custom_previous --candidate-label flash \
  --output artifacts/benchmark-flash-canonical-new
```

Omitting package flags instead compares fused efficient AOTI at `artifacts/aoti`
against canonical Flash at `artifacts/aoti-flash`. The harness preserves
alternating process order, idle checks, telemetry, fixed fed tokens and package
hashes. It also records the operator library hash and custom-op/rewrite sources.
Its default protocol uses four rounds, two warmups, five measured requests per
length and 64 decode steps.
