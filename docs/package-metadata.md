# Package metadata

Both model families deploy `prefill.pt2`, `decode.pt2`, `weights.pt`,
`prefill.json` and `decode.json` together. The JSON sidecars use
`schema_version: 5`; Qwen3 uses `runtime_abi_version: 4`, and Qwen3.5 uses
runtime ABI 1. The shared archive uses format 2.
These versions describe different contracts and evolve independently.

`compiler/common/metadata.py` writes and validates the common format.
`tools/packages.py` adds model ABI checks for Python tooling.
`runtime/common/spec.h` is the C++ reader; `runtime/<model>/runtime.h` validates
model-specific geometry and `runtime/common/state.h` implements state ownership.

## Metadata fields

| Fields | Meaning |
| --- | --- |
| `schema_version`, `runtime_abi_version`, `model_type`, `entry_point`, `token_range` | Metadata format, calling convention, model selection and per-call bounds |
| `torch_version`, `cxx11_abi`, `cuda_compute_capability` | Runtime compatibility |
| `checkpoint`, `vocab_size`, `eos_token_ids` | Checkpoint identity/manifest, vocabulary and explicit EOS list |
| `cache_length`, `state` | Allocated context capacity and ordered state descriptors |
| `export_options`, `tf32`, `triton_enable_fp_fusion`, `inductor_configs` | Export and numerical contract |
| `package_bytes`, `package_sha256`, `shared_weights` | Package and archive integrity |
| `model_specific` | DeltaNet numerical policy and minimal packing settings; empty for Qwen3 |

Both entries must agree on stable fields, including `model_specific`.
`entry_point`, `token_range`, package size and package hash may differ.
Missing/unknown fields or versions, legacy aliases and mixed pairs are rejected.
`COMPARISON_EXCLUDED_FIELDS` in the Python metadata module defines benchmark
comparison exceptions; shared weights are compared by logical identity and
source mappings separately from their archive checksum.

`inductor_configs` retains the required flags `emulate_precision_casts=True`,
`aot_inductor.package_constants_in_so=False` and
`always_keep_tensor_constants=True`. Other resolved compiler options, dependency
versions, graph/packing diagnostics and timings belong in JSON export-log events.
There are no package `provenance`, `diagnostics` or source-code-hash fields.
Retain the export log when investigating a build.

## Capacity and state

`--max-context-length` chooses `cache_length` before GPU allocation. The default
is the selected text config's `max_position_embeddings`; valid values are 512
through that ceiling. Supported models use default RoPE; no scaling factor is
inferred. KV descriptors contain exactly that many slots.

`token_range` describes one compiled invocation: decode `[1,1]`, prefill
`[2,512]`. The runner chunks a complete prompt and uses decode for a singleton
remainder. The prompt plus requested `--max-output-tokens` budget must fit the
capacity before execution begins; the budget is rejected rather than clamped.
There is no separate prompt cap. Changing capacity requires re-export of both
entries; editing metadata cannot enlarge compiled buffers or symbolic bounds.

Each ordered state descriptor contains `layer`, `kind`, `shape`, `dtype` and
`update`. KV uses `inplace`; convolution/recurrent state uses `replace`.
Readers derive names, contiguous strides, model geometry and output positions
from these descriptors and the runtime ABI. They are not duplicated under
`model_specific`. Reset retains KV allocations and fills them with the requested
value; replacement state is initialized to zero.

## Runtime ABI 1

This is the Qwen3.5 text interface.

- Batch size is one; model activations and logits are BF16. State descriptors
  retain individual dtypes, including FP32 recurrent state.
- Flat inputs are CUDA int64 IDs, CUDA int64 cache positions, ordered contiguous
  state tensors, then a CPU uint8 extent vector of length `valid_length + 1`.
  Only the extent's shape is read. The extra element avoids size-one specialization.
- KV buffers have full compiled capacity. Adapters update only appended K/V
  positions in place; the read-only Flash operator attends to the valid prefix.
- Convolution/recurrent inputs remain unchanged. Returned tensors replace them.
- Qwen3.5 returns last-token logits `[1, vocab_size]` followed by every state
  tensor in descriptor order.
  Returned K/V must alias the corresponding caller-owned buffers.

The common state policy checks returned shape, dtype, device, contiguity and
required aliases before adopting replacement state. Changing extent semantics,
input/output order or mutation behavior requires a new runtime ABI. Changing
weights or compiler tuning does not by itself change the calling convention.

## Runtime ABI 4: dynamic Qwen3 batches

One Qwen3 bundle accepts every batch size `B=1..8`. The range is hardcoded;
state descriptors specify the full eight-row allocation `[8,8,C,128]`.
Export and compilation enable `backed_size_oblivious` so AOTI retains B=1
in the symbolic range. HF rotary positions are flattened for the RoPE call
and restored afterward, avoiding a singleton-batch matmul specialization.

- Python inputs: `(input_ids, cache_position, kv, kv_lengths, query_lengths)`.
  IDs and positions are CUDA int64 `[B,T]`. `kv` contains 56 contiguous BF16
  views `[B,8,C,128]` into the first B rows of the caller's allocation.
- Flat AOTI inputs: IDs, positions, 56 KV tensors, KV lengths, query lengths.
  There are always 60 inputs, independent of B.
- The two length inputs are CPU int32 tensors `[B]`. Their values give each
  row's populated length after this call and real input count. The graph builds
  cumulative lengths and a CUDA copy once per call, shared across all layers.
- IDs are right-padded. Only real inputs update KV at that row's positions.
  A row must start with tokens; later it can consume 0–512 tokens per call.
  At least one row must be active. Width one selects decode; 2–512 selects prefill.
- Output is BF16 `[B,vocab_size]`, containing each row's last-real-token logits.
  Empty rows return zero logits without advancing state or position.
- Cache allocation and row identity persist across calls. Smaller B omits tail
  rows without modifying them. Larger B resumes those rows or initializes unused
  rows with their first tokens. Reset clears all eight rows without reallocating.
- The runner validates every supplied row before execution. `positions()` exposes
  all eight row counts; `position()` is their maximum for trace handling.

Scalar generation and benchmark commands execute B=1 from the same bundle.
`xinfer_validate` accepts explicit `batch_steps`. Per-row request replacement and
continuous-batching scheduling are not implemented. Qwen3 ABI 1/2/3 packages
must be re-exported; editing sidecars cannot change their compiled signature.

## Shared-weight archive and ownership

`weights.pt` contains parameters and lifted tensor constants used by the compiled
graphs. Constants are captured during lowering, after graph passes such as weight
packing. Exact per-entry runtime names are recorded; names are not matched by
suffix. `extra/weights.json` is embedded inside the archive, not a separate file.
It maps each entry's constants to tensor views over deduplicated storage,
preserving shape, stride, offset, dtype and device. Tied embeddings and shared
packed sources can therefore share storage without changing their views.

The sidecar `shared_weights` descriptor records `format: 2`, `file: weights.pt`,
logical identity, byte size and SHA-256. The embedded manifest also records the
exact package hashes. Archive bytes cover both mappings and tensor data; the
archive binds package hashes, not sidecar hashes.

One export invocation loads the model once, compiles both entries sequentially,
requires consistent shared tensor content, and serializes the archive once.
It stages the complete result in a sibling temporary directory and publishes
only after both entries succeed. Populated output directories are preserved.
Graph dumps and packing reports are diagnostics, not extra deployment files.

Python and C++ retain one tensor pool and inject constants into both loaders with
`load_constants(..., user_managed=True)` (the fourth C++ argument).
The native loader verifies its constant pointers against the supplied storage.
Weights remain immutable and their owner outlives both loaders; C++ member order
ensures loaders are destroyed first. Current packages require the supported
Torch 2.13 stack and weightless compilation, not embedded per-entry weights.

## Loading and native dependencies

Every native runtime construction validates metadata/model ABI, verifies each
package's size and hash, verifies the shared archive, and checks actual GPU
compatibility before loading GPU weights. Each package and archive is hashed
once per construction, and the archive manifest is parsed once. This is a
load-scoped result, not a persistent trust cache across runtime instances.

Both packages embed `libxinfer_attention_ops.so`; package checksums cover it.
Generic weight readers do not inspect or compare the embedded attention payloads.
Current exporters select one implementation for both entries, but readers do not
require byte equality between them. Standard AOTI loaders extract the native
library with the wrapper. LibTorch/CUDA remain external dependencies.
See [attention packaging](flash-attention.md#build-and-deployment) for the C shim,
linker search path and limits on selecting independent implementations.

Native `load_ms` includes integrity preflight, weight loading and loader setup.
Older measurements excluded checks performed during spec construction; compare
whole-process elapsed time when crossing that reporting change.
CPU instruction-set compatibility is not enforced; see
[deployment constraints](../environments/README.md#install-and-relocate-the-runtime).

## Compatibility and retained bundles

Current readers do not translate schema 1–4 or unversioned sidecars. Re-export
with the current exporter by default. Packages with embedded weights, the old
attention interface or an external attention dependency require re-export;
sidecar conversion cannot supply missing code or change the compiled ABI.

For a verified Qwen3.5 schema 3/4 bundle that already embeds attention and satisfies
runtime ABI 1, metadata-only conversion may reuse the immutable packages and
archive. Preserve the original directory, set schema 5 in both copied sidecars,
remove `max_prompt_length` if present, and validate the new pair with current
readers. Preserve capacity, token ranges, state and numerical/integrity fields.
Conversion does not increase capacity. Older schemas require validating their
fixed ABI descriptions first; do not assign an ABI version blindly. Keep legacy
bundles with their matching runtimes when retaining them as evidence.
