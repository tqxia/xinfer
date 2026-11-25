# Package metadata

Both model families deploy **four files** together: `prefill.pt2`, `decode.pt2`,
`weights.safetensors` and `metadata.json`. Metadata and model interfaces follow
the current implementation without separate version fields.

`compiler/common/metadata.py` writes and validates the common format.
`tools/packages.py` owns model ABI and shared-weight bundle checks for Python
tooling. `compiler/common/compile.py` owns AOTI compilation and paired bundle
publication; `compiler/common/weights.py` owns the shared-weight archive pool,
verified manifest reading and shared tensor loading.
`runtime/common/spec.h` is the C++ reader; `runtime/<model>/runtime.h` validates
model-specific geometry and `runtime/common/state.h` implements state ownership.

## Metadata fields

| Fields | Meaning |
| --- | --- |
| `model_type` | Model family (`qwen3` or `qwen35`) used to select runtime behavior |
| `torch_version`, `cxx11_abi`, `cuda_compute_capability` | Runtime compatibility |
| `cache_length`, `max_prefill_tokens` | Allocated context capacity and compiled per-call prefill ceiling (512) |
| `vocab_size`, `eos_token_ids` | Vocabulary upper bound and explicit EOS stop list |
| `checkpoint` | Prepared `model_manifest.json` object (`model`, `revision` and checkpoint files) |
| `state` | Ordered contiguous state descriptors (`layer`, `kind`, `shape`, `dtype`, `update`) |
| `entries` | Per-entry (`prefill`, `decode`) `.pt2` byte sizes and SHA-256 checksums |
| `shared_weights` | Shared `weights.safetensors` logical identity, byte size and SHA-256 checksum |

Missing/unknown fields and retired aliases are rejected.
`COMPARISON_EXCLUDED_FIELDS` in `compiler/common/metadata.py` excludes `entries`
and `shared_weights` from benchmark comparison; shared weights are compared by
logical identity and source mappings separately from their archive checksum.

Fixed export/compiler invariants (`backed_size_oblivious=True`, TF32 disabled,
`emulate_precision_casts=True`, weightless AOTI compilation) live in code rather
than `metadata.json`. Resolved compiler options, pass counts and compile timings
are printed as export-log events, and graph dumps / packing reports are saved as
separate export diagnostics. There are no package `provenance`, `diagnostics` or
source-code-hash fields. Retain the export log when investigating a build.

## Capacity and state

`--max-context-length` chooses `cache_length` before GPU allocation. The default
is the selected text config's `max_position_embeddings`; valid values are 512
through that ceiling (capped by `INT32_MAX // 8` across the 8 compiled batch
slots). Supported models use default RoPE; no scaling factor is inferred. KV
descriptors contain exactly `cache_length` slots.

Decode accepts 1 token per call; prefill accepts 2–`max_prefill_tokens` (512)
tokens per call. The runner chunks a complete prompt and uses decode for a
singleton remainder. The prompt plus requested `--max-output-tokens` budget must
fit `cache_length` before execution begins; the budget is rejected rather than
clamped. There is no separate prompt cap. Changing capacity requires re-export
of both entries; editing `metadata.json` cannot enlarge compiled buffers or
symbolic bounds.

Each ordered state descriptor contains `layer`, `kind`, `shape`, `dtype` and
`update`. KV uses `inplace`; convolution/recurrent state uses `replace`.
Readers derive names, contiguous strides, model geometry and output positions
from these descriptors and the model's calling convention. Reset retains KV
allocations and fills them with the requested value; replacement state is
initialized to zero.

## Dynamic batch interfaces

One prefill/decode bundle accepts every batch size B=1..8, including scalar
generation. The package ceiling is hardcoded to eight in state descriptors; at
runtime, `Runner` and `StatePolicy` accept an optional `allocated_batch_size`
in `1..8` (defaulting to the package ceiling 8; scalar CLI and benchmark tools
allocate 1 row). Export and compilation enable `backed_size_oblivious` to
retain B=1 in the symbolic range.

- Python inputs: `(input_ids, cache_position, state, kv_lengths, query_lengths)`.
  IDs and positions are CUDA int64 `[B,T]`. State inputs are contiguous views
  into the first B rows of the allocated state buffers.
- Flat AOTI inputs: IDs, positions, ordered state tensors, KV lengths, query
  lengths. Qwen3-0.6B/1.7B have 56 state tensors / 60 inputs; Qwen3.5-0.8B has
  48 / 52; Qwen3.5-4B has 64 / 68.
- The two length inputs are CPU int32 `[B]`: populated length after this call
  and real input count for each row. The graph builds cumulative lengths and
  their CUDA copy once, shared across layers.
- IDs are right-padded. A row starts with tokens; later it may consume 0–512
  tokens per call. At least one row must be active. Width one selects decode;
  widths 2–512 select prefill. Per-row positions identify contiguous appends.
- Logits are BF16 `[B,vocab_size]` for each row's last real token. Empty rows
  return zero logits without advancing any state or position.
- Qwen3 returns only logits and updates KV in place. Qwen3.5 returns logits
  followed by all state tensors: KV outputs alias input buffers, while fresh
  convolution/recurrent outputs replace the active rows. Recurrent state is FP32.
- State allocation and row identity persist across calls up to
  `allocated_batch_size`. Smaller B omits tail rows without modifying them.
  Larger B resumes those rows or starts unused rows with their first tokens.
  The runtime retains all allocated rows, combining functional outputs with the
  unchanged tail without modifying the previous functional input. Full-batch
  functional outputs (`B == allocated_batch_size`) are adopted directly.
- Reset clears all allocated rows; KV storage is retained. The state policy
  checks returned shape, dtype, device, contiguity and required aliases before
  adopting outputs.
- The runner validates every supplied row before execution. `positions()` exposes
  all allocated row counts; `position()` is their maximum for trace handling.

Qwen3 flattens independent positions around HF's RoPE call to avoid singleton
matmul specialization. Qwen3.5 computes the equivalent text-only frequencies
without HF's redundant cross-grid slice writes, which otherwise introduce a
`B >= 2` guard during export. Component tests require bitwise HF agreement.

Scalar generation and benchmark commands execute B=1 from the same bundle.
`xinfer_validate` accepts explicit `batch_steps`. Per-row request replacement and
continuous-batching scheduling are not implemented. The runtime calls the
current model interface directly; editing `metadata.json` cannot change a
package's compiled signature.

## Shared-weight archive and ownership

`weights.safetensors` contains parameters and lifted tensor constants used by the
compiled graphs. Constants are captured during lowering, after graph passes such as
weight packing. Exact per-entry runtime names are recorded; names are not matched by
suffix. The manifest JSON is embedded under `__metadata__.xinfer_manifest` in the
safetensors header, not a separate file. It maps each entry's constants to tensor
views over deduplicated storage, preserving shape, stride, offset, dtype and device.
Tied embeddings and shared packed sources can therefore share storage without
changing their views. Both model families record `identity` as
`{"model": "<family>", "checkpoint": <manifest>}` (`qwen3` or `qwen3.5`).

`metadata.json`'s `shared_weights` descriptor records the archive's logical
identity, byte size and SHA-256 (always targeting `weights.safetensors`). The
embedded manifest also records the exact package hashes. Archive bytes cover both
mappings and tensor data; the archive binds package hashes, not `metadata.json`
hashes. Native loading validates the safetensors header and copies deduplicated
storages directly from an `mmap`ped file without TorchScript (`torch.jit`)
dependencies.

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

Re-export with the current exporter by default. Packages with embedded weights,
the old attention interface, an external attention dependency or legacy
`prefill.json`/`decode.json` sidecars require re-export; metadata conversion
cannot supply missing code or change the compiled ABI. Keep historical bundles
with their matching runtimes when retaining them as evidence.
