# Package metadata

Both model families deploy `prefill.pt2`, `decode.pt2`, `weights.safetensors`,
`prefill.json` and `decode.json` together. Metadata and model interfaces follow
the current implementation without separate version fields. The shared archive
uses format 3.

`compiler/common/metadata.py` writes and validates the common format.
`tools/packages.py` owns model ABI and shared-weight bundle checks for Python
tooling. `compiler/common/weights.py` owns archive compilation, verified manifest
reading and shared tensor loading.
`runtime/common/spec.h` is the C++ reader; `runtime/<model>/runtime.h` validates
model-specific geometry and `runtime/common/state.h` implements state ownership.

## Metadata fields

| Fields | Meaning |
| --- | --- |
| `model_type`, `entry_point`, `token_range` | Model selection and per-call bounds |
| `torch_version`, `cxx11_abi`, `cuda_compute_capability` | Runtime compatibility |
| `checkpoint`, `vocab_size`, `eos_token_ids` | Checkpoint identity/manifest, vocabulary and explicit EOS list |
| `cache_length`, `state` | Allocated context capacity and ordered state descriptors |
| `export_options`, `tf32`, `triton_enable_fp_fusion`, `inductor_configs` | Export and numerical contract |
| `package_bytes`, `package_sha256`, `shared_weights` | Package and archive integrity |
| `model_specific` | DeltaNet numerical policy and minimal packing settings; empty for Qwen3 |

Both entries must agree on stable fields, including `model_specific`.
`entry_point`, `token_range`, package size and package hash may differ.
Missing/unknown fields, legacy aliases and mixed pairs are rejected.
`COMPARISON_EXCLUDED_FIELDS` in the Python metadata module defines benchmark
comparison exceptions; shared weights are compared by logical identity and
source mappings separately from their archive checksum.

`inductor_configs` retains the required flags `emulate_precision_casts=True`,
`aot_inductor.package_constants_in_so=False` and
`always_keep_tensor_constants=True`. Other resolved compiler options, dependency
versions, graph diagnostics and timings belong in JSON export-log events.
Detailed packing reports are saved as separate export diagnostics.
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
from these descriptors and the model's calling convention. They are not duplicated
under `model_specific`. Reset retains KV allocations and fills them with the
requested value; replacement state is initialized to zero.

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
  lengths. Qwen3 has 56 state tensors / 60 inputs; Qwen3.5-0.8B has 48 / 52;
  Qwen3.5-4B has 64 / 68.
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
current model interface directly; editing sidecars cannot change a package's
compiled signature.

## Shared-weight archive and ownership

`weights.safetensors` contains parameters and lifted tensor constants used by the
compiled graphs. Constants are captured during lowering, after graph passes such as
weight packing. Exact per-entry runtime names are recorded; names are not matched by
suffix. The manifest JSON is embedded under `__metadata__.xinfer_manifest` in the
safetensors header, not a separate file. It maps each entry's constants to tensor
views over deduplicated storage, preserving shape, stride, offset, dtype and device.
Tied embeddings and shared packed sources can therefore share storage without
changing their views.

The sidecar `shared_weights` descriptor records `format: 3`,
`file: weights.safetensors`, logical identity, byte size and SHA-256. The embedded
manifest also records the exact package hashes. Archive bytes cover both mappings and
tensor data; the archive binds package hashes, not sidecar hashes. Native loading
validates the safetensors header and copies deduplicated storages directly from an
`mmap`ped file without TorchScript (`torch.jit`) dependencies.

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
the old attention interface or an external attention dependency require re-export;
sidecar conversion cannot supply missing code or change the compiled ABI.

Keep legacy bundles with their matching runtimes when retaining them as evidence.
For retained bundles with the current dynamic-batch interface, remove the retired
`schema_version` and `runtime_abi_version` fields from both sidecars. This
metadata-only cleanup leaves the package and weight archive bytes unchanged.
Bundles with different inputs, state shapes or compiled kernels still need a
matching runtime or re-export.
