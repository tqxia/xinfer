# Package metadata

All supported models deploy one self-contained bundle directory containing
`prefill.pt2`, `decode.pt2`, `weights.safetensors`, `metadata.json`, and bundled
tokenizer files (`tokenizer.json`, `tokenizer_config.json`, optional
`chat_template.jinja`). Metadata and model interfaces follow the current
implementation without separate version fields.

- `compiler/common/metadata.py` writes and validates `metadata.json`.
- `compiler/common/compile.py` compiles and publishes paired bundles;
  `compiler/common/weights.py` owns the shared-weight archive pool and manifest.
- `tools/packages.py` owns Python-side ABI and bundle checks.
- `runtime/core/src/spec.rs` is the native Rust reader and geometry validator;
  `runtime/core/src/state.rs` owns state allocation and views.

## Metadata fields

| Fields | Meaning |
| --- | --- |
| `model_type` | Model family (`qwen3`, `qwen35` or `qwen38`) selecting runtime behavior |
| `torch_version`, `cxx11_abi`, `cuda_compute_capability` | Runtime compatibility |
| `cache_length`, `max_prefill_tokens` | Per-row context capacity and per-call prefill ceiling (`512`) |
| `vocab_size`, `eos_token_ids` | Vocabulary upper bound and explicit EOS stop list |
| `checkpoint` | Prepared `model_manifest.json` object (`model`, `revision`, checkpoint files) |
| `state` | Ordered contiguous in-place state descriptors (`layer`, `kind`, `shape`, `dtype`) |
| `entries` | Per-entry (`prefill`, `decode`) `.pt2` byte sizes and SHA-256 digests |
| `shared_weights` | `weights.safetensors` logical identity, byte size and SHA-256 digest |
| `tokenizer` | Bundled tokenizer byte sizes and SHA-256 digests (`tokenizer.json`, `tokenizer_config.json`, optional `chat_template.jinja`) |

Missing/unknown fields and retired aliases are rejected.
`COMPARISON_EXCLUDED_FIELDS` in `compiler/common/metadata.py` excludes `entries`
and `shared_weights` from benchmark comparison (weights are compared by logical
identity and source mappings separately).

`model_type` dispatches package tools and runtime checks: `qwen35` and `qwen38`
share hybrid state validation and calling convention in `runtime/core/src/spec.rs`
while keeping distinct `model_type` and shared-archive `identity.model` values
(`qwen3`, `qwen3.5` or `qwen3.8`). Fixed compiler invariants
(`backed_size_oblivious=True`, TF32 disabled, `emulate_precision_casts=True`,
weightless AOTI compilation) live in code rather than `metadata.json`; compiler
options, pass counts and timings are emitted as export logs, while graph dumps
and packing reports are separate export diagnostics.

## Capacity and state

`--max-context-length` chooses per-row `cache_length` before GPU allocation
(defaulting to the model config's `max_position_embeddings`; valid range `512`
through that ceiling, capped by `INT32_MAX // 8`). Supported models use default
RoPE. Changing capacity requires re-exporting both entries.

Decode accepts 1 token per call; prefill accepts 2–512 tokens per call. Longer
prompts are chunked automatically with decode handling a 1-token remainder. The
prompt plus output budget must fit `cache_length` before execution begins
(requests exceeding capacity are rejected, never clamped).

Every state tensor across all model families (KV cache, DeltaNet convolution and
FP32 recurrent state) updates in place, so descriptors carry only `layer`,
`kind`, `shape` and `dtype`. Reset retains all allocations (enforcing 16-byte
alignment so AOTI alignment guards never clone caller buffers), fills KV buffers
(`key`/`value`) with the requested fill value, and zeroes `conv`/`recurrent`
buffers.

## Dynamic batch interfaces

One bundle serves every batch size `B=1..8`. State descriptors declare the
package ceiling (`8`), while `Session` / `Runner` / `StatePolicy` accept an
optional `allocated_batch_size` in `1..8` (default `8`; scalar `xinfer_aoti` and
`xinfer_benchmark` allocate `1` row). Export enables `backed_size_oblivious` so
`B=1` remains in the symbolic range.

- **Python signature:** `(input_ids, cache_position, state, kv_lengths, query_lengths)`.
  IDs and positions are CUDA int64 `[B,T]`; length tensors are CPU int32 `[B]`
  (populated length after this call and real token count per row); `state` is a
  tuple of contiguous views into the active rows of allocated buffers.
- **Flat AOTI inputs:** IDs, positions, ordered state tensors, `kv_lengths`,
  `query_lengths` (Qwen3-0.6B/1.7B: 56 state tensors / 60 inputs; Qwen3.5-0.8B:
  48 / 52; Qwen3.5-4B: 64 / 68; Qwen3.8-27B-FP8: 128 / 132).
- **Padding and dispatch:** inputs are right-padded. A row starts with real
  tokens and may later consume 0–512 tokens per call (at least one row must be
  active). Width 1 invokes decode; widths 2–512 invoke prefill. Logits are BF16
  `[B,vocab_size]` for each row's last real token; empty rows return zero logits
  without advancing state or position.
- **Unified in-place ABI:** all model families return only `[B,vocab_size]` BF16
  logits and mutate all state tensors in place.
- **State ownership & continuous batching:** row identity persists across calls
  up to `allocated_batch_size`; narrower prefix views (`narrow(0, 0, B)`) leave
  tail rows untouched. `runtime/core/` also exposes per-slot execution and reset
  (`Session::step_slot`, `step_slots_range`, `reset_slot`), which
  `runtime/engine/` and `xinfer_serve` use for `B=1..8` continuous batching with
  interleaved chunked prefill.

## Shared-weight archive and ownership

`weights.safetensors` stores deduplicated parameters and lifted constants after
graph passes (such as linear packing), with an embedded JSON manifest under
`__metadata__.xinfer_manifest` mapping exact per-entry constant names to views
(shape, stride, offset, dtype, device) and recording both `.pt2` hashes and
`identity = {"model": "<family>", "checkpoint": <manifest>}`. E4M3 FP8 storage
(`float8_e4m3fn`) is supported with separate scale tensors.

One export invocation compiles both entries sequentially, verifies shared tensor
content, stages all bundle files in a sibling temporary directory, and publishes
atomically only on success (never overwriting a populated output path).

Python and native Rust retain one tensor pool and inject constants into both
loaders with `load_constants(..., user_managed=True)` (verified pointer-for-pointer
against the pool). Rust struct field drop order in `Model` drops both loaders
before `_weights`. Compilation applies the guarded
[Torch 2.13 header overlay](../compiler/patches/torch_2_13/README.md) so cloned
user-managed constant handles release cleanly with their loaders.

## Loading, native dependencies and compatibility

Every native `Model::open` validates metadata/ABI, verifies `.pt2`,
`weights.safetensors` and tokenizer file sizes and SHA-256 digests once per
load, and checks live GPU compute capability before loading weights.

All bundles embed `libxinfer_attention_ops.so` inside each `.pt2` (Qwen3.5 and
Qwen3.8 also embed `libxinfer_deltanet_ops.so`); AOTI extracts them beside the
wrapper via `$ORIGIN`. LibTorch and CUDA remain external dependencies; CPU ISA
compatibility is not checked at load time (see
[runtime installation](../environments/README.md#install-and-relocate-the-runtime)).

Re-export bundles whenever the package schema or compiled ABI changes (no legacy
format migration or `model_type` compatibility aliasing is provided). Keep
historical bundles paired with their matching runtimes when retained as evidence.
