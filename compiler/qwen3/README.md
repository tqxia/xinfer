# Qwen3 AOTI / Native Runtime

Qwen3 compiles BF16 AOTInductor bundles with valid-prefix FlashAttention and
native grouped-query attention. See
[environments](../../environments/README.md) for setup/builds,
[development tools](../../docs/development-tools.md) for shared CLIs and policy,
and [validation status](../../docs/validation-status.md) for current results.

## Dynamic-batch inference

Every bundle supports dynamic `B=1..8` with independent prompt lengths and cache
positions. `Session::step_batch` accepts `B` right-padded token vectors (up to
512 real tokens per row) and returns `[B,vocab_size]` BF16 logits for each
row's last real token; width 1 uses decode and widths 2–512 use prefill.

The package ceiling is 8 rows (`Session` allocates `1..8` rows — `1` for scalar
`xinfer_aoti`/`xinfer_benchmark`, up to `8` for `xinfer_serve` and
`xinfer_validate`). Row indices retain state across calls; `reset()` clears all
allocated rows, while `reset_slot()`, `step_slot()` and `step_slots_range()`
power per-slot recycling and continuous batching in `xinfer_serve`. Context
capacity is per row (at `C=1024`, an 8-row KV allocation is ~896 MiB plus
weights and workspaces).

```bash
"$PY" -m compiler.qwen3.export --model-dir work/models/qwen3-0.6b   --max-context-length 1024 --output artifacts/qwen3-batched
"$PY" -m tools.validation.qwen3.validate_batch   --model-dir work/models/qwen3-0.6b --packages artifacts/qwen3-batched   --output artifacts/validation-batched
```

`validate_batch` runs `B=1..8` and batch-transition fixtures against independent
BF16 HF references, adapted eager, Python AOTI and native AOTI (`xinfer_validate`),
checking 5% logit bounds, bitwise native parity, padding, row isolation, reset,
poisoned future KV, `511/512/513` boundaries, capacity and overflow. Per-group
reports live under `b1/`–`b8/`, `transitions/` and `summary.json`.

## Prepare and export

Build `xinfer_attention_ops` and native binaries in `build/` first (see
[requirements and targets](../../environments/README.md#requirements-and-targets)):

```bash
"$PY" -m tools.prepare download --model Qwen/Qwen3-0.6B
"$PY" -m tools.prepare download --model Qwen/Qwen3-1.7B
"$PY" -m compiler.qwen3.export   --model-dir work/models/qwen3-0.6b --output artifacts/qwen3-flash
```

Pinned revisions: `c1899de289a04d12100db370d81485cdf75e47ca` (0.6B) and
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (1.7B). See
[preparation](../../docs/development-tools.md#checkpoint-preparation-and-tokenization)
for verification, offline mode and tokenization.

One invocation compiles both entries into a fresh directory.
`--max-context-length` accepts `512` through the model's 40,960 ceiling
(defaulting to the full ceiling; e.g., pass `--max-context-length 16384` for
16K). Export uses BF16 cast emulation, TF32 disabled and standard Inductor
fusion. See [package metadata](../../docs/package-metadata.md) for bundle files
and [FlashAttention](../../docs/flash-attention.md) for native operator
packaging.

## Model interface

[model.py](model.py) owns loading, `KVCache` and `ModelStep`;
[adapters/attention.py](adapters/attention.py) replaces HF attention with
in-place Triton KV append (`xinfer::append_kv_batched`) and read-only
`xinfer::batched_flash_attention` while retaining HF projections, Q/K norm,
embeddings, RoPE and output head.

| Property | Qwen3-0.6B | Qwen3-1.7B |
| --- | --- | --- |
| Layers | 28 | 28 |
| Hidden / MLP | 1,024 / 3,072 | 2,048 / 6,144 |
| Query / KV heads | 16 / 8 | 16 / 8 |
| Head dimension | 128 | 128 |
| KV inputs | 56 BF16 views, each `[B,8,C,128]` | 56 BF16 views, each `[B,8,C,128]` |
| Output | Last-token BF16 logits `[B,151936]` | Last-token BF16 logits `[B,151936]` |

Geometry is derived from checkpoint config and `metadata.json` (any GQA head
multiple; head dim multiple of 8 in `8..256`). Python step signature:
`(input_ids, cache_position, (K0, V0, ..., K27, V27), kv_lengths, query_lengths)`
(60 flat AOTI inputs for 28 layers). See
[dynamic batch interfaces](../../docs/package-metadata.md#dynamic-batch-interfaces).
`runtime/core/src/spec.rs` checks model geometry; loading, state handling and
generation live in `runtime/core/src/`.

## Generate and validate

```bash
"$PY" -m tools.prepare encode --model-dir work/models/qwen3-0.6b   --prompt "What is the capital of France?" --output work/qwen3-prompt.ids
build/xinfer_aoti artifacts/qwen3-flash   work/qwen3-prompt.ids work/qwen3-generated.ids --max-output-tokens 64
"$PY" -m tools.prepare decode --model-dir work/models/qwen3-0.6b   --input work/qwen3-generated.ids

build/xinfer_serve artifacts/qwen3-flash --port 8000

AOTI_RUNTIME_CHECK_INPUTS=1 "$PY" -m tools.validation.validate   --packages artifacts/qwen3-flash --output artifacts/qwen3-validation-new
"$PY" -m tools.validation.qwen3.validate_runtime   --packages artifacts/qwen3-flash --output artifacts/qwen3-runtime-new
```

See [development tools](../../docs/development-tools.md#python-validation-tools)
for validation policy and
[benchmarks](../../docs/development-tools.md#benchmark-hf-and-native-aoti).

## Focused tests

After [CPU unit tests](../../docs/development-tools.md#cpu-unit-tests), run GPU
attention/export checks with `xinfer_attention_ops` built (uses random weights,
no checkpoint required):

```bash
"$PY" -m unittest compiler.common.tests.test_attention   compiler.qwen3.tests.test_attention compiler.qwen3.tests.test_export
```

## Numerical considerations

BF16 rounding is sensitive to RoPE multiply-add contraction and RMSNorm
reduction order; autotuning can change kernel reduction schedules even on
identical graphs. Matched-input logit bounds (`<= 5%`) and bitwise native/Python
parity govern acceptance, while HF state/token differences are diagnostic. See
[open numerical issues](../../docs/validation-status.md#open-numerical-issues).
