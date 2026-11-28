# Qwen3 AOTI / Native Runtime

Qwen3 compiles BF16 AOTInductor with valid-prefix FlashAttention and native
grouped-query attention. The original Hugging Face model provides the weights
and an independent eager reference. The shared native runtime performs inference
and greedy generation without Python. Batch-one generation is the default;
one package also supports explicit inference steps with any batch size from 1 through 8.

Use the [environment guide](../../environments/README.md) for setup/builds,
[development tools](../../docs/development-tools.md) for preparation, validation
and benchmarks, and [validation status](../../docs/validation-status.md) for
recorded results and unresolved issues. Historical passes do not cover every
capacity or newly compiled artifact.

## Dynamic-batch inference

Every export supports `B=1..8`, with independent prompt lengths and cache
positions. The range is fixed in code; there is no batch-size export option.
`Runner::step_batch` accepts `B` token vectors, pads them on the right and
returns `[B,vocab_size]` logits for each row's last real token. A call consumes
at most 512 real tokens per row. Width one uses decode; widths 2–512 use the
dynamic prefill package, even if one row has only one token.

The runtime allocates eight cache rows once and passes views of the first `B`
rows to each call. Row indices retain their identity: reducing `B` preserves
omitted rows; increasing it resumes those rows or starts previously unused rows.
Each row's first input must be nonempty. Later empty rows preserve cache and
position and return zero logits. At least one row must be active. `reset()`
clears all eight rows; individual request replacement and EOS scheduling remain
future work. Split longer prompts into calls at the API boundary.

Context capacity is per row. At capacity 1,024 the eight-row allocation is
896 MiB, independent of the current `B`, in addition to weights and temporary
buffers. The scalar generation and benchmark CLIs use B=1 from this same bundle.

```bash
"$PY" -m compiler.qwen3.export --model-dir work/models/qwen3-0.6b \
  --max-context-length 1024 --output artifacts/qwen3-batched
"$PY" -m tools.validation.qwen3.validate_batch \
  --model-dir work/models/qwen3-0.6b --packages artifacts/qwen3-batched \
  --output artifacts/validation-batched
```

The batch validator exercises every B=1..8 on that bundle, comparing distinct
text prompts and boundary cases against independent BF16 HF requests, adapted
eager and Python/C++ AOTI. It checks the shared 5% logit bounds, bitwise native
parity, padding, unused rows, cache ownership, independent positions, inactive
and resumed rows, changing B, reset, poisoned future KV, 511/512/513 boundaries,
full capacity and overflow. HF, adapted eager and Python AOTI are each loaded
once for the entire matrix. Common and multi-row fixtures run for each B;
batch-transition fixtures run once afterward. Per-batch reports live in
`b1/` through `b8/`, with transition results in `transitions/`; `summary.json`
uses these group names as keys. After the Python phase releases its models,
one native runtime executes all cases. Expected cache traces use sparse files with
unchanged bytes; C++ compares complete logits/state files byte for byte and
removes successful state dumps case by case. Reports retain their digests.
HF state differences remain diagnostic. See
[validation status](../../docs/validation-status.md) for results.

## Prepare and export

Run from the repository root with `PY` selected through the environment guide.
Build `xinfer_attention_ops` and the native executables in `build/` before
export. Export-time Torch and native LibTorch must come from the same installation.
Dependency pins live in the target
[requirements files](../../environments/README.md#requirements-and-targets).

```bash
"$PY" -m tools.prepare download --model Qwen/Qwen3-0.6B
"$PY" -m tools.prepare download --model Qwen/Qwen3-1.7B
"$PY" -m compiler.qwen3.export \
  --model-dir work/models/qwen3-0.6b --output artifacts/qwen3-flash
```

The pinned checkpoint revisions are `c1899de289a04d12100db370d81485cdf75e47ca`
(Qwen3-0.6B) and `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (Qwen3-1.7B).
Preparation downloads the required HF files (single-file or sharded safetensors)
and writes `model_manifest.json`.
Reuse a prepared directory on later exports; run download before enabling Hub
or Transformers offline mode. See the
[preparation commands](../../docs/development-tools.md#checkpoint-preparation-and-tokenization)
for verification and tokenization.

One export invocation compiles both prefill and decode. Use a new or empty output
directory. `--max-context-length` selects capacity from 512 through the model's
40,960-position ceiling; omitting it allocates that full ceiling. For example,
add `--max-context-length 16384` for a 16K bundle. Capacity is checked before GPU
model loading and changing it requires re-export.

There is no attention-backend or numerical-mode selector. Export uses BF16 cast
emulation, disabled TF32 and ordinary Inductor fusion. Compiler/tuning choices
can change rounding, so validate each new pair. The
[package contract](../../docs/package-metadata.md) defines the four deployment
files, shared-weight ownership and compatibility checks; graph dumps and export
logs are build diagnostics. The
[shared attention guide](../../docs/flash-attention.md) explains native packaging.

## Model interface

[model.py](model.py) owns loading, KV allocation, `KVCache` and the full-model
`ModelStep`. [adapters/attention.py](adapters/attention.py) owns the
`FlashAttention` layer replacement and its installation.

The adapter retains the original embedding, rotary embedding, decoder layers,
final normalization and vocabulary head. Each attention wrapper reuses HF
projections and Q/K normalization, appends K/V in place and invokes the shared
read-only batched FlashAttention operator over the populated prefixes. Each
layer uses one Triton KV-write launch and one native variable-length attention
call across the batch. Metadata is prepared once per step and reused by all
layers; packing excludes padded queries. See the [attention guide](../../docs/flash-attention.md)
for cache-layout handling.

| Property | Qwen3-0.6B | Qwen3-1.7B |
| --- | --- | --- |
| Layers | 28 | 28 |
| Hidden / MLP | 1,024 / 3,072 | 2,048 / 6,144 |
| Query / KV heads | 16 / 8 | 16 / 8 |
| Head dimension | 128 | 128 |
| KV inputs | 56 BF16 cache views, each `[B,8,C,128]` | 56 BF16 cache views, each `[B,8,C,128]` |
| Output | Last-token BF16 logits `[B,151936]` | Last-token BF16 logits `[B,151936]` |

The adapter and runtime derive layer count, KV heads and head dimension from the
checkpoint config and package metadata (supporting any FlashAttention-compatible
GQA head multiple and head dimension multiple of 8 in `8..256`).

`B` is the runtime batch size (1 through 8), and `C` is the exported capacity.
All batch sizes use the same Python step signature:
`(input_ids, cache_position, (K0, V0, ..., K27, V27), kv_lengths, query_lengths)`.
IDs and positions have shape `[B,T]`; the two CPU int32 length tensors have
shape `[B]`. KV inputs are views of the first B rows of the runtime's allocated
cache (1 through 8 rows; scalar CLI and benchmark allocate 1 row, defaulting to
8). Native AOTI always receives 60 flat inputs. See
[the dynamic batch interface](../../docs/package-metadata.md#dynamic-batch-interfaces)
for padding, inactive rows and state ownership.
Training, arbitrary masks and attention-weight outputs are unsupported.

Prefill accepts padded widths 2–512 per call; decode accepts width one,
including at position zero. The shared runner chunks longer prompts and requires
prompt plus output budget to fit capacity.
`runtime/core/src/spec.rs` checks model-specific geometry; loading, state handling
and generation live in `runtime/core/src/`.

## Generate and validate

```bash
"$PY" -m tools.prepare encode --model-dir work/models/qwen3-0.6b \
  --prompt "What is the capital of France?" --output work/qwen3-prompt.ids
build/xinfer_aoti artifacts/qwen3-flash \
  work/qwen3-prompt.ids work/qwen3-generated.ids --max-output-tokens 64
"$PY" -m tools.prepare decode --model-dir work/models/qwen3-0.6b \
  --input work/qwen3-generated.ids

AOTI_RUNTIME_CHECK_INPUTS=1 "$PY" -m tools.validation.validate \
  --packages artifacts/qwen3-flash --output artifacts/qwen3-validation-new
"$PY" -m tools.validation.qwen3.validate_runtime \
  --packages artifacts/qwen3-flash --output artifacts/qwen3-runtime-new
```

Use fresh token-output files and validation directories. The numerical chain
creates HF references during validation and cross-checks manual HF greedy
execution against `HF.generate()` before installing adapters. It follows the
[shared numerical policy](../../docs/development-tools.md#shared-numerical-policy).
The separate runtime tool checks shipping CLI/deployment behavior.
For performance work, use the shared
[benchmark commands](../../docs/development-tools.md#benchmark-hf-and-native-aoti).

## Focused tests

Run the [CPU tests](../../docs/development-tools.md#cpu-unit-tests) in the matching
model environment. With CUDA visible and the attention library built:

```bash
"$PY" -m unittest compiler.common.tests.test_attention \
  compiler.qwen3.tests.test_attention compiler.qwen3.tests.test_export
```

These use random weights without a checkpoint download. Operator tests cover
native/compiled attention; adapter and export tests cover cache mutations,
shape/length handling, rejection/reuse and production graph structure.
Structural export tests do not compile full-model AOTI packages. Full-model
validation remains a separate requirement when changing the implementation.

## Numerical considerations

BF16 rounding is sensitive to RoPE multiply-add contraction and RMSNorm reduction
order. Earlier probes isolated both effects; even identical graph text and
weights could produce different outputs when autotuning changed a reduction's
launch configuration. A retained tuning cache helps comparisons but does not
prove unchanged compiler choices.

The retired eager-numerics profile matched BF16 eager on its recorded cases,
but was not consistently closer to an independent FP32 control. It is therefore
not an alternate supported export path. Bounded matched-input logit error and
strict native/Python parity remain the acceptance criteria; HF state and token
differences remain visible diagnostics. The
[validation status](../../docs/validation-status.md#open-numerical-issues)
records the unresolved logit failures and tested profiles.
