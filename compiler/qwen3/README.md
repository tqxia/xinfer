# Qwen3-0.6B AOTI / C++

Qwen3 has one compiled profile: batch-one BF16 AOTInductor with valid-prefix
FlashAttention and native grouped-query attention. The original Hugging Face
model provides the weights and an independent eager reference. The shared C++
runtime performs inference and greedy generation without Python.

Use the [environment guide](../../environments/README.md) for setup/builds,
[development tools](../../docs/development-tools.md) for preparation, validation
and benchmarks, and [validation status](../../docs/validation-status.md) for
recorded results and unresolved issues. Historical passes do not cover every
capacity or newly compiled artifact.

## Prepare and export

Run from the repository root with `Q3_PY` selected through the environment guide.
Build `xinfer_attention_ops` and the native executables in `build/qwen3/` before
export. Export-time Torch and native LibTorch must come from the same installation.
The adapter checks its pinned Transformers version before importing private HF
implementation code. Dependency pins live in the per-device
[requirements files](../../environments/README.md#requirements-and-targets).

```bash
"$Q3_PY" -m tools.prepare download --model Qwen/Qwen3-0.6B
"$Q3_PY" -m compiler.qwen3.export \
  --model-dir work/models/qwen3-0.6b --output artifacts/qwen3-flash
```

The pinned checkpoint revision is `c1899de289a04d12100db370d81485cdf75e47ca`.
Preparation downloads the required HF files and writes `model_manifest.json`.
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
[package contract](../../docs/package-metadata.md) defines the five deployment
files, shared-weight ownership and compatibility checks; graph dumps and export
logs are build diagnostics. The
[shared attention guide](../../docs/flash-attention.md) explains native packaging.

## Model interface

The adapter retains the original embedding, rotary embedding, decoder layers,
final normalization and vocabulary head. Each attention wrapper reuses HF
projections and Q/K normalization, appends K/V in place and invokes the shared
read-only FlashAttention operator over the populated prefix.

| Property | Qwen3-0.6B |
| --- | --- |
| Layers | 28 |
| Query / KV heads | 16 / 8 |
| Head dimension | 128 |
| KV tensors | 56 caller-owned BF16 buffers, each `[1,8,C,128]` |
| Output | Last-token BF16 logits `[1,151936]` |

`C` is the exported capacity. The Python step signature is
`(input_ids, cache_position, (K0, V0, ..., K27, V27), kv_extent)`.
Native AOTI receives 59 flat inputs: IDs, positions, the 56 KV tensors and the
shape-only extent. Inputs are unpadded contiguous appends to an existing prefix.
Training, arbitrary masks and attention-weight outputs are unsupported.

[Runtime ABI 1](../../docs/package-metadata.md#runtime-abi-1) defines shared
input types, mutation and dispatch semantics. Prefill accepts 2–512 tokens per
call; decode accepts one token, including at position zero. The shared runner
chunks longer prompts and requires prompt plus output budget to fit capacity.
`runtime/qwen3/runtime.h` checks model-specific geometry; loading, state handling
and generation live in `runtime/common/`.

## Generate and validate

```bash
"$Q3_PY" -m tools.prepare encode --model-dir work/models/qwen3-0.6b \
  --prompt "What is the capital of France?" --output work/qwen3-prompt.ids
build/qwen3/xinfer_aoti artifacts/qwen3-flash \
  work/qwen3-prompt.ids work/qwen3-generated.ids --max-output-tokens 64
"$Q3_PY" -m tools.prepare decode --model-dir work/models/qwen3-0.6b \
  --input work/qwen3-generated.ids

AOTI_RUNTIME_CHECK_INPUTS=1 "$Q3_PY" -m tools.validation.validate \
  --packages artifacts/qwen3-flash --output artifacts/qwen3-validation-new
"$Q3_PY" -m tools.validation.qwen3.validate_runtime \
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
"$Q3_PY" -m unittest compiler.common.tests.test_attention \
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
[validation status](../../docs/validation-status.md#numerical-findings-to-preserve)
records the unresolved focused logit failure and historical evidence.
