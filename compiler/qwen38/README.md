# Qwen3.8-27B-FP8 AOTI / C++

This frontend owns Qwen3.8 defaults, checkpoint loading, FP8 export setup and
the export entry point. It reuses the dense Qwen3.5 text architecture, adapters,
export passes and native model ABI. Packages use `model_type="qwen38"`
and shared-weight identity `model="qwen3.8"`, selecting the shared Qwen3.5 runtime.
See [package metadata](../../docs/package-metadata.md) for the shared contract.
The checkpoint records the Qwen3.8 model ID and revision. Shared FP8 loading and
export adapters live in `compiler/common/adapters/finegrained_fp8.py`, with kernels in
`compiler/common/kernels/gemm/finegrained_fp8.py`.
The shared Qwen Gated DeltaNet adapter lives in
`compiler/common/adapters/gated_delta_net.py`.

`model.py` validates dynamic E4M3 FP8 with 128x128 weight blocks, untied
embeddings and default RoPE. Its loader preserves quantized weights and the
checkpoint's generation settings, with BF16 activations. Each loader constructs
its model on the meta device and restores FP32 rotary buffers inline. The
shared exporter receives the Qwen3.8 loader and step exporter as callbacks.
Validation and benchmarks select the loader using the package's `model_type`.
Validation also selects the frontend's export step for adapted eager execution.
The Qwen3.8 step installs FP8 adapters before invoking the shared graph export and packing passes;
loading alone preserves the original HF modules for reference execution.

Tool frontends live in `tools/prepare/qwen38.py`, `tools/validation/qwen38/`
and `tools/benchmark/qwen38.py`. Preparation routes checkpoint model IDs within
`tools/prepare/`; package-based tools dispatch directly by `model_type`.
The text extraction, validation and benchmark pipelines remain shared.

This is an experimental integration: the GB10 bundle runs natively and passes
native parity, but its full-model HF logit comparisons currently exceed the
required limits. See [validation status](../../docs/validation-status.md).

## Prepare and export

Select the independent `qwen38` environment and `build/qwen38/` using the
[environment guide](../../environments/README.md), then prepare and export with
the Qwen3.8 entry point:

```bash
"$Q38_PY" -m tools.prepare download --model Qwen/Qwen3.8-27B-FP8
"$Q38_PY" -m compiler.qwen38.export --max-context-length 1024
```

The defaults are `work/qwen38/27b-fp8/text` for prepared weights,
`artifacts/qwen38/27b-fp8/text` for exported packages, and `build/qwen38/` for
native compiler kernels. The preparation command selects the same checkpoint
directory. `--model-dir` and `--output` override the paths; existing checkpoints
prepared elsewhere can be reused with `--model-dir`.
`XINFER_ATTENTION_OPS_LIBRARY` can override the native attention library.

The checkpoint is pinned to `c3a462cc3d38cf351f3b26cbd763469eeb74ef81`.
Both prefill and decode are exported together. Omitted `--max-context-length`
uses the model's full configured capacity; select it explicitly as above.
`--enable-inductor-optimization` enables the same optional combo-kernel settings
as the shared exporter. See the
[shared export contract](../qwen35/README.md#prepare-and-export).

## FP8 execution

The text checkpoint has 400 E4M3 FP8 projections with BF16 128x128 weight
scales and dynamic per-token, per-128-value activation quantization. Other
parameters and activations remain BF16; recurrent state stays FP32. Preparation
retains the separate output head and `generation_config.json`, including both
EOS IDs. Vision and MTP weights are excluded.

HF eager uses Transformers' original `FP8Linear` with
[`kernels-community/finegrained-fp8`](https://huggingface.co/kernels/kernels-community/finegrained-fp8)
pinned to commit `03ae31933313f38fae2b050dbe2731024386c369` (v4). Export uses
xinfer's own `xinfer::fp8_linear` Triton kernel in
`compiler/common/kernels/gemm/finegrained_fp8.py` with a fixed 32-row launch tile so
dynamic batch/token dimensions stay symbolic (verified bitwise identical to the
pinned HF kernel). AOTI packages the kernels, so C++ inference needs neither
Python nor a Hub connection. Before offline validation or export, populate the
kernel cache for the pinned revision:

```bash
export KERNELS_CACHE="$PWD/work/qwen38/kernel-cache"
"$Q38_PY" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    "kernels-community/finegrained-fp8",
    repo_type="kernel",
    revision="03ae31933313f38fae2b050dbe2731024386c369",
    allow_patterns=["build/torch-cuda/*"],
    cache_dir=os.environ["KERNELS_CACHE"],
)
PY
```

For a disconnected target, transfer this cache from a connected host and set
`KERNELS_CACHE` there before enabling `HF_HUB_OFFLINE=1`. FP8 operators are not
BF16 linear-packing candidates; remaining BF16 projections use the existing passes.

This model has 64 layers, 48 DeltaNet layers with 48 value heads, and 16 full
attention layers with 24 query heads and four KV heads. Its 128 state tensors
reserve eight rows as usual. Text weights occupy about 27.44 GiB; state at
1,024-token capacity adds about 1.65 GiB. The config's default 262,144-token
capacity requires about 129 GiB of state alone: select capacity explicitly.

## Validation and tests

Validation compares the unmodified **FP8 HF checkpoint** against adapted eager,
Python AOTI and C++ AOTI. It keeps the existing logit limits and strict native
parity; it does not measure quantization loss against unquantized model weights.
The reference protocol records FP8 storage and the pinned kernel revision.
See [validation status](../../docs/validation-status.md) for tested full-model scope.

The [shared architecture guide](../qwen35/README.md#model-interface-and-state)
owns tensor/state interfaces and attention/DeltaNet implementation details.

Use `Q38_PY` from the environment guide. CPU frontend checks and GPU FP8
projection/AOTI checks run separately:

```bash
CUDA_VISIBLE_DEVICES='' "$Q38_PY" -m unittest \
  compiler.qwen38.tests.test_model compiler.qwen38.tests.test_export \
  tools.prepare.tests.test_common tools.prepare.tests.test_qwen38 \
  tools.validation.tests.test_qwen38 tools.benchmark.tests.test_qwen38
"$Q38_PY" -m unittest \
  compiler.qwen38.tests.test_model.QuantizedCheckpoint compiler.common.tests.test_finegrained_fp8_adapter
```

Shared architecture tests use the Qwen3.5 modules. DeltaNet kernel tests live in
`compiler/common/tests/gated_delta_net/`, including the 27B classes in
`test_recurrent` and `test_aoti`; see the
[focused test commands](../qwen35/README.md#focused-tests). Set
`XINFER_ATTENTION_OPS_LIBRARY="$Q38_BUILD/libxinfer_attention_ops.so"` when running
standalone attention and model export tests with the Qwen3.8 build.

Full-model tools select the Qwen3.8 frontend from `model_type="qwen38"` and
default to native executables in `build/qwen38/`, with the shared Qwen3.5 calling
convention:

```bash
"$Q38_PY" -m tools.validation.validate \
  --packages artifacts/qwen38/27b-fp8/text \
  --model-dir work/qwen38/27b-fp8/text \
  --output artifacts/qwen38/27b-fp8/validation-new
"$Q38_PY" -m tools.validation.qwen38.validate_batch \
  --packages artifacts/qwen38/27b-fp8/text \
  --model-dir work/qwen38/27b-fp8/text \
  --output artifacts/qwen38/27b-fp8/batch-validation-new
"$Q38_PY" -m tools.validation.qwen38.validate_runtime \
  --output artifacts/qwen38/27b-fp8/runtime-new
"$Q38_PY" -m tools.benchmark run \
  --packages artifacts/qwen38/27b-fp8/text \
  --output artifacts/qwen38/27b-fp8/benchmark-new
```

For a custom native build, pass `--native` to numerical validation or benchmarking;
runtime checks accept `--binary` and `--cli`. The runtime frontend also supplies
Qwen3.8 checkpoint and package defaults. Existing bundles retain their metadata
and can be reused with explicit checkpoint, package and native paths.
Follow the [development tools](../../docs/development-tools.md) for the shared
numerical policy, runtime checks and benchmark commands.
