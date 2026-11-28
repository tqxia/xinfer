# Qwen3.8-27B-FP8 AOTI / Native Runtime

Owns Qwen3.8 defaults, FP8 checkpoint loading and export entry point, reusing
the [Qwen3.5 architecture and pipeline](../qwen35/README.md#model-interface-and-state).
Bundles use `model_type="qwen38"` and archive identity `model="qwen3.8"`. Shared
FP8 adapters/kernels live in `compiler/common/adapters/finegrained_fp8.py` and
`compiler/common/kernels/gemm/finegrained_fp8.py`; DeltaNet lives in
`compiler/common/adapters/gated_delta_net.py`.

`model.py` validates dynamic E4M3 FP8 (`128x128` blocks), untied BF16 embeddings
and output head, and default RoPE, restoring FP32 rotary buffers inline on load.
Tool frontends live in `tools/prepare/qwen38.py`, `tools/validation/qwen38/` and
`tools/benchmark/qwen38.py`.

This integration is experimental: GB10 bundles pass native bitwise parity and
runtime checks, while full-model HF logit comparisons still exceed the 5% limit
(see [validation status](../../docs/validation-status.md)).

## Prepare and export

Build `build/` via [environments](../../environments/README.md), then prepare
and export (pinned revision `c3a462cc3d38cf351f3b26cbd763469eeb74ef81`):

```bash
"$PY" -m tools.prepare download --model Qwen/Qwen3.8-27B-FP8
"$PY" -m compiler.qwen38.export --max-context-length 1024
```

Defaults: checkpoint `work/qwen38/27b-fp8/text`, bundle
`artifacts/qwen38/27b-fp8/text` (override with `--model-dir` and `--output`;
override native libraries with `XINFER_ATTENTION_OPS_LIBRARY` /
`XINFER_DELTANET_OPS_LIBRARY`). Always set `--max-context-length` explicitly
(the config's 262,144 ceiling requires ~129 GiB of state for 8 rows). See the
[shared Qwen3.5 export contract](../qwen35/README.md#prepare-and-export).

## FP8 execution

The text checkpoint contains 400 E4M3 FP8 projections with BF16 `128x128` weight
scales and dynamic per-token, per-128-value activation quantization; other
weights/activations stay BF16 and recurrent state stays FP32 (vision and MTP are
excluded).

HF eager uses Transformers' `FP8Linear` with
[`kernels-community/finegrained-fp8`](https://huggingface.co/kernels/kernels-community/finegrained-fp8)
pinned to `03ae31933313f38fae2b050dbe2731024386c369` (v4). Export uses
`xinfer::fp8_linear` (`compiler/common/kernels/gemm/finegrained_fp8.py`, fixed
32-row tile, bitwise identical to the pinned HF kernel) packaged into `.pt2`, so
native inference needs neither Python nor Hub access. Populate `KERNELS_CACHE`
before offline export or validation:

```bash
export KERNELS_CACHE="$PWD/work/qwen38/kernel-cache"
"$PY" - <<'PY'
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

FP8 linears skip BF16 linear packing; remaining BF16 projections use the shared
passes. Geometry: 64 layers (48 DeltaNet with 48 value heads; 16 full attention
with 24 query / 4 KV heads), 128 state tensors (package ceiling 8 rows; scalar
CLIs allocate 1 row). Text weights take ~27.44 GiB; 8-row state at `C=1024` adds
~1.65 GiB.

## Validation and tests

Validation compares the unmodified **FP8 HF checkpoint** against adapted eager,
Python AOTI and native AOTI (`cpp_aoti`) under the
[shared numerical policy](../../docs/development-tools.md#shared-numerical-policy)
(measuring execution fidelity, not quantization loss vs unquantized weights).

```bash
# CPU frontend checks
CUDA_VISIBLE_DEVICES='' "$PY" -m unittest   compiler.qwen38.tests.test_model compiler.qwen38.tests.test_export   tools.prepare.tests.test_common tools.prepare.tests.test_qwen38   tools.validation.tests.test_qwen38 tools.benchmark.tests.test_qwen38

# GPU FP8 projection/adapter checks (plus shared DeltaNet tests in compiler/qwen35/README.md#focused-tests)
"$PY" -m unittest   compiler.qwen38.tests.test_model.QuantizedCheckpoint   compiler.common.tests.test_finegrained_fp8_adapter

# Full-model validation and benchmarking
"$PY" -m tools.validation.validate   --packages artifacts/qwen38/27b-fp8/text   --model-dir work/qwen38/27b-fp8/text   --output artifacts/qwen38/27b-fp8/validation-new
"$PY" -m tools.validation.qwen38.validate_batch   --packages artifacts/qwen38/27b-fp8/text   --model-dir work/qwen38/27b-fp8/text   --output artifacts/qwen38/27b-fp8/batch-validation-new
"$PY" -m tools.validation.qwen38.validate_runtime   --output artifacts/qwen38/27b-fp8/runtime-new
"$PY" -m tools.benchmark run   --packages artifacts/qwen38/27b-fp8/text   --output artifacts/qwen38/27b-fp8/benchmark-new
```

Pass `--native` to numerical validation or benchmarking (and `--binary` /
`--cli` to `validate_runtime`) to override `build/`.
