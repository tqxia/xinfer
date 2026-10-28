# Qwen3.5 Inductor tuning experiments

Use commit `6a0b4c7` (dynamic prefill plus packaged Triton GDN) as the control.
The old fixed-prefill commit `f67bdef` is not the control for this experiment.
The aim is to improve batch-one BF16 decode, especially its linear projections,
without mixing in weight packing, quantization, new kernels or CUDA Graphs.
Run these original tuning arms from the pinned unpacked control checkout.
The latest exporter always packs linears and has no packing switch. If tuning
the latest code instead, keep packing enabled in every arm and record that
baseline change; its projection count is 97. Compare packed and unpacked
checkouts separately once a tuning configuration has been selected.

Host 103 measured about 3.70 ms/token: 187 GEMVs account for about 75% of summed
GPU kernel time. The 248,320-by-1,024 LM-head matrix contains about 509 MB of
BF16 weights. There is a weight-bandwidth limit; a smaller kernel count or
higher occupancy alone does not establish a speedup.

This guide specifies experiments, not measured tuning wins. Current HF numerical
acceptance remains open. Preserve all existing precision settings and thresholds.

## 1. Environment and experiment arms

Run in a separate test checkout on the measurement host, using its existing
model environment and matching LibTorch. Keep the same Torch/CUDA/Triton/ABI,
checkpoint, GPU, CPU-thread settings and target-only version-guard patch as the
host-103 comparison. Do not upgrade dependencies during this comparison.
The source checks behind this guide use Torch 2.7.0; host 103 uses 2.7.0a0, so
verify option availability and generated code there.

In `compiler/qwen35/export.py`, replace the existing `options` assignment with
this explicit common configuration, then add the chosen arm's override:

```python
        options = {
            "emulate_precision_casts": True,
            "max_autotune": False,
            "max_autotune_gemm": False,
            "max_autotune_pointwise": False,
            "coordinate_descent_tuning": False,
        }
        options.update({})  # Replace {} with exactly one arm below.
```

Use a separate output directory for every arm. Export **both** entry points
before changing the source to another arm. The exporter records this dictionary
in package metadata; the runtime requires matching settings and source hashes
for the package pair. Copy the modified exporter into that arm's evidence.
Do not change global container dependencies or the development checkout.

| Arm | Override passed to `options.update(...)` | Purpose |
| --- | --- | --- |
| `e0-control` | `{}` | Explicit untuned control on the current model path |
| `e1-matvec` | `{"coordinate_descent_tuning": True}` | Generated matrix-vector reduction plus coordinate-descent search |
| `e2-gemm` | `{"max_autotune_gemm": True, "max_autotune_gemm_backends": "ATEN,TRITON"}` | Compare available ATen and Triton GEMM candidates |
| `e3-combined` | Union of `e1` and `e2` | Optional second stage, after measuring the individual arms |
| `e4-exhaustive` | `e2` plus `{"max_autotune_gemm_search_space": "EXHAUSTIVE"}` | Optional larger GEMM search if `e2` is promising; expect more compile cost |

Start with e0, e1 and e2. Keep the DEFAULT search space for e2. If e3 is the
winner, an exhaustive search can instead be added to that configuration, under
a new arm name. Do not silently drop unsupported options; mark the arm unsupported.

**What e1 actually changes:** Torch 2.7's `aten.mm` decomposition turns CUDA
M=1 or N=1 matrix products into multiply-plus-reduction when coordinate descent
is enabled. This can replace cuBLAS GEMV with generated Triton reductions and
allow adjacent operations to fuse. Its opmath wrapper promotes BF16 computation
to FP32 and casts the result back, but the reduction order can still change.
It can also tune other generated kernels; this is not an isolated GEMV-only switch.

**What e2 changes:** Inductor benchmarks eligible GEMM candidates from the
specified backends. It may still choose ATen/cuBLAS. This does not exhaustively
tune cuBLAS's internal algorithm space. With e1 enabled, M=1 products may already
be reductions, so e3 is not simply two searches over the same GEMV candidates.

Do not use a broad `max_autotune=True` or a `torch.compile` mode as a substitute
for these named arms. Mode presets can enable several options together. AOTI
already generates a C++ wrapper; changing `cpp_wrapper` is not this experiment.
Do not change TF32, `use_fast_math`, precision casts, `--eager-numerics`, attention
backend, or the GDN kernel's warp/tile settings. CUDA Graph replay is separate
runtime work; a compiler flag alone does not provide it in our C++ loop.

Avoid environment-only tuning for these runs. The current exporter does not
record implicit environment overrides in `inductor_configs`, which could make
differently tuned package pairs appear compatible. Use explicit dictionaries,
a clean environment, and record any additional nondefault settings.

## 2. Prepare and export one arm

All commands run from the test checkout root. Adjust the Python/build paths to
those already validated on the target host. Use Bash for these examples.

```bash
QWEN_PY=work/qwen35/venv/bin/python
QWEN_BUILD=work/profile103/build  # Or work/qwen35/build in a normal checkout.
QWEN_GPU=1                     # Physical nvidia-smi index on host 103.
QWEN_UUID=$(nvidia-smi -i "$QWEN_GPU" --query-gpu=uuid --format=csv,noheader)
export CUDA_VISIBLE_DEVICES="$QWEN_UUID"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Change this after selecting the corresponding options dictionary.
QWEN_ARM=e0-control
QWEN_RUN="artifacts/qwen35/0.8b/inductor-tuning/$QWEN_ARM"
mkdir -p "$QWEN_RUN" "work/inductor-tuning/$QWEN_ARM/tmp" \
  "work/inductor-tuning/$QWEN_ARM/cache"
export TMPDIR="$PWD/work/inductor-tuning/$QWEN_ARM/tmp"
export TORCHINDUCTOR_CACHE_DIR="$PWD/work/inductor-tuning/$QWEN_ARM/cache"
```

Use fresh caches per arm. Unset inherited tuning switches before importing
Torch in the export subprocesses:

```bash
unset TORCHINDUCTOR_MAX_AUTOTUNE TORCHINDUCTOR_MAX_AUTOTUNE_GEMM
unset TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE TORCHINDUCTOR_COORDINATE_DESCENT_TUNING
unset TORCHINDUCTOR_COORDINATE_DESCENT_CHECK_ALL_DIRECTIONS
unset TORCHINDUCTOR_COORDINATE_DESCENT_RADIUS
unset TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE
```

Verify the installed config accepts the requested keys before starting an arm;
use `getattr(torch._inductor.config, key)` for the keys in its dictionary.
Record Torch's full version/git version, Triton/CUDA versions, C++ ABI, GPU UUID
and SM, source commit/diff, and the complete options dictionary. Preserve the
host-103 exact-build guard; do not weaken version checks to make a package load.

**Autotuning itself needs an idle GPU.** A busy compile can bake in poor kernel
choices even when the final benchmark runs idle. Serialize all compilation,
validation and timing GPU work. Use the same five-sample <=10% idle guard before
and after export, without `--allow-busy`:

```bash
(
set -e
cp compiler/qwen35/export.py "$QWEN_RUN/export.py"
git diff -- compiler/qwen35/export.py > "$QWEN_RUN/export.patch"
"$QWEN_PY" -m compiler.qwen35.benchmark check-idle --gpu "$QWEN_GPU" \
  --output "$QWEN_RUN/before-export-idle.json"
TORCH_LOGS=output_code "$QWEN_PY" -u -m compiler.qwen35.export prefill \
  --output "$QWEN_RUN/packages" > "$QWEN_RUN/prefill-export.log" 2>&1
TORCH_LOGS=output_code "$QWEN_PY" -u -m compiler.qwen35.export decode \
  --output "$QWEN_RUN/packages" > "$QWEN_RUN/decode-export.log" 2>&1
"$QWEN_PY" -m compiler.qwen35.benchmark check-idle --gpu "$QWEN_GPU" \
  --output "$QWEN_RUN/after-export-idle.json"
)
```

Stop after any failed export or idle check. Existing package paths are not
silently overwritten. Check both JSON files contain the selected options and
matching source/checkpoint/state/toolchain fields. Retain compile time, package
size/checksum and the autotuning logs. Note that the export sample for dynamic
prefill has T=32; test all prompt lengths rather than assuming one tuned shape
wins everywhere.

Torch 2.7's package loader extracts directly into `/tmp`, ignoring `TMPDIR`.
Leave roughly 3 GB plus margin there for one loaded package pair. The variables
above move compilation scratch/cache, not the C++ loader's extraction directory.

## 3. Validate before interpreting performance

Run these commands individually, checking each result as described below.
Use the unchanged validators and the same native binaries for all arms:

```bash
"$QWEN_PY" -m compiler.qwen35.tests.validate_runtime \
  --packages "$QWEN_RUN/packages" --binary "$QWEN_BUILD/qwen35_validate" \
  --cli "$QWEN_BUILD/qwen35_aoti" --output "$QWEN_RUN/runtime"
"$QWEN_PY" -m compiler.qwen35.tests.validate_model \
  --packages "$QWEN_RUN/packages" --output "$QWEN_RUN/model"
"$QWEN_PY" -m compiler.qwen35.tests.validate_end_to_end \
  --packages "$QWEN_RUN/packages" --native "$QWEN_BUILD/qwen35_validate" \
  --output "$QWEN_RUN/end-to-end"
```

Runtime validation must pass, including bitwise C++/Python AOTI parity, input
preservation, carried state, dispatch, repeated requests and rejection checks.
Full numerical validators can exit 1 for the already-open HF acceptance gates.
Require complete reports with `native_bitwise=true`; a crash, missing report,
new mutation, or invalid output is not an expected numerical failure.

Compare each arm against e0 on the same cases: logit/state maximum errors,
per-state failure counts, teacher top-token differences with margins, and chat
outputs. Check the length-22 benchmark prediction differences at indices 1 and
31 explicitly; do not assume they are ties. Keep the FP32 controls and all
existing tolerances. No arm closes HF acceptance merely by matching e0.

Inspect generated code and package contents as well as timings:

- Did e1 actually replace the intended M=1 products with fused reductions?
- Did e2 select new GEMM kernels, or keep ATen/cuBLAS?
- Are FP32 multiply/reduction casts fused into the kernel? Avoid materializing
  a vocabulary-by-hidden FP32 product or a full expanded weight temporary.
- Are BF16 result boundaries retained before nonlinear consumers and argmax?
- Does peak memory or package size grow unexpectedly?

A drop in the number of kernels named `gemv` can simply mean projections moved
into the generic Triton category. Attribute work by projection/weight and source
origin, not kernel-name categories alone. Capture profiles separately from clean
timing; compare operator/kernel timings only within matching measurement scopes.

## 4. Measure native control versus native candidate

Use the current untuned e0 packages as A and one tuned arm as B, with the same
rebuilt runtime binary. Reuse the exact host-103 fixed-token workload:

```bash
QWEN_CONTROL=artifacts/qwen35/0.8b/inductor-tuning/e0-control/packages
QWEN_WORKLOAD=artifacts/qwen35/0.8b/comparison103/clean-workload.json
```

That workload has six prompt lengths, two additional warmups, five measured
requests per worker and 64 fixed decode inputs. It produces 65 predictions per
request, including the prediction after prefill. EOS remains data. Keep input
IDs identical across arms.

Use the existing host-103 paired-comparison driver with package paths changed,
or alternate fresh native workers as A/B, B/A, A/B, B/A. For example, one worker:

```bash
(
set -e
"$QWEN_PY" -m compiler.qwen35.benchmark check-idle --gpu "$QWEN_GPU" \
  --output "$QWEN_RUN/before-worker-idle.json"
CUDA_VISIBLE_DEVICES="$QWEN_UUID" "$QWEN_BUILD/qwen35_benchmark" \
  "$QWEN_RUN/packages" "$QWEN_WORKLOAD" "$QWEN_RUN/candidate-round-1.json" \
  > "$QWEN_RUN/candidate-round-1.log" 2>&1
"$QWEN_PY" -m compiler.qwen35.benchmark check-idle --gpu "$QWEN_GPU" \
  --output "$QWEN_RUN/after-worker-idle.json"
)
```

Use unique filenames and idle checks for every worker; collect GPU telemetry
throughout each worker as in the previous comparison. The shared
`compiler.common.benchmark.run_worker` already supplies that mechanism. Do not
run a profiler during clean measurements. The regular `benchmark run` command
compares HF with one native package pair; it is not a native A-versus-B driver.

Report pooled warmed medians/p10/p90, per-round paired ratios and prediction
agreement for every length, not only the best length. Four process rounds with
five measured samples give 20 samples per arm/length. Keep load time and
`first_request` separate: the reported 16.4x length-22 gain was warmed latency,
not cold startup. Retain first-prediction, decode-ms/token and whole-request
results; optimize decode without hiding a prefill regression.

Recommended result columns:

| Arm | Effective options | Decode median and paired ratios | Prefill by length | First-use/load | Memory/package bytes | Numerical changes | Generated projection kernels |
| --- | --- | --- | --- | --- | --- | --- | --- |
| e0 | Explicit untuned | Control | Control | Control | Control | Existing failures | Current cuBLAS baseline |

Promote only repeatable improvements larger than observed run-to-run variation.
A configuration that compiles successfully or produces fewer named GEMVs has
not established a speedup. After choosing a tuning configuration, test weight
packing as a separate arm with that configuration held fixed. Packing changes
matrix shapes and may change which configuration wins.

## Sources and scope

- [Torch 2.7 config definitions](https://github.com/pytorch/pytorch/blob/v2.7.0/torch/_inductor/config.py)
- [M=1 decomposition and opmath promotion](https://github.com/pytorch/pytorch/blob/v2.7.0/torch/_inductor/decomposition.py#L289)
- [ATen/Triton GEMM candidate selection](https://github.com/pytorch/pytorch/blob/v2.7.0/torch/_inductor/kernel/mm.py#L356)
- [AOTI config forwarding](https://github.com/pytorch/pytorch/blob/v2.7.0/torch/_inductor/__init__.py)
- [Config scope before compilation/decomposition](https://github.com/pytorch/pytorch/blob/v2.7.0/torch/_inductor/compile_fx.py)
- [Model build, runtime and numerical issues](../../compiler/qwen35/README.md)

`decompose_mem_bound_mm` is not the proposed M=1 tuning route here. Modern
backend examples requiring newer Torch versions or Blackwell GPUs should not
be copied into this pinned L20 experiment without a separate compatibility study.
