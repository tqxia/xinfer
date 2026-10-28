# Qwen3.5 text-only AOTI / C++

The target is **Qwen/Qwen3.5-0.8B**, pinned to
`2fc06364715b967f1860aea9cf38778875588b17`. Vision and MTP are outside this
experiment. Python prepares the model and tokenizes/decodes offline; C++ loads
the compiled packages and performs generation.

Prefill is always dynamic: each input of 2–512 tokens uses one prefill call.
A single token uses the decode package. Batch size is one and context capacity
is 1,024. BF16 weights/activations, FP32 RoPE frequencies, and 48 explicit state
tensors cover 18 conv/recurrent pairs and six KV pairs. Recurrent states use
FP32. Inputs remain unchanged; returned tensors replace the previous state.

With packed packages, C++ and Python AOTI match bitwise across 12,201
teacher-forced tensors. Arithmetic and Chinese greedy output match HF; the
capital answer omits a period at an HF tie. HF numerical acceptance remains
**open**: conv/recurrent errors exceed the unchanged 5% limits, and
teacher-forced top-token differences include a nonzero HF margin. Final
performance measurements require an idle GPU.

## Source layout

- `model.py`: prepared text checkpoint, functional `TextStep` and state ABI.
- `adapters/delta.py`: reuse HF weights, projections, convolution and gated norm;
  replace the 18 DeltaNet cores with the Triton recurrence for prefill and decode.
- `kernels/recurrent.py`: kernel and `torch.library.triton_op` wrapper.
- `export.py`: strict export and AOTInductor packaging.
- `prepare.py`: offline token IDs; `reference.py`: HF/Python AOTI validation runners.
- `tests/`: core, model, C++ and end-to-end checks.
- `diagnostics/`: retained HF block experiments and focused numerical reproducers.

Native sources live in `runtime/qwen35/`; small shared helpers live in
`compiler/common/` and `runtime/common/`.

## Dynamic full-model prefill and Triton GDN

`torch.library.triton_op` keeps the GDN operator visible in the exported graph.
Its implementation uses `wrap_triton` so AOTInductor can compile the kernel and
include its cubin in each `.pt2` package. The C++ loader needs the two packages,
their JSON metadata and matching LibTorch/CUDA dependencies. It needs no Python,
Triton installation or separate custom-op library at inference time.

The separate cubin builder, CUDA-driver launcher and ATen chunk implementation
have been removed. Export has no backend or dynamic-prefill switch. Older fixed
or external-custom-op packages need to be re-exported for this runtime.

The kernel is a readable sequential baseline: one program owns one head and
32 value columns, carries FP32 state through T tokens, and fuses Q/K
normalization, decay, state update and output reduction. It supports contiguous
`[1,T,16,128]` Q/K/V, `[1,T,16]` gates and `[1,16,128,128]` state. The operator
wrapper establishes contiguous inputs and allocates fresh outputs. Q/K/V/beta
may be BF16 or FP32; log decay and state are FP32. **TODO: chunked prefill.**

## Prepare, export and build

Use `requirements.txt` in an isolated environment; the development environment
is `work/qwen35/venv`. Export-time Torch and native LibTorch must match. Optional
`fla` and `causal_conv1d` must be absent so the surrounding HF operations use the
validated tensor implementation. Rebuild and re-export on the target GPU and
toolchain; the existing SM89 artifacts are not a portable deployment bundle.

Run from the repository root. Export refuses to overwrite compiled packages;
use `--output` to select a fresh package directory for a repeated experiment.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
QWEN_PY=work/qwen35/venv/bin/python
# Keep large compilation scratch files off a small system /tmp filesystem.
mkdir -p work/qwen35/tmp work/qwen35/inductor-cache
export TMPDIR="$PWD/work/qwen35/tmp"
export TORCHINDUCTOR_CACHE_DIR="$PWD/work/qwen35/inductor-cache"
# Prepare once, or copy the prepared text directory including its manifest.
"$QWEN_PY" -m compiler.qwen35.model
"$QWEN_PY" -m compiler.qwen35.export prefill
"$QWEN_PY" -m compiler.qwen35.export decode

QWEN_CMAKE_PREFIX=$("$QWEN_PY" -c 'import torch; print(torch.utils.cmake_prefix_path)')
QWEN_SM=$("$QWEN_PY" -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')
cmake -S . -B work/qwen35/build -G Ninja \
  -DCMAKE_PREFIX_PATH="$QWEN_CMAKE_PREFIX" -DTORCH_CUDA_ARCH_LIST="$QWEN_SM" \
  -DCMAKE_BUILD_TYPE=Release -DQWEN_BUILD_QWEN35=ON \
  -DQWEN_BUILD_VALIDATION=ON -DQWEN_BUILD_BENCHMARK=ON
cmake --build work/qwen35/build \
  --target qwen35_aoti qwen35_validate qwen35_recurrent qwen35_benchmark -j2
```

Ensure `nvcc` is on `PATH`, or pass `-DCMAKE_CUDA_COMPILER=/path/to/nvcc`.
Export produces `prefill.pt2`, `decode.pt2` and their `.json` metadata under
`artifacts/qwen35/0.8b/text`. Metadata records `delta_backend=triton_op`, token
ranges, source checksums, toolchain and GPU architecture. Keep the package pair
and metadata together. The runtime rejects incompatible pairs.

[Torch 2.7's package loader](https://github.com/pytorch/pytorch/blob/v2.7.0/torch/csrc/inductor/aoti_package/model_package_loader.cpp#L42)
extracts directly into `/tmp`, ignoring `TMPDIR`.
Leave room there for both model libraries: about 3 GB plus margin for this pair.

## Generate and validate

```bash
"$QWEN_PY" -m compiler.qwen35.prepare encode \
  --prompt "What is 17 + 25?" --output work/prompt.ids
work/qwen35/build/qwen35_aoti artifacts/qwen35/0.8b/text \
  work/prompt.ids work/generated.ids
"$QWEN_PY" -m compiler.qwen35.prepare decode --input work/generated.ids

"$QWEN_PY" -m compiler.qwen35.tests.validate_recurrent \
  --output artifacts/qwen35/0.8b/recurrent-new
"$QWEN_PY" -m compiler.qwen35.tests.validate_model \
  --output artifacts/qwen35/0.8b/model-new
"$QWEN_PY" -m compiler.qwen35.tests.validate_runtime \
  --output artifacts/qwen35/0.8b/runtime-new
"$QWEN_PY" -m compiler.qwen35.tests.validate_end_to_end \
  --output artifacts/qwen35/0.8b/end-to-end-new
```

Use `--packages` and `--binary`/`--native`/`--cli` to select other artifact and
build directories. Validation output directories must be fresh. The recurrent
suite checks BF16/FP32, 15 lengths, zero/nonzero state, carried-state partitions,
repeats, input preservation and C++ execution through four core AOTI packages.
The native processes verify that neither Python nor custom-op libraries are
mapped. Full-model checks retain all state tensors, HF/FP32 controls, dispatch
counts, masked-future checks and CLI rejection cases. Numerical validators exit
nonzero for acceptance failures and retain the reports.

## Projection weight packing

Qwen3.5 export always runs the generic pass in
`compiler/common/passes/pack_linears.py` after strict export and before
Inductor decomposition. It combines independent, bias-free linear
projections sharing the same input. Weights must be frozen parameters with
exclusive storage and no other graph consumers; shared/tied/trainable weights
are skipped. Packed parameters are created once during export. Replaced
parameters are removed, and there is no inference-time weight concatenation.

For this model, 48 groups combine DeltaNet inputs, MLP gate/up and attention QKV,
reducing 187 linear calls to 97. Split outputs retain the original BF16 dtype
and contiguous layout for downstream views. Contiguous copies can cost time
in prefill; packing reduces launches rather than the number of weight bytes.
Packing is the default for both prefill and decode. No additional model
adapters or native libraries are needed.

```bash
"$QWEN_PY" -m compiler.qwen35.export prefill
"$QWEN_PY" -m compiler.qwen35.export decode
"$QWEN_PY" -m compiler.qwen35.tests.validate_runtime \
  --packages artifacts/qwen35/0.8b/text \
  --output artifacts/qwen35/0.8b/packing-runtime-new
"$QWEN_PY" -m compiler.qwen35.tests.validate_model \
  --packages artifacts/qwen35/0.8b/text \
  --output artifacts/qwen35/0.8b/packing-model-new
"$QWEN_PY" -m compiler.qwen35.tests.validate_end_to_end \
  --packages artifacts/qwen35/0.8b/text \
  --output artifacts/qwen35/0.8b/packing-end-to-end-new
```

Exports default to `artifacts/qwen35/0.8b/text`. Use a fresh `--output` directory
if older unpacked packages already occupy that path. The metadata's
`linear_packing` field records logical groups and policy version; C++ and Python
reject mismatched pairs. The separate `prefill.packing.json` / `decode.packing.json`
files also map FX outputs and source parameter names to each packed constant,
which helps profiling attribution.

The model validator adds `packing_vs_adapter` and `aoti_vs_packing` comparisons
for packed packages, using eager exported graphs to distinguish packing effects
from compilation. Changed matrix shapes can change floating-point reductions;
all original HF/FP32 checks and limits remain in place. Unpacked eager graphs
remain an internal numerical control; the export CLI has no packing switch.
Hold packing constant across compiler-tuning arms. For a packing A/B experiment,
use an unpacked control checkout with the same selected Inductor configuration.

## Benchmark

```bash
"$QWEN_PY" -m compiler.qwen35.benchmark run --gpu 0 \
  --validation artifacts/qwen35/0.8b/end-to-end-new/summary.json \
  --output artifacts/qwen35/0.8b/benchmark-new
```

Profiles compare HF full-prompt eager with native AOTI using identical fixed
tokens. EOS is data during timing. Reports retain numerical acceptance,
reset/prefill/decode timings, memory, workload, provenance and telemetry.
`--allow-busy` is for functional checks and marks timings provisional. Final
measurements require an idle environment.

[The Inductor tuning guide](../../docs/experiments/qwen35-inductor-tuning.md)
defines separate generated-GEMV and GEMM-autotuning experiments, package-pair
requirements, validation and native A/B measurement.

## Numerical issues and historical evidence

- **Q35-NUM-001:** lost BF16 rounding before DeltaNet gated normalization.
  `python -m compiler.qwen35.diagnostics.probe_rounding` reproduces the issue;
  exit zero means reproduced, not fixed.
- **Q35-NUM-002:** partition-sensitive greedy choices in earlier experiments.
  Triton full-model tests also record differences already present before
  Inductor; they cannot all be attributed to the lost rounding boundary.
- `python -m compiler.qwen35.diagnostics.probe_dynamic` records the original
  HF tensor fallback's bounded dynamic-export limitation. The Triton adapter
  avoids tracing that chunk scan.

[The experiment record](../../docs/experiments/qwen35.md) preserves earlier
fixed-prefill, ATen custom-op and manually packaged Triton results as history.
Their removed source files can be recovered from Git history or the recorded
source snapshots. The retained 4B HF single-block probe is diagnostic only.
Generated evidence remains ignored in `artifacts/qwen35/`; builds and caches
remain ignored in `work/`.
