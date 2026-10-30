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

One functional dispatcher selects fused recurrence for T<=97 and B3/C64
chunked FP32 prefill for longer requests. The threshold is provisional. Focused
validation compares it with HF recurrent/chunked operations and an independent
FP32 control; full-model and native checks cover generation and carried state.
HF/compiler numerical acceptance remains **open**. The experiment record retains
the earlier validation results, including the retired standalone-core control.
No timing result is claimed on the busy development GPU.

## Source layout

- `model.py`: prepared text checkpoint, functional `TextStep` and state ABI.
- `adapters/delta.py`: reuse HF weights, projections and gated norm; call unified
  Triton preparation/recurrence for prefill and decode.
- `kernels/recurrent.py`: fused recurrence, parallel preparation, functional
  branch operators and shape-based `torch.cond` dispatch.
- `kernels/chunked.py`: C64 FP32 chunking with B3's 32-column U/W transform.
- `export.py`: strict export and AOTInductor packaging.
- `prepare.py`: offline token IDs; `reference.py`: HF/Python AOTI validation runners.
- `tests/`: fused-operator, model, C++ and end-to-end checks.
- `diagnostics/`: retained HF block experiments and focused numerical reproducers.

Native sources live in `runtime/qwen35/`; small shared helpers live in
`compiler/common/` and `runtime/common/`.

## Conditional GDN validation

The standalone test exercises the production kernels directly. T<=97 uses the
existing fused recurrence; longer inputs use C64 chunking with the measured B3
32-column U/W transform. Threshold 97 comes from an isolated-kernel crossover
and still needs full-model performance evaluation on an idle target.

The condition uses `qkv.shape[1]` directly. Each branch owns fresh output/state
tensors, and only the long branch allocates chunk workspaces. Strict export
retains one dynamic prefill range, T=2..512. AOTI packages both branches and
generates a native shape comparison; T=1 uses a separate decode package.

```bash
"$QWEN_PY" -m compiler.qwen35.tests.validate_cond \
  --binary work/qwen35/build/qwen35_fused \
  --output artifacts/qwen35/0.8b/cond-validation-new
```

The output must be fresh. The validator retains exported/decomposed graphs,
packaged C++ wrappers/cubin hashes, raw state fixtures and numerical reports.
It reuses the existing Python-free C++ runner. `dispatch_passed` reports the
dispatch/backend regression gate separately from the existing HF/FP32 numerical
acceptance; inspect both. Noncontiguous inputs are checked through the Python
dispatcher; exported package inputs retain the contiguous-layout ABI.

## Dynamic full-model prefill and Triton GDN

`torch.cond` retains both GDN branches in the exported graph. Each branch uses
`torch.library.triton_op` and `wrap_triton` so AOTInductor can compile the kernels and
include their cubins in each `.pt2` package. The C++ loader needs the two packages,
their JSON metadata and matching LibTorch/CUDA dependencies. It needs no Python,
Triton installation or separate custom-op library at inference time.

The separate cubin builder, CUDA-driver launcher and ATen chunk implementation
have been removed. Export has no backend or dynamic-prefill switch. Older fixed
or external-custom-op packages need to be re-exported for this runtime.

The adapter retains the same functional tensor interface for both entry points:

```text
fused(qkv, b, a, conv, state, weight, A_log, dt_bias)
    -> core, next_conv, next_recurrent
```

QKV is `[1,T,6144]`, A/B are `[1,T,16]`, convolution state is `[1,6144,4]`,
convolution weights are `[6144,4]`, and decay parameters are `[16]`. Recurrent
state is `[1,16,128,128]`; core output is `[1,T,16,128]`. Inputs and parameters
use matching BF16 or FP32, with recurrent state always FP32. The wrapper
establishes contiguous inputs and allocates fresh outputs.

Decode keeps the existing fused kernel: one program owns one head and 32 value
columns and combines bias-free width-four convolution/history handling, SiLU,
beta/log-decay preparation, Q/K normalization, state update and output reduction.
Four value tiles share Q/K preparation; only tile zero writes their convolution
history.

Short prefill reuses fused recurrence. Long prefill prepares Q/K/V and gates in
parallel across tokens and heads, then runs
four chunk kernels: cumulative decay and triangular solve, U/W transforms,
state propagation between chunks, and parallel outputs. The fixed chunk size
is 64. It is an internal compute tile; dynamic T=2..512 still uses one prefill
package, with masked neutral tails rather than separate length buckets. The
U/W output-feature tile is 32; the dot reduction still covers all 64 chunk rows.
Workspaces, state arithmetic and matrix products use FP32 with explicit IEEE
dot precision. Convolution/SiLU and beta retain their input-dtype rounding.

Both paths retain the same eight-input/three-output operator interface and
require no extra flag. Input projections, z-gated normalization and output
projection stay outside. Focused validation uses HF recurrent and chunked
references; the runtime continues to load packaged Triton without Python.

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
  --target qwen35_aoti qwen35_validate qwen35_fused qwen35_benchmark -j2
```

Ensure `nvcc` is on `PATH`, or pass `-DCMAKE_CUDA_COMPILER=/path/to/nvcc`.
Export produces `prefill.pt2`, `decode.pt2` and their `.json` metadata under
`artifacts/qwen35/0.8b/text`. Metadata records `delta_backend=triton_op`,
the short/long branch operators and algorithms, `delta_short_threshold=97`,
`delta_chunk_size=64`, `delta_transform_tile=32` and `delta_dot_precision=ieee`,
plus per-entry conditional graph counts, token ranges, source
checksums, toolchain and GPU architecture. Keep the package pair and metadata
together. The runtime rejects incompatible pairs.

[Torch 2.7's package loader](https://github.com/pytorch/pytorch/blob/v2.7.0/torch/csrc/inductor/aoti_package/model_package_loader.cpp#L42)
extracts directly into `/tmp`, ignoring `TMPDIR`.
Leave room there for both model libraries: about 3 GB plus margin for this pair.

### Optional compiler optimizations

Pass `--enable-inductor-optimization` to enable the current Inductor optimization
settings. This generic flag currently sets `combo_kernels=True` and
`benchmark_combo_kernel=True`; future optimizations can extend the same flag.
Export both entry points with matching settings into a fresh directory:

```bash
"$QWEN_PY" -m compiler.qwen35.export prefill --enable-inductor-optimization \
  --output artifacts/qwen35/0.8b/text-optimized
"$QWEN_PY" -m compiler.qwen35.export decode --enable-inductor-optimization \
  --output artifacts/qwen35/0.8b/text-optimized
```

Metadata records the `enable_inductor_optimization` flag and resolved
`inductor_configs`; the runtime rejects package pairs with different configs.
The flag is off by default and
requires no runtime switch. Rerun the model, runtime and end-to-end validators with
`--packages artifacts/qwen35/0.8b/text-optimized`, since these optimizations can
change floating-point results.

## Generate and validate

```bash
"$QWEN_PY" -m compiler.qwen35.prepare encode \
  --prompt "What is 17 + 25?" --output work/prompt.ids
work/qwen35/build/qwen35_aoti artifacts/qwen35/0.8b/text \
  work/prompt.ids work/generated.ids
"$QWEN_PY" -m compiler.qwen35.prepare decode --input work/generated.ids

"$QWEN_PY" -m compiler.qwen35.tests.validate_fused \
  --output artifacts/qwen35/0.8b/fused-validation-new
"$QWEN_PY" -m compiler.qwen35.tests.validate_model \
  --output artifacts/qwen35/0.8b/model-new
"$QWEN_PY" -m compiler.qwen35.tests.validate_runtime \
  --output artifacts/qwen35/0.8b/runtime-new
"$QWEN_PY" -m compiler.qwen35.tests.validate_end_to_end \
  --output artifacts/qwen35/0.8b/end-to-end-new
```

The focused fused validator compares output and both carried states with
HF convolution, gates, recurrence and chunked prefill. It reports `fusion_passed` separately
from full FP32 numerical acceptance, retains every comparison failure and
cross-checks the HF path against the same FP32 control. Python JIT, Python AOTI
and standalone C++ must agree bitwise. Inspect complete reports before treating
a nonzero exit as an expected numerical gate.

Chunking reorders arithmetic across sequence partitions. Core output and
recurrent-state partition comparisons use the existing BF16/FP32 numerical
limits, while convolution history remains exact. Repeated requests and
same-input JIT/AOTI/C++ comparisons retain their bitwise requirements.

Use `--packages` and `--binary`/`--native`/`--cli` to select other artifact and
build directories. Validation output directories must be fresh.
The focused fused suite checks preparation, core output and both carried states
through the `qwen35_fused` native target. Its 83 cases / 400 calls cover BF16/FP32, T=1..512
boundaries, mixed prefill/decode continuations, whole versus partitioned inputs
and repeated requests through four fused AOTI packages, including chunk-boundary
and prefill gate-extreme cases. These describe test coverage, not acceptance.
The native processes verify that neither Python nor custom-op libraries are
mapped. Full-model checks retain all state tensors, HF/FP32 controls, dispatch
counts, masked-future checks and CLI rejection cases. Numerical validators exit
nonzero for acceptance failures and retain the reports.

Recorded reports from the initial unified implementation are under
`artifacts/qwen35/0.8b/unified-fused-{validation,model,runtime}/summary.json`;
the full-model packages are in `artifacts/qwen35/0.8b/text-unified-fused`.
Those reports predate removal of the standalone recurrent control. Historical
cleanup revalidation is recorded in
`artifacts/qwen35/0.8b/remove-recurrent-validation/summary.json`: all 396 native
calls / 1,188 tensors match Python AOTI bitwise; fusion checks pass. The fused
and HF paths each retain 125 FP32 accuracy failures, so overall acceptance is open.
All of these reports precede the chunked prefill PoC. Re-export and run the
current validators into fresh output directories for its validation.
The [experiment record](../../docs/experiments/qwen35.md#unified-deltanet-preparation-and-recurrence-2026-10-29)
records the code-generation workaround, numerical limits and temporary raw
fixture location.

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
