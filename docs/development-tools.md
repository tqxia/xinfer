# Benchmark comparisons, numerical diagnostics and profiling

Run these commands from the repository root with the matching Python/LibTorch
environment and previously exported packages. Use fresh output directories.

## Compare HF eager with compiled Qwen3

`run` compares the unmodified HF eager reference with one FlashAttention AOTI
package pair. The compiled profile is named `aoti`; there is no separate
eager-numerics compilation profile. Each process receives the same prompts and
forced decode tokens, so differences in predicted tokens do not change the
measured workload.

```bash
cmake -S . -B build -DQWEN_BUILD_BENCHMARK=ON
cmake --build build --target qwen_benchmark -j2
.venv/bin/python -m compiler.qwen3.benchmark run \
  --packages artifacts/aoti \
  --native build/qwen_benchmark \
  --output artifacts/benchmark-new
```

The Qwen3 build always links `libxinfer_attention_ops.so`. Deploy that library
with the native executable and matching LibTorch. Previously exported efficient
packages must be re-exported using the current exporter.

## Compare two compiled profiles

`compiler/qwen3/benchmark.py compare` runs two package directories through the same
native benchmark executable. Each directory contains prefill and decode packages.
Both profiles receive identical prompts and forced decode tokens. Four default
process rounds alternate baseline/candidate order, with two warmups and five
measured requests per prompt length in each process. Loading and first use are
reported separately from steady-state latency.

```bash
.venv/bin/python -m compiler.qwen3.benchmark compare \
  --baseline-packages artifacts/aoti-baseline \
  --candidate-packages artifacts/aoti-candidate \
  --baseline-label baseline --candidate-label candidate \
  --native build/qwen_benchmark \
  --output artifacts/comparison-new
```

Both package pairs must use the fixed FlashAttention operator and extent-input
ABI, with matching checkpoint, toolchain, tensor contracts and numerical
compilation settings. Each pair's prefill/decode metadata is checked as well.
Labels name the measurements; they do not select another execution backend.
`--reference` selects the directory containing the saved capital prompt and
continuation; it defaults to `artifacts/reference`.

Results include per-request timings, latency distributions, predicted-token
agreement, memory samples, GPU telemetry, source snapshots and package/binary
hashes. Prediction agreement is diagnostic; use the correctness validator for
numerical acceptance. GPU-idle checks run before and after each worker. Use
`--allow-busy` only for harness checks; its timings are marked provisional.

## Validate runtime correctness and report HF differences

```bash
.venv/bin/python -m compiler.qwen3.tests.validate_cpp \
  --packages artifacts/aoti-candidate \
  --output artifacts/validation-candidate-new
```

The default validation report records HF cache errors and differing token IDs,
including top-two margins and the HF score gap to the compiled-selected token.
Those differences are accepted diagnostic results for the fused Flash profile;
the report distinguishes HF parity from runtime correctness. Logit differences
must still satisfy the existing 5% relative-L2 and peak-normalized limits.

C++/Python AOTI bitwise agreement, finite-value checks, cache writes and masking,
reset/reuse, stable storage, boundary rejection and native/Python generation
agreement remain strict. Independent HF eager and FP32 controls are retained.
This policy does not change model arithmetic or make HF numerical equivalence a
claim of the compiled profile. See [FlashAttention](flash-attention.md) for the
recorded differences and supported workload.

## Profile the native runtime

`qwen_profile` calls the same `Runtime` as the CLI. NVTX ranges label each case,
reset, prefill and decode step, with nested forward and argmax/host-read ranges.
Model loading and warmup finish before `cudaProfilerStart()`. Build it explicitly;
profiling is OFF by default and adds no NVTX instrumentation to the normal CLI.

```bash
# First configure the matching LibTorch/CUDA toolchain as described in README.md.
cmake -S . -B build -DQWEN_BUILD_PROFILE=ON
cmake --build build --target qwen_profile -j2
mkdir -p artifacts/profile-new
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=artifacts/profile-new/model \
  ./build/qwen_profile artifacts/aoti artifacts/comparison-new/workload.json \
  artifacts/profile-new/requests.json
nsys export --type=sqlite --output=artifacts/profile-new/model.sqlite \
  artifacts/profile-new/model.nsys-rep
.venv/bin/python tools/profiling/analyze_profile.py \
  artifacts/profile-new/model.sqlite artifacts/profile-new/analysis.json
```

The workload uses the benchmark's `warmup`, `repeats`, prompt and fixed decode
token fields. For a shorter trace, use a workload with fewer measured requests
and decode steps. Optional `expected_predictions` on a case enables comparison
against a previously measured baseline; otherwise predictions are only recorded.

The analyzer attributes GPU operations through CUDA launch correlation IDs inside
the NVTX ranges. It reports stage wall time, the union of GPU operation intervals,
intervals without recorded GPU work, kernel categories and CUDA API durations.
Host and GPU times overlap and must not be added. Kernel classification uses name
heuristics and assumes the final linear kernel is the LM head; review those
assumptions after toolchain/model changes. This is trace attribution, not a
hardware-counter measurement of memory bandwidth or SM occupancy. Use the
unprofiled benchmark for performance comparisons.
