# Benchmark comparisons, numerical diagnostics and profiling

Run these commands from the repository root with the matching Python/LibTorch
environment and previously exported packages. Use fresh output directories.

## Compare two compiled profiles

`compiler/benchmark.py compare` runs two package directories through the same
native benchmark executable. Each directory contains prefill and decode packages.
Both profiles receive identical prompts and forced decode tokens. Four default
process rounds alternate baseline/candidate order, with two warmups and five
measured requests per prompt length in each process. Loading and first use are
reported separately from steady-state latency.

```bash
.venv/bin/python compiler/benchmark.py compare \
  --baseline-packages artifacts/aoti-baseline \
  --candidate-packages artifacts/aoti-candidate \
  --baseline-label baseline --candidate-label candidate \
  --native build/qwen_benchmark \
  --output artifacts/comparison-new
```

The packages must have matching checkpoint, toolchain, tensor contracts and
numerical compilation settings. Both may use eager numerics; labels do not choose
the numerical profile. A changed attention backend is allowed. The existing
`run` command retains the eager / eager-numerics AOTI / fused AOTI comparison.
`--reference` selects the directory containing the saved capital prompt and
continuation; it defaults to `artifacts/reference`.

Results include per-request timings, latency distributions, predicted-token
agreement, memory samples, GPU telemetry, source snapshots and package/binary
hashes. Prediction agreement is diagnostic; use the correctness validator for
numerical acceptance. GPU-idle checks run before and after each worker. Use
`--allow-busy` only for harness checks; its timings are marked provisional.

## Collect numerical failures

```bash
.venv/bin/python compiler/validate_cpp.py \
  --packages artifacts/aoti-candidate \
  --collect-logit-errors --collect-top1-errors \
  --output artifacts/validation-candidate-new
```

`--collect-logit-errors` records logit comparisons outside the existing 5%
relative-L2 or peak-normalized limits. `--collect-top1-errors` records differing
token IDs, top-two margins and the reference score gap to the compiled-selected
token. These options allow those comparisons to continue, then save a failed
`summary.json` and exit nonzero. Without the flags, they fail immediately.

C++/Python AOTI bitwise checks, finite-value checks, cache invariants and generation
fixture assertions remain strict. Full-cache error collection and FP32 controls
are retained. The options do not change model arithmetic or acceptance limits.

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
