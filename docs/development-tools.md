# Benchmark comparisons and numerical diagnostics

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
