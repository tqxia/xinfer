# Profiling xinfer on GB10

Use `nsys` for native execution timelines and `ncu` for kernel-level analysis.
Reuse an existing capture only when package, native binary, GPU, environment,
and workload all match.

## Prepare the workload

Reuse `workload.json` from an existing `tools.benchmark` result so prompt and
forced decode tokens match. For focused captures, copy it to a scratch file with
one relevant case and fewer repeats (keep positive warmup and repeat counts);
never overwrite benchmark evidence.

Set absolute paths on the GB10 host for `NATIVE` (`xinfer_benchmark`),
`PACKAGES`, `WORKLOAD`, `CAPTURE`, `NCU_CAPTURE`, and `RESULT`, using the same
`GPU_UUID` as the benchmark. Check `nsys --version` / `ncu --version` and
`--help` before selecting flags. If profiling alters library resolution, use a
documented local command workaround rather than modifying repo environments.

## Native timeline with NSYS

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$GPU_UUID" \
  nsys profile --trace=cuda --sample=none --output="$CAPTURE" \
  "$NATIVE" "$PACKAGES" "$WORKLOAD" "$RESULT"

nsys stats --report cuda_gpu_kern_sum,cuda_api_sum "$CAPTURE.nsys-rep"
nsys export --type sqlite --output "$CAPTURE.sqlite" "$CAPTURE.nsys-rep"
```

`xinfer_benchmark` does not emit NVTX phase ranges, so whole-process totals mix
loading, warmup, prefill, and decode:

- Inspect the sqlite schema, then use CUDA API/kernel correlations and repeated
  launch patterns to isolate steady-state decode steps (exclude load, warmup,
  and prefill; verify step counts against `workload.json`). Do not rely on a
  guesswork time cutoff alone — compare captures with different decode counts if
  phase boundaries are ambiguous.
- Compute kernel counts and GPU time per decode token; inspect CPU launch gaps,
  copies, and synchronization separately. Because kernels can overlap, summing
  kernel durations is not wall time.
- If the dominant bottleneck is runtime/host orchestration (e.g., launch
  overhead needing CUDA Graphs), record the finding and note it as out-of-scope
  runtime work.

## Kernel analysis with NCU

Use `ncu` to diagnose why a hotspot kernel is slow (roofline balance, memory
throughput, occupancy, register pressure, stalls). Derive `KERNEL_REGEX`,
`SKIP`, and `COUNT` from the variant's launch sequence (`--launch-skip` counts
only launches matching the regex; rewrites can change both names and counts):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$GPU_UUID" \
  ncu --kernel-name-base demangled --kernel-name "regex:$KERNEL_REGEX" \
  --launch-skip "$SKIP" --launch-count "$COUNT" \
  --section SpeedOfLight --section LaunchStats \
  --section Occupancy --section MemoryWorkloadAnalysis \
  --export "$NCU_CAPTURE" "$NATIVE" "$PACKAGES" "$WORKLOAD" "$RESULT"

ncu --import "$NCU_CAPTURE.ncu-rep" --page raw --csv > "$NCU_CAPTURE.csv"
```

- Check `--list-sections` and `--query-metrics` for host-supported metrics.
- Verify captured kernel names, grid/block dimensions, and invocation counts
  against the intended call sites; a matching name alone does not identify a
  shape.
- NCU replays launches and alters cache/clock state, so its timings never
  establish end-to-end latency. Record replay/cache/clock settings consistently.
- **If NCU is unavailable or permission-blocked:** record the exact error in the
  lab notebook, fall back to `nsys` + roofline math (bytes moved / GPU time) +
  generated Triton/PTX/SASS inspection, and note the missing counter evidence
  explicitly when making kernel claims.

## Inspecting compiled packages

Always confirm that an intended pass or kernel rewrite reached the measured
artifact and did not silently alter unrelated kernels:

- Check export logs (`decoder_projection_gemv` `replaced > 0`, `Packed ...`
  counts, `compile_options`, `compiled`) and staged `*.graph.txt` /
  `*.packing.json` files.
- Inspect the `.pt2` zip archive (`unzip -l decode.pt2`) and its `*.wrapper.cpp`
  to verify generated kernel calls, launch configs, and C-shim linkage.
- When a graph rewrite changes kernel names, Inductor may miss retained tuning
  cache entries for *unchanged* ops. Copying a tuning cache does not guarantee
  isolation — compare unaffected kernel bodies/launch configs against baseline
  before attributing a speedup or regression to your edit.

## Baseline calibration (kernel inner loop)

Before ranking standalone kernel revisions, compare unprofiled local baseline
GPU time against the corresponding operation or fused region in native `nsys`:

- Match shapes, strides, dtypes, actual implementation, launch config, and
  timing scope.
- If timings diverge enough to distort candidate ranking, check cache residency,
  surrounding traffic, warmup, clocks, and harness overhead. Adjust the fixture
  or document the gap before trusting local speedups.

## Measurement disagreements

When local timing, NCU counters, `nsys`, and paired native timing disagree in a
way that affects the decision: first check scope, cache state, and actual
generated code. If still unresolved, run one time-boxed follow-up check; if
uncertainty remains, document it and trust controlled unprofiled paired native
timing for model latency. Improved counters alone do not prove lower latency.
