# Investigate decode with existing profilers

Read this for the initial native baseline profile. Within each experiment
iteration, profile the candidate after validation and unprofiled paired
benchmarking. Use `nsys` for native execution and targeted `ncu` captures for
kernel analysis.

Reuse an existing capture when its packages/native implementation, device,
software environment and relevant workload match the variant being analyzed. A
baseline capture supplies the before evidence; analyze a matching candidate
capture to explain the change and identify the next target. If matching
profiling evidence is missing, obtain it or report the gap before choosing the
next candidate.

## Prepare the workload

Reuse `workload.json` from an existing xinfer benchmark result. Its cases
contain the actual prompt and forced decode tokens. For a focused capture, write
a new scratch copy with one relevant case and fewer repetitions; keep positive
warmup and repeat counts. Record those changes. Do not overwrite the benchmark
evidence.

Set `NATIVE`, `PACKAGES`, `WORKLOAD`, `CAPTURE`, `NCU_CAPTURE` and `RESULT` to
absolute paths on the execution host; `NATIVE` is the existing
`xinfer_benchmark`. Use a fresh capture prefix and result filename for each run.
Select the same `GPU_UUID` as the benchmark.

Run the tools directly on the GPU host. Check their installed `--help` and
versions before choosing capture/filter options; supported metrics depend on the
tool and device. If profiling changes library resolution, inspect the actual
environment and dependencies; apply only a local, documented workaround rather
than changing the project environment or Dockerfile as part of the optimization.

## Profile native execution with NSYS

A starting
[NSYS capture](https://docs.nvidia.com/nsight-systems/UserGuide/index.html):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$GPU_UUID" \
  nsys profile --trace=cuda --sample=none --output="$CAPTURE" \
  "$NATIVE" "$PACKAGES" "$WORKLOAD" "$RESULT"
```

This profiles the native executable directly, including initialization, model
loading, first use, warmups, prefill and decode. The current executable does not
provide NVTX phase labels. Whole-process kernel totals are therefore not decode
totals. Separate phases during analysis without adding a runtime driver.

Start with kernel/CUDA API summaries, then export for phase-specific analysis:

```bash
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum "$CAPTURE.nsys-rep"
nsys export --type sqlite --output "$CAPTURE.sqlite" "$CAPTURE.nsys-rep"
```

Inspect the exported schema before querying. Use API/kernel correlations, the
native call sequence and repeated launch patterns to identify requests and
decode steps. Check their counts against the workload, excluding prefill and
warmups; a time-window cutoff alone does not establish a phase boundary. If
needed, compare captures with different decode counts and the same prompt.

Record the selected request/context positions, steps and boundary evidence.
Compute kernel counts and GPU time per selected token; examine CPU submission
gaps, copies and synchronization separately. Kernel durations can overlap, so
their sum is not wall time. Retain uncertainty in phase attribution. Trace
copies and synchronization to their cause; runtime-only findings belong to the
future runtime skill.

## Baseline calibration

For the kernel inner loop, compare unprofiled local baseline GPU time with the
corresponding operation or complete region in the native NSYS capture. Match
shapes, strides, dtypes, actual implementations, launch configurations and timing
scope. For discrepancies that could change candidate ranking, check cache
residency, surrounding work, warmup, clocks and measurement overhead. Adjust the
fixture or explain the difference before using local gains to rank revisions;
identical absolute timings are not required. Record the comparison and its limits
with the baseline evidence, using the capture-reuse conditions above.

## Analyze selected kernels with NCU

NCU evidence is required for baseline and candidate kernel analysis, including
the [kernel inner loop](kernel-optimization-loop.md#kernel-inner-loop).
Follow selected launches into the corresponding export graph, generated wrapper,
Triton/CUDA source or PTX/SASS, using the actual package/build artifacts.
Shapes, dispatch and call sites help disambiguate repeated names. Fusion, combo
kernels and external calls can make a one-kernel/one-node mapping impossible;
retain that ambiguity.

Choose a few decode invocations to investigate sensitivity to context, shape,
layout or launch configuration, accounting for their share of native decode
cost. Derive `KERNEL_REGEX`, `SKIP` and `COUNT` from each variant's launch
sequence: `--launch-skip` counts only launches matching the kernel filter.
Rewrites can change both names and counts.

For a memory/occupancy question, a starting capture and CSV export are:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$GPU_UUID" \
  ncu --kernel-name-base demangled --kernel-name "regex:$KERNEL_REGEX" \
  --launch-skip "$SKIP" --launch-count "$COUNT" \
  --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis \
  --export "$NCU_CAPTURE" "$NATIVE" "$PACKAGES" "$WORKLOAD" "$RESULT"
ncu --import "$NCU_CAPTURE.ncu-rep" --page raw --csv > "$NCU_CAPTURE.csv"
```

Use `--list-sections` and `--query-metrics` to select supported data for the
question. Verify the captured names, grid/block sizes and invocation counts
against the intended call sites; a matching name alone does not identify a
shape. Nsight Compute can replay launches to collect counters, so its timings
and cache conditions do not establish end-to-end latency. Record replay, cache
and clock-control settings and keep comparisons consistent with the question; do
not assume NCU's cache state matches native execution.
[NVIDIA CLI guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)

If NCU collection is unavailable or permission-blocked, retain the error and
defer kernel analysis and further kernel hypotheses until collection is
available. Preparation and correctness work can continue; timing or `nsys`
summaries do not replace the required NCU evidence.

## Record the evidence

Add captures as evidence in the [investigation record](records/format.md),
naming the revision(s) and exact export where applicable. Link commands,
versions, artifact/workload identities, capture/results and notes on selected
invocations, observations and uncertainty. Use a context override for a modified
workload. For candidates, confirm the intended generated code or dispatch
changed. Return these findings to the
[experiment loop](experiment-loop.md#6-profile-and-analyze) for comparison with the
hypothesis and unprofiled paired measurements, feedback and next-target
selection. Resolve uncertain phase attribution before selecting a candidate;
local probes do not replace native measurement.
