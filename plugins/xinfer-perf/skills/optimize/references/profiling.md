# Investigate decode with existing profilers

Read this for the initial native baseline profile. Within each experiment
iteration, profile the candidate after validation and unprofiled paired
benchmarking. Use `nsys` for native execution and targeted `ncu` captures for
kernel analysis. NCU evidence is required for baseline and candidate kernel
analysis, including the [kernel inner loop](techniques/kernel-optimization.md#kernel-inner-loop).
Run the tools directly on the GPU host. Check their installed `--help` and
versions before choosing capture/filter options; supported metrics depend on
the tool and device.

Reuse an existing capture when its packages/native implementation, device,
software environment and relevant workload match the variant being analyzed.
Record those identities and the capture path in the experiment notes. A baseline
capture supplies the before evidence; analyze a matching candidate capture to
explain the change and identify the next target. If matching profiling evidence
is missing, obtain it or report the gap before choosing the next candidate.

## Start with the native workload

Reuse `workload.json` from an existing xinfer benchmark result. Its cases contain
the actual prompt and forced decode tokens. For a focused capture, write a new
scratch copy with one relevant case and fewer repetitions; keep positive warmup
and repeat counts. Record those changes. Do not overwrite the benchmark evidence.

Set `NATIVE`, `PACKAGES`, `WORKLOAD`, `CAPTURE` and `RESULT` to absolute paths on
the execution host; `NATIVE` is the existing `xinfer_benchmark`. Use a fresh
capture prefix and result filename. Select the same `GPU_UUID` as the benchmark.
A starting capture, subject to the installed CLI help, is:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$GPU_UUID" \
  nsys profile --trace=cuda --sample=none --output="$CAPTURE" \
  "$NATIVE" "$PACKAGES" "$WORKLOAD" "$RESULT"
```

This profiles the native executable directly, including initialization, model
loading, first use, warmups, prefill and decode. The current executable does not
provide NVTX phase labels. Whole-process kernel totals are therefore not decode
totals. Locate repeated one-token calls using the workload, timeline and native
call sequence; describe uncertain phase attribution as uncertain. If necessary,
compare focused captures with different decode counts while holding the prompt
fixed. Do not add a runtime driver or claim that a time-window cutoff proves a
phase boundary. Nsight Systems supports capture and post-collection analysis
without application changes. [NVIDIA guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

## Ask a concrete question

- Many small launches: which operations repeat per decode step, and where are
  CPU submission gaps versus device execution?
- A dominant kernel: is its time sensitive to context, shape, layout or launch
  configuration? What fraction of decode could changing it actually affect?
- Copies or synchronization: which compiler-generated operation causes them?
  If the cause is runtime behavior, record it for the future runtime skill.

Use `nsys stats`, its available report list, and exports before reading raw data.
Start with kernel/CUDA API summaries, then inspect a bounded time range or query
the exported database. Kernel durations can overlap; their sum is not wall time.

For kernel analysis, collect matching baseline and candidate NCU evidence. Use
kernel/launch filters and select the smallest useful sections or metrics.
Discover them with `--list-sections` and `--query-metrics`. Check that the selected
invocation matches the requested phase and shape rather than setup or warmup.
Nsight Compute can replay launches to collect counters, so its timings and cache
conditions do not establish end-to-end latency. Record
replay/cache settings and keep comparisons consistent with the question; do not
assume NCU's cache state matches native execution.
[NVIDIA CLI guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)

## Connect the evidence to an experiment

Follow a kernel name into the corresponding export graph, generated wrapper,
Triton/CUDA source or PTX/SASS, using the actual package/build artifacts. Shapes,
dispatch and call sites help disambiguate repeated names. Fusion, combo kernels
and external calls can make a one-kernel/one-node mapping impossible; retain that
ambiguity.

For initial target selection, cite the baseline observation, its relevance to
the requested phase and the mechanism a change could improve. Inspect the
associated source to choose a compiler option, graph change, adapter or kernel
experiment. Investigate uncertain phase attribution before selecting a candidate.

Within each completed iteration, confirm the candidate changed the intended
generated code or dispatch. Compare its profile with the baseline, hypothesis
and unprofiled paired measurements to explain the result and choose the next
target. Repeat benchmarking when noise or new evidence warrants it. A Python
probe or microbenchmark can explain a mechanism but does not replace the
deployed path.

Keep profiler commands, versions and result paths in the experiment notes. If
NCU collection is unavailable or permission-blocked, retain the error and defer
kernel analysis and further kernel hypotheses until collection is available.
Preparation and correctness work can continue; timing or `nsys` summaries do not
replace the required NCU evidence.

If profiling changes library resolution, inspect the actual environment and
dependencies; apply only a local, documented workaround rather than changing the
project environment or Dockerfile as part of the optimization.
