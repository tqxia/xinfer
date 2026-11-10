# Investigate decode with existing profilers

Read this when measurements need an explanation. Run `nsys` and `ncu` directly
on the GPU host. Check their installed `--help` and versions before choosing
capture/filter options; supported metrics depend on the tool and device.

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

For a specific kernel question, use `ncu` kernel/launch filters and select the
smallest useful sections or metrics. Discover them with `--list-sections` and
`--query-metrics`. Check that the selected invocation belongs to decode rather
than setup or prefill. Nsight Compute can replay launches to collect counters,
so its timings and cache conditions do not establish end-to-end latency.
[NVIDIA CLI guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)

## Connect the evidence to an experiment

Follow a kernel name into the corresponding export graph, generated wrapper or
Triton source, using the actual package/build artifacts. Shapes, dispatch and
call sites help disambiguate repeated names. Fusion, combo kernels and external
calls can make a one-kernel/one-node mapping impossible; retain that ambiguity.

Inspect enough source to propose a compiler option, graph change, adapter or
kernel experiment. Confirm the candidate changes the intended generated code
or dispatch, then return to unprofiled native measurements. A Python probe or
microbenchmark can explain a mechanism but does not replace the deployed path.

Keep profiler commands, versions and result paths in the experiment notes. If
counters are unavailable, report the limitation and use the evidence available.
If profiling changes library resolution, inspect the actual environment and
dependencies; apply only a local, documented workaround rather than changing the
project environment or Dockerfile as part of the optimization.
