# Qwen3.5 CUDA Graph decode on GB10 — 2026-10-31

This experiment is retired. At the user's request, all CUDA Graph implementation,
CLI, validation-tool and benchmark-integration changes were discarded; this
document is retained as the experiment record. The measured 2.7–3.1% decode gain
did not justify the added runtime complexity, setup cost and memory use.

The experiment reused the existing Qwen3.5-0.8B AOTI packages and passed all
replay validation. vLLM was stopped for the successful validation and benchmark.
The repository's runtime remains ordinary AOTI; the experimental `--cudagraph`
option is no longer available.

## Historical implementation (removed)

The experiment used `dev-gb10-20261030`, based on `2cf7246`, Torch
`2.13.0+cu130`, Triton `3.7.1`, CUDA 13.0 and driver 580.159.03. The exact
development sources and binary/package hashes are retained with each run;
the base Git commit alone does not reconstruct the discarded experimental code.

The experimental `CudaGraphRuntime` in `runtime/qwen35/cudagraph.h` borrowed a
loaded `Packages` pair and owned one private stream, fixed token and position
tensors, and 48 fixed state tensors. Separate token/position allocations preserved
the package's 16-byte alignment requirement. The package loaders used
`run_single_threaded=true` and required serialized use.

Setup performed three decode warmups, then captured the existing `decode.run`
and 48 copies from returned functional states into the persistent state bank.
It cleared the buffers after setup. Each user decode filled the token and
position tensors and replayed the graph. The position changed as GPU tensor
data; shapes and addresses stayed fixed. Request reset cleared existing buffers
without reallocating them. A T=1 prompt also used replay. Prefill T=2..512
remained uncaptured and copied its returned state into the same bank.

Caller/private stream events ordered updates and consumption without a host
wait in the decode replay path. Argmax, the host token read, EOS handling and tracing
remained outside capture. Graph logits/state accessors returned borrowed buffers
that subsequent calls overwrote; retained results required explicit snapshots.
The graph and its buffers could not outlive the borrowed packages.

The functional `Runtime` retained its original state replacement behavior and
default loader mode. Validation shared one package pair between the functional
and graph runtimes to avoid loading a second set of weights. Neither the export
graph, kernels, package ABI nor HF numerical thresholds changed.

The state bank contained 32,342,016 bytes (30.84375 MiB): 18 convolution tensors,
18 FP32 recurrent tensors, and six K/V pairs. Copy-back added approximately
61.69 MiB of reads/writes per decoded token. These copies were inside the graph.

## Validation

The experimental native validator compared raw tensor bytes, including signed
zero, after every call. It checked all logits and states, and separately verified
that the functional reference's input tensors remained unchanged. Graph snapshots were
explicit; the existing input-immutability checks were not relaxed.

The successful `validation-r3/summary.json` records:

- 1,214 calls: 1,185 decode replays and 29 prefill calls.
- 1,214 bitwise logit comparisons and 58,272 bitwise state comparisons.
- 58,272 functional-input immutability checks.
- 29 resets, stable token/position/state addresses, and exactly one capture.
- All eight benchmark workloads, each with its complete 64-token continuation.
- T=1/2/63/96/99/127/129 boundaries, greedy decoding, repeated requests,
  poisoned future KV, mixed prefill/decode and 512 decode steps to capacity 1,024.
- Six rejected-input cases, preserving state, positions, buffers and counters.
- No Python libraries linked or mapped in the native worker.

The Python wrapper verified actual package hashes, archived source snapshots,
and checked that sources, packages, metadata, executable and workload remained
unchanged during validation. A graph benchmark required a successful report
for the matching packages and runtime headers.

The experimental user-facing CLI also passed: ordinary and captured generation produced
identical 32-token outputs and byte-identical 31,784,960-byte FP32 logit traces.
The request stopped at its token limit. Zero generation budget, duplicate
`--cudagraph` and a missing option value were rejected before creating output.

This established equivalence to the AOTI implementation used in this experiment.
Its HF numerical acceptance remained open and was recorded separately.

## Performance

The same packages and fixed teacher-fed token workloads were used in four
fresh, sequential processes: ordinary→graph, then graph→ordinary. Each worker
ran eight prompt lengths, 64 decode tokens, one cold request, two warmups and
five timed repetitions per length. The table pools ten timed samples per
profile/length. Both paths include token preparation, argmax and host token
reads; EOS does not shorten the workload.

| Prompt tokens | Ordinary ms/token | Graph ms/token | Ordinary tok/s | Graph tok/s | Decode speedup |
|---:|---:|---:|---:|---:|---:|
| 22 | 13.7383 | 13.3470 | 72.79 | 74.92 | 1.029× |
| 64 | 13.7115 | 13.3327 | 72.93 | 75.00 | 1.028× |
| 65 | 13.7265 | 13.3164 | 72.85 | 75.10 | 1.031× |
| 97 | 13.7111 | 13.3368 | 72.93 | 74.98 | 1.028× |
| 98 | 13.7290 | 13.3380 | 72.84 | 74.97 | 1.029× |
| 128 | 13.7284 | 13.3602 | 72.84 | 74.85 | 1.028× |
| 511 | 13.7103 | 13.3453 | 72.94 | 74.93 | 1.027× |
| 512 | 13.6976 | 13.3344 | 73.01 | 74.99 | 1.027× |

Decode saves 0.363–0.410 ms/token. Complete-request speedup is 1.023–1.029×, saving
22.70–25.56 ms. Short prefill T=22..128 added approximately 0.66–0.84 ms; it
remained uncaptured and transferred its state into the persistent bank. The
small mixed changes at T=511/512 are not evidence of prefill acceleration.

All 520 distinct benchmark prediction positions match across profiles. Each
graph worker records exactly one capture and 4,096 replays, including cold,
warmup and timed requests. Post-run checks verify unchanged sources, executable
and packages against the archived environment.

### Startup and memory

Graph setup takes 299.42 and 313.35 ms in the two workers. This includes three
warmups taking 272.65 and 285.89 ms; capture itself takes 9.72 and 9.82 ms.
Native `runtime_setup_ms` additionally includes constructor overhead and is
304.78/319.05 ms, versus 16.41/18.01 ms for ordinary AOTI. `load_ms` includes
package loading and runtime setup; do not add setup to it again. Package loads
vary around 15–17 seconds and are excluded from steady-state measurements.

The maximum timed caching-allocator peak rises from 169.40 to 200.72 MiB
(+31.32 MiB), and reserved memory from 182 to 236 MiB (+54 MiB). These counters
exclude some AOTI constant storage and are not total model-memory footprints.

### Interpretation and measurement conditions

This compares the complete captured runtime with ordinary native AOTI. It
includes persistent buffers, the single-threaded loader, stream events and
state copies, so it does not isolate `cudaGraphLaunch` overhead alone.

Fifteen of sixteen individual round/length comparisons improve. Round 1,
T=22 is 0.994×: two early baseline samples are unusually fast at about 11 ms/token.
Initial busy GPU telemetry shows 916–923 MHz before 611 MHz. The observations
are consistent with a clock transition, but request samples lack timestamps
that would prove causality. Almost all other busy samples, and every busy
graph sample, show 611 MHz. Clocks were not locked or changed.

All eight idle boundary checks pass without `--allow-busy`. During the benchmark,
the companion monitor records no surviving vLLM process, a minimum of about
113.96 GiB available host memory, temperatures of 38–43°C, and no sampled active clock-event
flags. There are 86 swap-in pages and no swap-out pages; memory-pressure stall
time increases by only 2 microseconds over the monitored benchmark window.

The first two validation attempts failed with CUDA OOM while loading the AOTI
package containers, before graph capture. vLLM still reserved 105,878 MiB then.
An attempt to release clean cache for this task's own package/scratch files did
not resolve loading. The user stopped vLLM; the subsequent full validation and
all performance workers completed normally. Those failed runs are retained and
are excluded from performance results.

The measured gain is modest. The discarded implementation copied the entire
functional state and performed a host token read each step. Persistent in-place
state updates and device-side token handling remain possible future experiments;
this run does not quantify their potential gains.

## Historical reproduction commands

These commands describe the removed experiment and do not run against the
current repository as-is. To revisit the benchmark or validator, first restore
the experimental sources from both retained artifact directories:

- `artifacts/gb10-20261031/qwen35/cudagraph/validation-r3/source/`
- `artifacts/gb10-20261031/qwen35/cudagraph/benchmark/source/`

Those snapshots contain the experimental runtime, native validator, benchmark,
Python wrappers and build definitions. Restore them in a separate checkout based
on `2cf7246`, then configure its build using the environment and paths in
[the GB10 guide](../gb10.md) and choose fresh output directories. The CLI changes
to `runtime/qwen35/main.cpp` were not archived, so the commands below reproduce
only the validation and benchmark paths:

```bash
cmake --build "$GB_BUILD" --target \
  qwen35_benchmark qwen35_cudagraph_validate -j2

"$Q35_PY" -m compiler.qwen35.tests.validate_cudagraph \
  --packages "$Q35_PACKAGES" --model-dir "$Q35_MODEL" \
  --binary "$GB_BUILD/qwen35_cudagraph_validate" \
  --output artifacts/gb10-cudagraph-validation-new

"$Q35_PY" -m compiler.qwen35.benchmark run --gpu 0 \
  --packages "$Q35_PACKAGES" --model-dir "$Q35_MODEL" \
  --native "$GB_BUILD/qwen35_benchmark" \
  --profiles cpp_aoti cpp_aoti_cudagraph \
  --cudagraph-validation artifacts/gb10-cudagraph-validation-new/summary.json \
  --validation artifacts/gb10-20261030/qwen35/end-to-end-v2-sequential-r2/summary.json \
  --lengths 22 64 65 97 98 128 511 512 \
  --decode-steps 64 --warmup 2 --repeats 5 --rounds 2 \
  --output artifacts/gb10-cudagraph-benchmark-new
```

## Retained evidence

All paths below are relative to `/home/tqxia/workspace/xinfer`:

- `artifacts/gb10-20261031/qwen35/cudagraph/validation-r3/`: successful native
  validation, source snapshots, attestation, logs and telemetry.
- `artifacts/gb10-20261031/qwen35/cudagraph/benchmark/`: raw workers, workload,
  package metadata, source snapshots, summary, independent `analysis.json`,
  `resource-analysis.json` and `post-run-verification.json`.
- `artifacts/gb10-20261031/qwen35/cudagraph/cli-smoke/`: token outputs, raw
  logit traces, parser rejections and summary.
- `artifacts/gb10-20261031/qwen35/cudagraph-resources/`: raw resource monitoring.
- `artifacts/gb10-20261031/qwen35/cudagraph/{validation,validation-r2}/`:
  the two pre-capture OOM attempts.

The retained source snapshots, build helpers and generated evidence remain
ignored historical artifacts under `work/` and `artifacts/`. They are not active
repository implementation. No compiled packages or model weights changed.
