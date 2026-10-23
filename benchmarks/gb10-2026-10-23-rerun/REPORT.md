# GB10 full eager / AOTI benchmark rerun

Measured on 2026-10-23 (Asia/Shanghai) in `/home/tqxia/workspace/xinfer`.
Source commit: `01f16c696b505c47da50604574d4373bda40016c` on `dev-gb10`.

## Comparison with the earlier 611 MHz run

The GPU now boosts in the preflight compute diagnostic: BF16 median 2,288 MHz and 97.57 TFLOP/s; FP16 median 2,197 MHz and 95.77 TFLOP/s. The previous diagnostic stayed at 611 MHz and about 28.44 TFLOP/s. No clock settings or power limits were changed for this rerun.

This is a controlled workload comparison with the [earlier full run](../gb10-2026-10-23/REPORT.md): benchmark source files, native executable, model manifest, compiled packages and exact workload hashes all match. The Git commit differs only because the earlier report was committed. CPU threads and idle-check policy also match. Operating conditions and filesystem cache were not fixed, so this is not a frequency-only experiment.

| Prompt tokens | Profile | Previous decode tokens/s | Rerun decode tokens/s | Gain | Prefill gain |
| ---: | --- | ---: | ---: | ---: | ---: |
| 25 | Python eager | 34.2 | 69.5 | 2.03x | 2.05x |
| 25 | AOTI eager numerics | 50.7 | 92.0 | 1.81x | 1.86x |
| 25 | AOTI fused | 55.2 | 98.0 | 1.78x | 1.82x |
| 128 | Python eager | 34.0 | 69.5 | 2.04x | 2.24x |
| 128 | AOTI eager numerics | 50.7 | 91.8 | 1.81x | 2.15x |
| 128 | AOTI fused | 55.1 | 97.7 | 1.77x | 2.15x |
| 512 | Python eager | 34.1 | 69.3 | 2.03x | 2.72x |
| 512 | AOTI eager numerics | 50.7 | 91.9 | 1.81x | 2.82x |
| 512 | AOTI fused | 55.2 | 97.6 | 1.77x | 2.88x |

All benchmark predictions match the prior run. Each profile and length is deterministic across all 15 measured requests. Both AOTI profiles match eager on all 65 predictions per request. The fused profile still carries its previously established full-cache numerical limitation; matching these tokens does not remove that limitation.

## Method

- Qwen3-0.6B, BF16, batch 1, static 1,024-position KV cache, PyTorch 2.13.0+cu130, Triton 3.7.1, Transformers 4.51.3.
- NVIDIA GB10 (SM121), ARM64, driver 580.159.03, CUDA Toolkit 13.0. Default clock/power settings were retained.
- Prompt lengths 25 / 128 / 512, each followed by 64 teacher-forced decode steps. EOS tokens are ordinary fed inputs in this benchmark.
- Three fresh-process rounds with rotating profile order. Each length has one separately recorded first request, two warmups, then five measured requests per round.
- 15 measured samples per profile/length; 135 measured requests total. CPU threads are fixed to one; CPU affinity is recorded, not pinned.
- The GPU passed five consecutive utilization samples at or below 10% before and after each worker. No `--allow-busy` override was used.
- Timings synchronize CUDA at boundaries and include tensor allocation, argmax and host token synchronization. Prefill excludes cache reset; request time includes reset, prefill and 64 decode steps. Loading and tokenization are excluded from warmed timings.
- Both AOTI profiles run in standalone C++; eager runs the same tensor-cache adapter in Python. The comparison includes both compilation and Python-versus-C++ execution effects.

## Warmed performance

Values are medians over the 15 measured requests. Decode rate is 1,000 divided by median milliseconds per decode step.

| Prompt tokens | Profile | Prefill ms | Decode ms/token | Decode tokens/s | Request ms |
| ---: | --- | ---: | ---: | ---: | ---: |
| 25 | Python eager | 15.11 | 14.39 | 69.5 | 936.77 |
| 25 | AOTI eager numerics | 11.07 | 10.87 | 92.0 | 707.41 |
| 25 | AOTI fused | 10.29 | 10.20 | 98.0 | 664.04 |
| 128 | Python eager | 16.52 | 14.40 | 69.5 | 938.99 |
| 128 | AOTI eager numerics | 12.17 | 10.89 | 91.8 | 710.05 |
| 128 | AOTI fused | 11.09 | 10.23 | 97.7 | 666.83 |
| 512 | Python eager | 26.80 | 14.44 | 69.3 | 951.56 |
| 512 | AOTI eager numerics | 19.49 | 10.89 | 91.9 | 716.95 |
| 512 | AOTI fused | 17.38 | 10.25 | 97.6 | 674.16 |

Speedups use eager latency divided by AOTI latency.

| Prompt tokens | Eager-numerics prefill | Eager-numerics decode | Fused prefill | Fused decode |
| ---: | ---: | ---: | ---: | ---: |
| 25 | 1.37x | 1.32x | 1.47x | 1.41x |
| 128 | 1.36x | 1.32x | 1.49x | 1.41x |
| 512 | 1.38x | 1.33x | 1.54x | 1.41x |

## Variability and process loading

The p10–p90 intervals below describe the observed samples, not confidence intervals.

| Prompt tokens | Profile | Prefill p10–p90 ms | Decode p10–p90 ms/token |
| ---: | --- | ---: | ---: |
| 25 | Python eager | 15.05–15.17 | 14.37–14.44 |
| 25 | AOTI eager numerics | 11.04–11.11 | 10.86–10.89 |
| 25 | AOTI fused | 10.26–10.31 | 10.18–10.25 |
| 128 | Python eager | 16.49–16.91 | 14.36–14.46 |
| 128 | AOTI eager numerics | 12.09–12.21 | 10.88–10.92 |
| 128 | AOTI fused | 11.02–11.10 | 10.20–10.28 |
| 512 | Python eager | 26.69–27.40 | 14.41–14.51 |
| 512 | AOTI eager numerics | 19.39–19.69 | 10.87–10.92 |
| 512 | AOTI fused | 17.37–17.40 | 10.24–10.27 |

| Profile | Median fresh-process model/package load | Min–max load |
| --- | ---: | ---: |
| Python eager | 7.84 s | 6.97–8.96 s |
| AOTI eager numerics | 10.79 s | 9.62–10.94 s |
| AOTI fused | 10.55 s | 10.05–11.82 s |

Load time excludes Python import and CUDA initialization. The OS file cache was not cleared, so these are fresh-process loads, not controlled cold-disk loads. `rounds.json` records each process and its first requests.

## Clock and power context

Automatic clock control was retained. These are telemetry samples with at least 50% GPU utilization, spanning warmup and measured requests rather than individually aligned measurement windows.

| Profile | Active-sample median SM clock | Active-sample SM clock range | Median reported GPU power |
| --- | ---: | ---: | ---: |
| Python eager | 2515 MHz | 2470–2515 MHz | 28.74 W |
| AOTI eager numerics | 2515 MHz | 2424–2515 MHz | 29.96 W |
| AOTI fused | 2509 MHz | 2431–2515 MHz | 30.83 W |

The GPU reports a maximum SM clock of 3,003 MHz. A timestamped during-run snapshot is saved in `power-clock-snapshot.txt`. Automatic clock control was retained; this is not a locked-clock peak-throughput benchmark. Reported GPU power is not total system/module power. Active samples include model loading, warmups and measured requests; they are not aligned to individual timing windows.

## Memory

Maximum across the three rounds. MiB = 1,048,576 bytes.

| Prompt tokens | Profile | Allocator peak allocated MiB | Allocator peak reserved MiB | Sampled device-use delta MiB |
| ---: | --- | ---: | ---: | ---: |
| 25 | Python eager | 1289.5 | 1292.0 | 3762.8 |
| 25 | AOTI eager numerics | 152.5 | 154.0 | 5128.2 |
| 25 | AOTI fused | 152.5 | 154.0 | 5126.4 |
| 128 | Python eager | 1291.8 | 1296.0 | 3767.2 |
| 128 | AOTI eager numerics | 154.8 | 158.0 | 5132.2 |
| 128 | AOTI fused | 154.8 | 158.0 | 5124.3 |
| 512 | Python eager | 1300.7 | 1322.0 | 3794.5 |
| 512 | AOTI eager numerics | 168.0 | 182.0 | 5154.7 |
| 512 | AOTI fused | 167.0 | 182.0 | 5148.9 |

AOTI weights can bypass the PyTorch allocator, so allocator peaks alone are not total model memory. Device-use deltas come from sampled `cudaMemGetInfo` values relative to each process before model loading; they are global and use GB10 unified memory. They are not exact per-process or transient peaks. Both AOTI loaders keep independent weight copies. NVML memory-used telemetry is unavailable on this GB10; supported utilization, power and clock telemetry is retained in the raw run directory.

## Numerical correctness

The full numerical validation below is carried forward from the prior run on identical package hashes; it was not repeated for this timing rerun. Prediction agreement below is checked afresh by the benchmark.

| Profile | Three-path acceptance | Logit top-1 differences / 163 | Full-cache tensors outside 5% limits | Full-cache max relative L2 | Full-cache max peak-relative error |
| --- | --- | ---: | ---: | ---: | ---: |
| AOTI eager numerics | PASS | 0 | 0/56 | 0.000% | 0.000% |
| AOTI fused | FAIL | 0 | 9/56 | 2.244% | 9.489% |

C++ and Python AOTI agree bit-for-bit on tested logits and full cache contents for both profiles. Eager numerics also matches HF BF16 bit-for-bit on all 163 tested rows and 56 cache tensors. Fused acceptance and error metrics above are preserved without relaxing the original thresholds; FP32 controls are retained in the full validation report.

Each benchmark request predicts 65 tokens: one prefill prediction and 64 decode predictions. Agreement with eager on the fixed benchmark inputs:

| Prompt tokens | Eager-numerics agreement | Fused agreement |
| ---: | ---: | ---: |
| 25 | 65/65 | 65/65 |
| 128 | 65/65 | 65/65 |
| 512 | 65/65 | 65/65 |

Agreement on these inputs does not establish general model quality. This benchmark does not include batching, sampling, HTTP serving, CUDA Graph replay, quantization, or optimized serving kernels.

## Reproduction and evidence

From the remote repository root, with the two package pairs already built:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python compiler/benchmark.py run \
  --eager-packages artifacts/aoti-eager --fused-packages artifacts/aoti \
  --lengths 25 128 512 --decode-steps 64 --warmup 2 --repeats 5 --rounds 3 \
  --output artifacts/benchmark-gb10-full-new
```

Use an absent output directory. This full rerun is under `artifacts/benchmark-gb10-full-rerun-20261023T150003Z/`: nine worker reports, idle checks, per-second telemetry, logs, source snapshots, hashes and summary. Fused validation is under `artifacts/three-path-fused/`; eager-numerics validation is under `artifacts/three-path-eager/`.

This tracked directory retains the summary, environment/package/source hashes, exact workload, run manifest, per-round results and a compact correctness report. Large model packages and raw traces remain Git-ignored.
