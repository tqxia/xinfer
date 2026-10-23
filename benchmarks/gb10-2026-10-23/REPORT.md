# GB10 full eager / AOTI benchmark

Measured on 2026-10-23 (Asia/Shanghai) in `/home/tqxia/workspace/xinfer`.
Source commit: `dcb07fa3f49e4e40b1bc07c46ccf587da383e110` on `dev-gb10`.

AOTI eager numerics improves decode by **1.48–1.49x** and prefill by
**1.33–1.50x** over the eager adapter in this run, while preserving bitwise
agreement on the validation suite. Fused AOTI improves decode by 1.61–1.62x
but fails the full-cache numerical criterion on 9/56 tensors. These full-run
measurements supersede the earlier short smoke timing for comparisons.

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
| 25 | Python eager | 31.00 | 29.22 | 34.2 | 1902.30 |
| 25 | AOTI eager numerics | 20.63 | 19.72 | 50.7 | 1283.58 |
| 25 | AOTI fused | 18.69 | 18.12 | 55.2 | 1179.37 |
| 128 | Python eager | 37.07 | 29.41 | 34.0 | 1920.05 |
| 128 | AOTI eager numerics | 26.19 | 19.71 | 50.7 | 1288.65 |
| 128 | AOTI fused | 23.79 | 18.14 | 55.1 | 1185.91 |
| 512 | Python eager | 72.89 | 29.36 | 34.1 | 1953.22 |
| 512 | AOTI eager numerics | 55.00 | 19.73 | 50.7 | 1319.23 |
| 512 | AOTI fused | 50.01 | 18.13 | 55.2 | 1211.74 |

Speedups use eager latency divided by AOTI latency.

| Prompt tokens | Eager-numerics prefill | Eager-numerics decode | Fused prefill | Fused decode |
| ---: | ---: | ---: | ---: | ---: |
| 25 | 1.50x | 1.48x | 1.66x | 1.61x |
| 128 | 1.42x | 1.49x | 1.56x | 1.62x |
| 512 | 1.33x | 1.49x | 1.46x | 1.62x |

## Variability and process loading

The p10–p90 intervals below describe the observed samples, not confidence intervals.

| Prompt tokens | Profile | Prefill p10–p90 ms | Decode p10–p90 ms/token |
| ---: | --- | ---: | ---: |
| 25 | Python eager | 30.83–31.66 | 28.91–29.53 |
| 25 | AOTI eager numerics | 20.55–20.67 | 19.63–19.76 |
| 25 | AOTI fused | 18.61–18.78 | 18.10–18.33 |
| 128 | Python eager | 36.89–37.52 | 29.26–29.61 |
| 128 | AOTI eager numerics | 26.15–26.35 | 19.65–19.76 |
| 128 | AOTI fused | 23.73–23.92 | 18.10–18.41 |
| 512 | Python eager | 72.77–73.55 | 29.27–29.98 |
| 512 | AOTI eager numerics | 54.92–55.74 | 19.69–19.78 |
| 512 | AOTI fused | 49.93–50.62 | 18.11–18.16 |

| Profile | Median fresh-process model/package load | Min–max load |
| --- | ---: | ---: |
| Python eager | 9.33 s | 7.90–9.89 s |
| AOTI eager numerics | 11.66 s | 11.62–11.84 s |
| AOTI fused | 11.43 s | 9.68–12.90 s |

Load time excludes Python import and CUDA initialization. The OS file cache was not cleared, so these are fresh-process loads, not controlled cold-disk loads. `rounds.json` records each process and its first requests.

## Clock and power context

Automatic clock control was retained. These are telemetry samples with at least 50% GPU utilization, spanning warmup and measured requests rather than individually aligned measurement windows.

| Profile | Active-sample median SM clock | Active-sample SM clock range | Median reported GPU power |
| --- | ---: | ---: | ---: |
| Python eager | 611 MHz | 611–1618 MHz | 6.11 W |
| AOTI eager numerics | 611 MHz | 611–910 MHz | 6.70 W |
| AOTI fused | 611 MHz | 611–611 MHz | 6.94 W |

The GPU reports a maximum SM clock of 3,003 MHz. A during-run spot check recorded P0 and no active thermal, power-cap or hardware slowdown; it is saved in `power-clock-snapshot.txt`. The sampled clock behavior does not establish its cause. This report measures the current automatic-clock configuration, not locked-clock peak hardware throughput. Reported GPU power is not total system/module power.

## Memory

Maximum across the three rounds. MiB = 1,048,576 bytes.

| Prompt tokens | Profile | Allocator peak allocated MiB | Allocator peak reserved MiB | Sampled device-use delta MiB |
| ---: | --- | ---: | ---: | ---: |
| 25 | Python eager | 1289.5 | 1292.0 | 2148.1 |
| 25 | AOTI eager numerics | 152.5 | 154.0 | 5138.6 |
| 25 | AOTI fused | 152.5 | 154.0 | 5182.4 |
| 128 | Python eager | 1291.8 | 1296.0 | 2102.4 |
| 128 | AOTI eager numerics | 154.8 | 158.0 | 5146.5 |
| 128 | AOTI fused | 154.8 | 158.0 | 5150.6 |
| 512 | Python eager | 1300.7 | 1322.0 | 2117.8 |
| 512 | AOTI eager numerics | 168.0 | 182.0 | 5182.0 |
| 512 | AOTI fused | 167.0 | 182.0 | 5197.9 |

AOTI weights can bypass the PyTorch allocator, so allocator peaks alone are not total model memory. Device-use deltas come from sampled `cudaMemGetInfo` values relative to each process before model loading; they are global and use GB10 unified memory. They are not exact per-process or transient peaks. Both AOTI loaders keep independent weight copies. NVML memory-used telemetry is unavailable on this GB10; supported utilization, power and clock telemetry is retained in the raw run directory.

## Numerical correctness

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

Use an absent output directory. The original full run is under `artifacts/benchmark-gb10-full/`: nine worker reports, idle checks, per-second telemetry, logs, source snapshots, hashes and summary. Fused validation is under `artifacts/three-path-fused/`; eager-numerics validation is under `artifacts/three-path-eager/`.

This tracked directory retains the summary, environment/package/source hashes, exact workload, run manifest, per-round results and a compact correctness report. Large model packages and raw traces remain Git-ignored.

To create the fused pair on this environment, use fresh output paths if they
already exist:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_HOME=/usr/local/cuda
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
export TORCHINDUCTOR_COMPILE_THREADS=2
export TORCHINDUCTOR_CACHE_DIR="$PWD/work/inductor-fused"
.venv/bin/python compiler/export.py decode --output artifacts/aoti
.venv/bin/python compiler/export.py prefill --output artifacts/aoti
.venv/bin/python compiler/validate_aoti.py --packages artifacts/aoti
.venv/bin/python compiler/validate_cpp.py --packages artifacts/aoti --output artifacts/three-path-fused-new
```

The last command is expected to retain a failing exit status for the recorded
full-cache numerical discrepancies. The accepted eager-numerics build and
validation commands are in `GB10.md` at the repository root.
