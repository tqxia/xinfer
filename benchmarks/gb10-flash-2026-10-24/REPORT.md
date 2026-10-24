# GB10: fused AOTI versus native FlashAttention

Date: 2026-10-24 (Asia/Shanghai). Implementation commit: `add91eafc9b5a7bd8f6503461589becd0d0c7def` on `dev-gb10`.
Host: `tqxia@192.168.1.36`, repository `/home/tqxia/workspace/xinfer`.

The optional FlashAttention path uses native GQA and attends only to the valid KV
prefix. The original fused packages are unchanged. This comparison uses no CUDA
Graph replay. **Functional checks pass, but numerical acceptance remains failed:**
17/56 full-context cache tensors exceed the unchanged 5% limits, versus 9/56 for
the original fused baseline. The new backend remains experimental and is not the default.

## End-to-end native performance

Medians of 20 measured requests per length/profile. Prefill and decode include
GPU argmax and the host token read. Decode is the per-token average across 64
forced steps. These are runtime measurements, not isolated attention speedups.

| Prompt tokens | Fused prefill ms | Flash prefill ms | Prefill speedup | Fused decode ms/token | Flash decode ms/token | Decode speedup | Decode tokens/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 10.359 | 7.394 | 1.40× | 10.535 | 7.973 | 1.32× | 94.9 → 125.4 |
| 128 | 11.160 | 8.095 | 1.38× | 10.537 | 8.096 | 1.30× | 94.9 → 123.5 |
| 512 | 17.512 | 11.733 | 1.49× | 10.524 | 8.285 | 1.27× | 95.0 → 120.7 |

| Prompt tokens | Profile | Prefill p10–p90 ms | Decode p10–p90 ms/token | Median request ms, including reset | Peak Torch allocated MiB |
|---:|---|---:|---:|---:|---:|
| 25 | aoti_fused | 10.333–10.390 | 10.500–10.601 | 685.32 | 152.5 |
| 25 | aoti_flash | 7.376–7.419 | 7.969–7.998 | 518.40 | 144.7 |
| 128 | aoti_fused | 11.127–11.221 | 10.502–10.597 | 686.23 | 154.8 |
| 128 | aoti_flash | 8.071–8.131 | 8.083–8.110 | 526.98 | 146.8 |
| 512 | aoti_fused | 17.330–17.607 | 10.440–10.598 | 691.83 | 167.0 |
| 512 | aoti_flash | 11.667–11.766 | 8.259–8.297 | 542.70 | 155.0 |

Four fresh-process rounds alternate baseline/candidate order. Each worker runs
one first-use request, two warmups, and five measured requests at each length.
There are 120 measured requests in total. All 16 before/after idle checks passed;
no `--allow-busy` override was used. Both profiles use the same rebuilt binary,
one CPU thread, checkpoint, tokens, BF16 weights/cache, fusion settings and context
capacity. No weights, dependency versions, power settings or clocks were changed.

Active telemetry (GPU utilization at least 50%): baseline median SM clock
2509 MHz (range 2405–2515),
Flash median 2502 MHz
(range 2457–2509).
These are automatic-clock results. Per-round speedups and telemetry are in
`analysis.json` and the individual worker files. Candidate/base prediction agreement
is 65/65, 65/65, and 65/65
at lengths 25, 128 and 512, respectively; every repeated request is deterministic.
Torch allocator peaks exclude some AOTI weight allocations; device-memory samples
in `summary.json` are sampled process deltas, not exact transient peaks.

## Correctness and numerical limits

- Strict `AOTI_RUNTIME_CHECK_INPUTS=1` validation: 26 cases and 185 logit rows.
- All 185 top-1 predictions and fixture greedy sequences agree with HF BF16.
- Native/Python AOTI logits and all 56 full-context cache tensors agree bitwise.
- Cache storage, write regions, reset/reuse, poisoned future slots, one-token
  prompts, lengths 2/7/128/511/512, offset chunked prefill, final legal slot and
  overflow rejection pass. Extra chunk boundaries reach valid lengths
  1/3/16/32/63/64/128/256/512/1023/1024, with and without poisoned future storage.
- Native CLI smoke tests pass, including invalid inputs, mixed-package rejection,
  generation limits and Python-free linkage/loading.
- The rebuilt runtime reproduces all 25 recorded baseline logit/cache files
  bitwise (24 cases, 163 rows), independently confirming baseline compatibility.
- Nine isolated attention shapes match an explicit FP32 masked/GQA reference
  within the existing 5% limits, and are bitwise invariant to future-slot poison.

| Profile | Worst logits L2 vs HF BF16 | Full-cache L2 vs HF BF16 | Full-cache peak-relative vs HF BF16 | Failing cache tensors | Full-cache L2 vs FP32 | Full-cache peak-relative vs FP32 |
|---|---:|---:|---:|---:|---:|---:|
| Fused baseline | 3.110% | 2.244% | 9.489% | 9/56 | 4.142% | 26.805% |
| Fused + FlashAttention | 3.883% | 5.441% | 44.055% | 17/56 | 5.094% | 18.566% |

The original 5% criteria are unchanged. `validate_cpp.py` intentionally exits 1
after recording the complete numerical report. Eager FlashAttention also shows
the large cache difference versus eager efficient attention: worst relative L2
5.476% and
peak-relative error 44.512%.
Thus the large difference is present without AOTI. The isolated FP32 test supports
correct causality/GQA semantics. The full-model FP32 control has mixed results:
Flash has a lower worst peak-relative error but a higher worst relative L2 error.
Neither bitwise equivalence nor an overall quality improvement is claimed.
This finite fixture suite is not a model-quality evaluation.

## Implementation

- `compiler/model.py` registers a small project-owned HF attention adapter, retaining
  Qwen's projections, Q/K normalization, RoPE and output projection. It invokes
  `aten._scaled_dot_product_flash_attention`, the same native operator used by
  PyTorch's lower-right causal-bias dispatcher. Its lower-right alignment handles
  initial prefill, offset chunked prefill and single-token decode uniformly.
- K/V retain eight heads. There is no 8→16 materialized head expansion and no dense
  attention mask. Generated code contains 28 FlashAttention calls per forward and
  zero efficient-attention calls. The calls receive cache views directly;
  generated wrappers have no cache-copy API calls. See `generated-code-excerpts.txt`.
- The native runtime retains 56 caller-owned logical `[1,8,1024,128]` buffers.
  One spare storage position per head keeps prefix strides uniform at capacity
  (98 KiB extra in the native allocation; 112 KiB in Python's export allocation).
  A persistent 1,025-byte CPU buffer provides a view of
  length `valid_length+1`; only its shape is read. The compiled model updates the
  fixed-size caches and constructs the valid-prefix views internally.
- Fixed mutation-input shapes plus the extent tensor avoid AOTI's size-one
  specialization. PyTorch 2.13's experimental `backed_size_oblivious` export mode
  handles intermediate size-one views. Strict input checks remain enabled during
  validation; no guards or error tolerances are disabled. No extra package is needed.
- Metadata enforces matching attention, cache layout, shape contract, numerical
  mode and Torch ABI. Older full-cache packages remain supported.

The direct native attention operator and symbolic export option are internal or
experimental PyTorch interfaces; the exact Torch 2.13/CUDA 13 environment is pinned
and packages must be regenerated and validated after toolchain changes.
Final decode/prefill export+compile times were 30.0/32.6 seconds;
package sizes are 1,196,566,407/1,196,879,511 bytes. Earlier guarded
prefix-view probes are preserved separately under `artifacts/*probe*`; they are
not the packages benchmarked here.

## Kernel verification and remaining bottleneck

A separate short Nsight capture after the timed benchmark covers three prefills
and 24 decode steps, following two warmup requests per length. All
11,248 GPU kernels are accounted for by the NVTX ranges.
The trace contains `pytorch_flash::flash_fwd_kernel`,
`pytorch_flash::flash_fwd_splitkv_kernel`, and split-combine variants, with no
KV-head-expansion kernels. See `profile-stats_cuda_gpu_kern_sum.csv` for exact names.
Decode launches 397 kernels/token at length 25 and 425 at lengths 128/512;
the latter use an additional split-combine kernel per layer.

At prompt length 128, attention takes 0.254 ms/token
in the short trace. Linear projections plus the vocabulary head take 6.811 ms/token,
or 93.3% of GPU kernel time. The vocabulary head alone takes
2.018 ms/token. Linear/weight-read performance remains the main next
opportunity. These trace values are diagnostic and are not substituted for the
unprofiled benchmark results above; no DRAM hardware counters were collected.

## Reproduce

Run from `/home/tqxia/workspace/xinfer`. Choose fresh artifact directories when rerunning.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python compiler/export.py decode --attention-backend flash --output artifacts/aoti-flash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python compiler/export.py prefill --attention-backend flash --output artifacts/aoti-flash
./scripts/build-gb10.sh
AOTI_RUNTIME_CHECK_INPUTS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python compiler/validate_cpp.py --packages artifacts/aoti-flash --output artifacts/three-path-flash
# The validator records the known numerical failure; inspect summary.json.
PYTHONPATH=compiler OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python compiler/tests/flash_attention.py --validation artifacts/three-path-flash --output artifacts/aoti-flash/attention-validation.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python compiler/benchmark.py compare --baseline-packages artifacts/aoti --candidate-packages artifacts/aoti-flash --output artifacts/benchmark-gb10-flash-20261024
```

`environment.json` records source/binary/package hashes, package metadata and the
model revision. Raw logits, cache dumps and source snapshots remain under
`artifacts/three-path-flash`, `artifacts/three-path-fused-current-native`,
`artifacts/native-smoke-flash` and `artifacts/benchmark-gb10-flash-20261024`.
This directory preserves compact results and supporting evidence.
