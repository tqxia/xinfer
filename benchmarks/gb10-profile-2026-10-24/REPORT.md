# Fused AOTI profiling on GB10

Measured 2026-10-24 (Asia/Shanghai), Qwen3-0.6B BF16, batch 1, on ASUS GX10 / NVIDIA GB10. PyTorch 2.13.0+cu130, Triton 3.7.1, driver 580.159.03; Nsight Systems 2025.3.2.

**The primary decode bottleneck is GPU execution of linear projections, especially the vocabulary head. The best first BF16 opportunity is removing padded/repeated-KV attention work. CUDA Graph replay provides a measured, smaller 8–9% gain.**

This profiles the same fused packages as the prior full benchmark. They retain their known numerical limitation: 9/56 full-cache tensors fail the existing comparison against HF eager. Profiling and the experiments here do not waive or fix that limitation.

## What was measured

- A separate native driver calls the unchanged production `Runtime`. NVTX ranges identify reset, prefill, each decode step, forward dispatch, and argmax/host retrieval.
- Two warmup requests per prompt length precede capture. Three measured requests at each length 25/128/512, with eight teacher-forced decode steps per request: 9 prefills and 72 decode steps. Loading and warmup are outside the trace.
- Each request reproduces the corresponding nine predictions from the full fused benchmark. The optional driver was also measured without profiling.
- The graph ablation uses full 64-step continuations, two warmups and three measured requests per mode/length, alternating direct/graph order. Both modes use persistent inputs and the same single-threaded AOTI runner.
- GPU activity was initially idle with no compute processes. During the profile, samples with utilization >=50% reported SM clocks 2,411–2,509 MHz (median 2,457), temperatures 52–55 C, and median GPU power 28.0 W. Clocks/power limits were not changed.
- GPU kernel duration and CUDA API/NVTX timing are measured. CPU statistical sampling and GPU hardware counters were unavailable under current permissions. DRAM-bandwidth and occupancy numbers are therefore not claimed.

## Decode time budget

Representative case: 128-token prompt, mean of 24 individual profiled decode steps. Percentages below use end-to-end step wall time. CUDA kernel and API times overlap; they must not be added together.

| Component | Time per token | Share of wall time |
| --- | ---: | ---: |
| Linear projections, including vocabulary head | 7.316 ms | 68.2% |
| Attention kernels | 1.406 ms | 13.1% |
| KV-head expansion copies | 0.732 ms | 6.8% |
| Other GPU work | 0.234 ms | 2.2% |
| Time without a recorded GPU operation | 1.040 ms | 9.7% |

Total: **10.728 ms/token**. GPU operations occupy **90.3%** of the interval. Linear projections account for **75.5% of GPU kernel time**.

There are **454 kernels per decode token**: 197 linear projections (7 per layer plus vocabulary head), 28 attention kernels, 56 KV expansion copies, 56 fused cache-update kernels, 86 other fused kernels, 28 SiLU/multiply kernels, one argmax, and two input-fill kernels. One additional device-to-host copy transfers an 8-byte token ID; a small memset is also recorded.

The host spends about 2.14 ms dispatching forward work and 8.59 ms in the argmax/host-retrieval range. The latter contains about 8.57 ms in `cudaStreamSynchronize`, waiting for previously launched GPU work. The argmax kernel itself takes only about 7 microseconds. Removing the synchronization call alone would not recover those 8.57 ms.

## Linear projections: largest cost

The generated wrapper issues 197 cuBLAS matrix products per token. At decode batch 1 these are matrix-vector-shaped operations: approximately one FLOP per BF16 weight byte before other traffic. This is strong evidence of weight-traffic / small-matrix efficiency limits, rather than dense matrix-multiplication compute saturation; it is an inference from shapes and timings, not a hardware-counter bandwidth measurement.

The projection breakdown below averages all 72 decoded steps across the three prompt lengths; the main time-budget table uses the 128-token case only.

| Projection | Calls per token | Mean GPU time per token |
| --- | ---: | ---: |
| q_proj | 28 | 0.730 ms |
| k_proj | 28 | 0.291 ms |
| v_proj | 28 | 0.291 ms |
| o_proj | 28 | 0.763 ms |
| gate_proj | 28 | 0.821 ms |
| up_proj | 28 | 0.918 ms |
| down_proj | 28 | 1.487 ms |
| lm_head | 1 | 2.037 ms |

The vocabulary head is a 151,936 x 1,024 BF16 matrix (311.2 MB, 296.75 MiB). It alone consumes about **2.04 ms/token**, roughly 21% of GPU kernel time. Total logical linear-weight reads are about 1.192 GB/token; actual DRAM traffic can differ due to caches and kernel access patterns.

Opportunities: benchmark better kernels/autotuning for these exact matrix shapes; combine Q/K/V and gate/up projections where profitable; evaluate weight-only quantization if changing precision is acceptable. Combining projections reduces launches but does not remove weight bytes. Quantization targets the largest traffic term, but dequantization overhead and model-quality changes must be measured. For serving throughput, batching can reuse weights across requests, but requires extending the current batch-1 contract and is not a guaranteed per-request latency improvement.

## Attention: avoidable work and a measured candidate

The compiled path uses memory-efficient attention with a dense future mask over all **1,024 cache slots**. Before every layer it expands **8 KV heads to 16 query heads**, materializing 224 MiB of temporary KV outputs per decoded token across 28 layers. This is logical materialized data, not measured DRAM traffic. The generated calls and shapes are preserved in `generated-code-excerpts.txt`.

Attention plus expansion takes about **2.14 ms/token** at the representative decode point. At a 512-token prefill it takes **6.04 ms of 17.61 ms**, or 34.3% of wall time. This also explains why current decode latency changes little between short and long prompts: the attention key dimension remains 1,024.

A synthetic attention-only probe compared the current full-cache/repeated-KV path with FlashAttention on the valid prefix, using native GQA. Decode has no future keys after slicing, so it needs no causal mask; initial prefill uses causal attention over its populated prefix.

| Phase | Prompt tokens | Current path per layer | Valid-prefix Flash/native GQA | Attention-only ratio |
| --- | ---: | ---: | ---: | ---: |
| decode | 25 | 72.02 us | 4.55 us | 15.8x |
| decode | 128 | 72.24 us | 6.78 us | 10.6x |
| decode | 512 | 71.50 us | 7.59 us | 9.4x |
| prefill | 25 | 74.08 us | 4.61 us | 16.1x |
| prefill | 128 | 86.59 us | 7.81 us | 11.1x |
| prefill | 512 | 227.24 us | 31.50 us | 7.2x |

**These ratios are not end-to-end speedups.** This is a combined backend/layout/length change on synthetic tensors, with repeated data and CUDA Graphs amortizing Python dispatch. It does not isolate the individual contributions or reproduce all model-layer cache behavior. Outputs were finite; maximum observed relative L2 difference was 0.2405% and maximum absolute difference 0.00390625. This is not full-model numerical acceptance.

Implementing it requires an attention path that accepts native KV heads and valid lengths. A decode-specific kernel can consume the existing cache with a sequence-length argument; alternatively use carefully chosen length buckets. Initial prefill can use the populated prefix and causal attention. Chunked prefill starting at a nonzero cache position requires the correct causal alignment to that prefix. Validate masking, cache mutation, boundary positions, logits, and the full cache again after export. Merely flipping a backend flag is insufficient with the current explicit repetition and dense mask.

## CUDA Graph replay: modest gain confirmed

The existing default AOTI loader initially rejected capture because its completion-event query is not capture-safe. The installed loader exposes a supported `run_single_threaded=true` path. The isolated diagnostic uses that path, persistent input/position buffers, and captures decode plus argmax. Input updates and one host token retrieval per step remain outside replay.

| Prompt tokens | Direct dispatch | Graph replay | Speedup | Replay tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 25 | 10.534 ms/token | 9.677 ms/token | 1.089x | 103.3 |
| 128 | 10.558 ms/token | 9.664 ms/token | 1.092x | 103.5 |
| 512 | 10.467 ms/token | 9.700 ms/token | 1.079x | 103.1 |

All measured direct/graph prediction sequences match the original fused benchmark. For each measured pair, final logits and all 56 full-context KV tensors match bit-for-bit between direct and graph execution. This verifies equivalence to the fused path, not correction of its discrepancy versus HF eager.

This is a diagnostic ablation, not the full three-process benchmark or a change to the production CLI. Both modes use the same persistent-input/single-threaded setup, so the ratio isolates replay within that setup. A production integration must enforce one caller per runner, retain stable storage and output lifetimes, and preserve EOS handling and generation limits. Capture memory and startup costs also need measurement.

## Recommended implementation order

1. **Native-GQA attention using valid cache lengths.** Best first BF16 opportunity: it directly removes measured redundant work, particularly for longer prefill. Keep the current path as a reference.
2. **Persistent inputs plus CUDA Graph decode.** A measured 8–9% gain with parity against fused direct execution. Integrate the supported single-threaded runner contract explicitly.
3. **Linear-kernel tuning, then a separate quantization track.** Linear layers dominate after the first two changes; prioritize the vocabulary head and MLP down projection. Quantization has greater potential to reduce weight traffic but changes the numerical/quality contract.
4. **Batching if aggregate serving throughput is the objective.** The current model/runtime is explicitly batch 1; this needs a broader design change.

RMSNorm, SiLU and standalone argmax are small fractions of the current budget. Cache reset costs about 0.72 ms once per request, not per generated token. They are lower priorities for long-generation latency. Avoid interpreting the large synchronization entry in the API table as an independent CPU bottleneck.

## Measurement limits and provenance

- Nsight timing sums are grouped by CUDA launch correlation within NVTX ranges. GPU busy time is the union of kernel/memory-operation intervals; idle time is the remainder of each synchronized stage. Nested ranges and overlapping CPU/GPU work are not double-counted.
- Profiler perturbation was small in the observed decode medians: unprofiled 10.592/10.652/10.661 ms versus traced 10.704/10.656/10.831 ms for 25/128/512 prompts. These short captures are for attribution; use the full benchmark for performance claims.
- The prior full benchmark measured about 97.6–98.0 fused tokens/s at median clocks around 2.51 GHz. This profile sampled somewhat lower clocks and uses fewer decode steps; its absolute timings should not replace that benchmark.
- The GPU driver exposes hardware counters only to administrators; noninteractive sudo was unavailable. Kernel occupancy, achieved DRAM bandwidth, SM stall reasons and roofline saturation remain unmeasured. No permission, clock, power or firmware settings were changed.
- Model-package hashes and the production benchmark executable match the prior benchmark. Source commit before adding these diagnostics: `d877b53`. Exact source/package hashes and tool versions are in `environment.json`.

## Reproduce

From `/home/tqxia/workspace/xinfer`, using fresh output paths:

```bash
cmake -S . -B build -DQWEN_BUILD_PROFILE=ON
cmake --build build --target qwen_profile qwen_graph_probe -j2
mkdir -p artifacts/profile-fused-new
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=artifacts/profile-fused-new/fused \
  ./build/qwen_profile artifacts/aoti \
  benchmarks/gb10-profile-2026-10-24/workload.json \
  artifacts/profile-fused-new/profiled.json
nsys export --type=sqlite --output=artifacts/profile-fused-new/fused.sqlite \
  artifacts/profile-fused-new/fused.nsys-rep
.venv/bin/python compiler/analyze_profile.py \
  artifacts/profile-fused-new/fused.sqlite artifacts/profile-fused-new/analysis.json
./build/qwen_graph_probe artifacts/aoti \
  benchmarks/gb10-2026-10-23-rerun/workload.json \
  artifacts/profile-fused-new/graph-probe.json
.venv/bin/python compiler/profile_attention_probe.py \
  artifacts/profile-fused-new/attention-probe.json
```

Raw traces, SQLite export, full per-stage analysis and generated-code evidence are under `artifacts/profile-fused-gb10-20261024/`. Compact evidence accompanies this report in `benchmarks/gb10-profile-2026-10-24/`. The profiling tools are optional; the existing runtime, packages and inference CLI remain unchanged.

References: [Nsight Systems range attribution](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html), [PyTorch native GQA design](https://dev-discuss.pytorch.org/t/added-grouped-query-attention-to-scaled-dot-product-attention-api/2340), [FlashInfer KV-cache decode interface](https://docs.flashinfer.ai/generated/flashinfer.decode.single_decode_with_kv_cache.html).
