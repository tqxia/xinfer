# Historical optimization experiments

These probes preserve the experiments behind the [GB10 fused-AOTI profiling
report](../benchmarks/gb10-profile-2026-10-24/REPORT.md). They are manually run
reproducers with narrow assumptions. Use `compiler/benchmark.py`,
`compiler/validate_cpp.py` and
`compiler/tests/custom_attention.py` to assess the current implementation.

## Attention opportunity probe

[attention_probe.py](attention_probe.py) compares efficient SDPA over a full
1,024-slot cache with repeated KV heads against native-GQA Flash SDPA over the
valid prefix. It fixes the original Qwen dimensions and prompt lengths 25/128/512,
uses synthetic inputs, and checks finite outputs while recording numerical errors.
It does not load model packages or call `xinfer::flash_attention`.

CUDA Graph replay amortizes Python dispatch in this microbenchmark. Reused tensors
and repeated attention calls do not reproduce the cache behavior of 28 model
layers, and this probe does not enable graph replay in the inference runtime.

```bash
.venv/bin/python experiments/attention_probe.py artifacts/attention-probe-new.json
```

## CUDA Graph opportunity probe

[cuda_graph_probe.cpp](cuda_graph_probe.cpp) compares direct calls with CUDA Graph
replay using persistent inputs and checks predictions, final logits and full
caches for equality. It supports only efficient-attention AOTI packages. It has
no extent input or strategy for changing valid-prefix shapes during replay, so
it rejects Flash packages before initializing CUDA. Supporting those packages
requires a separate design and correctness investigation.

```bash
cmake -S . -B build -DQWEN_BUILD_EXPERIMENTS=ON
cmake --build build --target qwen_graph_probe -j2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./build/qwen_graph_probe artifacts/aoti \
  benchmarks/gb10-2026-10-23-rerun/workload.json \
  artifacts/graph-probe-new.json
```

Configure the GB10 toolchain first and use fresh output paths. Both CMake options
`QWEN_BUILD_PROFILE` and `QWEN_BUILD_EXPERIMENTS` default to OFF; enabling profiling
builds the maintained NVTX driver, while enabling experiments builds this probe.
Neither changes normal inference behavior.
