# Maintained development tools

Run the commands below from the repository root. The shell entry points remain
in `scripts/`. These utilities support setup and profiling; model export and
validation remain in `compiler/`.

| Tool | Role | When to use it |
|---|---|---|
| [setup/download_model.py](setup/download_model.py) | Download the pinned Qwen3 checkpoint and verify its hashes; verify an existing manifest without downloading again | New environments or checkpoint verification |
| [setup/fix_cusparselt_metadata.py](setup/fix_cusparselt_metadata.py) | Repair the cuSPARSELt 0.8.1 ARM64 wheel's platform metadata; never change library bytes | Called by `scripts/setup-gb10.sh`; remove when the pinned dependency no longer needs it |
| [setup/fetch_flashinfer.sh](setup/fetch_flashinfer.sh) | Fetch pinned FlashInfer and CCCL headers without installing Python packages | Optional native FlashInfer decode builds; see the [guide](../docs/gb10-flashinfer-decode.md) |
| [profiling/profile.cpp](profiling/profile.cpp) | Add NVTX stages around the shared inference runtime | Profile new packages or investigate performance changes |
| [profiling/analyze_profile.py](profiling/analyze_profile.py) | Attribute CUDA work to those stages and summarize kernel/API time | Analyze a new Nsight Systems SQLite export |

The metadata repair is restricted to a Linux AArch64 venv and the affected
version/tag; it is a no-op after correction. The downloader accepts
`--model-dir`, retains checkpoint identity checks, and needs no `PYTHONPATH`.

```bash
.venv/bin/python tools/setup/download_model.py
cmake -S . -B build -DQWEN_BUILD_PROFILE=ON
cmake --build build --target qwen_profile -j2
mkdir -p artifacts/profile-new
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nsys profile \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=artifacts/profile-new/model \
  ./build/qwen_profile artifacts/aoti-flash \
  benchmarks/gb10-profile-2026-10-24/workload.json \
  artifacts/profile-new/profiled.json
nsys export --type=sqlite --output=artifacts/profile-new/model.sqlite \
  artifacts/profile-new/model.nsys-rep
.venv/bin/python tools/profiling/analyze_profile.py \
  artifacts/profile-new/model.sqlite artifacts/profile-new/analysis.json
```

Use fresh output paths. Configure the GB10 toolchain with `scripts/build-gb10.sh`
first. The profiling driver uses the same `Runtime` as inference and accepts
both efficient and custom-op Flash packages. The analyzer expects this driver's
NVTX ranges and a compatible Nsight SQLite schema. Kernel categories use name
heuristics, including the assumption that the last linear kernel is the LM head;
review those classifications after toolchain or model changes.

Historical measurements and source hashes under `benchmarks/` retain their
original paths. Experiment-specific probes live in [experiments/](../experiments/README.md),
with a separate `QWEN_BUILD_EXPERIMENTS` option. The small BF16 RoPE reproducer
remains in [compiler/tests/rope_precision.py](../compiler/tests/rope_precision.py)
for investigations of Torch/Triton numerical changes.
