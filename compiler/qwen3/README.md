# Qwen3-0.6B AOTI / C++

Qwen3 has one supported compiled profile: fused BF16 AOTInductor with valid-prefix
FlashAttention and native GQA. The original HF model supplies the weights and
independent eager/FP32 references. The runtime performs inference and generation
in C++ without Python.

The former eager-numerics compilation profile and efficient-attention export
backend are retired. Existing efficient packages must be re-exported. Known HF
cache/token differences are documented in [numerical history](../../docs/experiments/qwen3-numerics.md);
[the attention guide](../../docs/flash-attention.md) describes the fixed operator
and package ABI. Qwen3.5 retains its separate adapter, environment and policy.

## Environment

Run these commands on `kikoxia-any4.devcloud.woa.com`, port `36000`, as `root`.

```bash
cd /data/workspace/develop/qwen3-aoti-poc
```

The project `.venv` contains its own Hugging Face dependencies. Only Torch,
Triton, and their NVIDIA packages are linked from `/data/workspace/abo/.venv`.
ABO's unrelated packages are not on the project's import path. These links are
read-only dependencies by convention: do not upgrade or uninstall the shared
packages through this venv. The C++ build must use this same Torch build.

- Python 3.12.13; PyTorch runtime `2.7.0+cu126`; Triton 3.3.0.
- Transformers 4.51.3; tokenizers 0.21.4; safetensors 0.5.3.
- CUDA build 12.6; C++11 ABI enabled; NVIDIA L20, compute capability 8.9.
- `compiler/qwen3/requirements.txt` pins the main packages; `compiler/qwen3/requirements.lock` records the
  complete resolved package set. Torch's distribution metadata says `2.7.0`;
  its runtime version includes `+cu126`.

To recreate the project venv on this host, first create it in an absent `.venv`
directory, then link the existing native packages and install the locked extras:

```bash
/data/workspace/abo/.venv/bin/python -m venv .venv
.venv/bin/python - <<'PY'
from pathlib import Path
base = Path('/data/workspace/abo/.venv/lib/python3.12/site-packages')
dest = Path('.venv/lib/python3.12/site-packages')
paths = [base / name for name in ('torch', 'torchgen', 'functorch', 'triton', 'nvidia')]
for pattern in ('torch-*.dist-info', 'triton-*.dist-info', 'nvidia*.dist-info'):
    paths.extend(base.glob(pattern))
for source in paths:
    if not source.exists():
        raise FileNotFoundError(source)
    (dest / source.name).symlink_to(source, target_is_directory=True)
PY
.venv/bin/python -m pip install -r compiler/qwen3/requirements.lock
.venv/bin/python -m pip check
```

## Prepare offline inputs

The default checkpoint is
`/data/workspace/develop/qwen3-rust-demo/models/qwen3-0.6b`, revision
`c1899de289a04d12100db370d81485cdf75e47ca`. Helpers verify and read these files;
they do not modify the checkpoint or download another copy.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
.venv/bin/python -m compiler.qwen3.prepare verify
.venv/bin/python -m compiler.qwen3.prepare encode \
  --prompt 'What is the capital of France? Answer in one short sentence.' \
  --output work/prompt.ids
# Run once in an absent output directory to generate independent HF fixtures.
.venv/bin/python -m compiler.qwen3.tests.validate --output artifacts/reference
```

Reference generation checks manual HF DynamicCache decoding against HF
`generate()` and a fresh-cache repetition. It writes prompt/generated IDs,
little-endian float32 logits, text, metadata and a summary for factual,
arithmetic and Chinese fixtures. Thinking is disabled and greedy decoding is a
repeatable diagnostic. Both checkpoint EOS IDs are honored. Existing reference
fixtures remain usable; do not overwrite them during a compiler comparison.

## Build before export

The Qwen3 build always produces and links `libxinfer_attention_ops.so`; there is
no attention build switch. Both Python export and native inference require this
library and matching LibTorch/CUDA. The first CMake configuration fetches the
pinned nlohmann/json headers; later builds reuse the cached dependency.

```bash
/data/workspace/abo/.venv/bin/cmake -S . -B build -G Ninja \
  -DCMAKE_PREFIX_PATH=/data/workspace/abo/.venv/lib/python3.12/site-packages/torch/share/cmake \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DTORCH_CUDA_ARCH_LIST=8.9 -DCMAKE_BUILD_TYPE=Release \
  -DQWEN_BUILD_VALIDATION=ON -DQWEN_BUILD_BENCHMARK=ON
/data/workspace/abo/.venv/bin/cmake --build build -j2
export XINFER_ATTENTION_OPS_LIBRARY="$PWD/build/libxinfer_attention_ops.so"
```

Python defaults to `build/libxinfer_attention_ops.so`; the environment variable
selects another build directory. Native binaries link the library directly and
must be deployed with it and their matching LibTorch/CUDA libraries. A `.pt2`
package does not bundle this external operator. Native linkage must remain free
of `libpython` and `libtorch_python`.

## Export the canonical package pair

Choose a new directory. The exporter refuses to replace existing packages.
These examples intentionally leave historical `artifacts/aoti/` packages alone.

```bash
export TORCHINDUCTOR_COMPILE_THREADS=2
export TORCHINDUCTOR_CACHE_DIR="$PWD/work/inductor-qwen3-flash"
.venv/bin/python -m compiler.qwen3.export decode --output artifacts/qwen3-flash
.venv/bin/python -m compiler.qwen3.export prefill --output artifacts/qwen3-flash
```

Both commands accept `--model-dir`. There is no numerical-mode or attention
selector. Export retains `emulate_precision_casts=True` and disables TF32;
ordinary Inductor fusion remains enabled. Use the same toolchain and tuning
cache for both entry points and record their actual compilation settings.
Changing the compiler or tuning cache can change rounding, so validate every
new pair rather than assuming cross-build bitwise reproducibility.

The directory contains `prefill.pt2`, `decode.pt2`, matching JSON metadata, and
exported graph text. Metadata records the fixed Flash backend, custom operator
ABI, extent/cache contract, checkpoint, Torch/CUDA versions, GPU architecture,
C++ ABI and actual numerical settings. Keep these fields even though the CLI no
longer selects a backend: they prevent incompatible packages being mixed.

### Package contract

Python signature: `(input_ids, cache_position, (K0, V0, ..., K27, V27), kv_extent)`.
The native loader receives 59 flat inputs: IDs, positions, 56 cache tensors, and
the extent tensor. Inputs are:

- CUDA int64 IDs `[1,T]` and contiguous absolute positions `[T]`.
- Caller-owned contiguous CUDA BF16 K/V buffers `[1,8,1024,128]`, sharing the
  same stable storage between prefill and decode. Native allocations satisfy
  the 16-byte alignment requirement.
- A CPU uint8 extent vector of shape `[populated_length + 1]`. Only its shape is
  read; its data is ignored. This carries prefix length without a CUDA scalar
  synchronization or size-one specialization.
- One output: last-token CUDA BF16 logits `[1,151936]`.

Prefill accepts 2–512 tokens; decode accepts one token, including a one-token
prompt at position zero. Total context capacity is 1,024. Earlier positions
must already be populated. Runtime checks reject invalid IDs, nonpositive or
oversized budgets, incompatible metadata and cache overflow before model work.
This is an inference-only, unpadded, batch-one Qwen3-0.6B contract; arbitrary
masks, training and attention-weight requests are unsupported.

## Validate

```bash
.venv/bin/python -m compiler.qwen3.tests.custom_attention \
  --output artifacts/qwen3-attention-check
.venv/bin/python -m compiler.qwen3.tests.attention_adapter \
  --output artifacts/qwen3-adapter-check
.venv/bin/python -m compiler.qwen3.tests.validate_aoti --packages artifacts/qwen3-flash
AOTI_RUNTIME_CHECK_INPUTS=1 .venv/bin/python -m compiler.qwen3.tests.validate_cpp \
  --packages artifacts/qwen3-flash --native build/qwen_validate \
  --output artifacts/qwen3-validation
.venv/bin/python -m compiler.qwen3.tests.smoke_cpp \
  --packages artifacts/qwen3-flash --output artifacts/qwen3-native-smoke
```

Use fresh output directories. Validation separates two conclusions:

- Runtime correctness requires finite results, bitwise native/Python AOTI
  agreement, correct cache writes/masking, stable storage, reset/reuse, legal
  boundaries/overflow rejection and matching native/Python generation.
- HF logit error remains bounded by the existing 5% row-relative L2 and
  peak-normalized error limits. HF cache differences and differing top-token or
  greedy-sequence choices are recorded diagnostics, with parity reported
  separately from the overall result. A successful run does not claim HF
  bitwise or token parity.

Logit comparisons use identical fed tokens, including when greedy trajectories
differ. Independent HF eager and FP32 controls remain in the reports. Focused
attention/operator checks retain their own numerical correctness requirements.
The original HF SDPA reference is not replaced by the compiled Flash adapter.
See [numerical history](../../docs/experiments/qwen3-numerics.md) for the recorded
14/56 cache differences and two top-token mismatches on L20.

## Generate in C++

```bash
./build/qwen_aoti artifacts/qwen3-flash work/prompt.ids work/generated.ids \
  --max-new-tokens 64 --trace work/native-logits.f32
.venv/bin/python -m compiler.qwen3.prepare decode --input work/generated.ids
```

The native executable runs without an activated venv or interpreter. It resets
the cache for each request and stops at EOS or the requested limit. Output IDs
include EOS when reached; trace rows are raw little-endian float32. Input,
output and trace paths must differ. The prompt plus complete requested budget
must fit 1,024 tokens even if EOS might arrive earlier. Select the visible GPU
with `CUDA_VISIBLE_DEVICES`.

## Benchmark and profile

```bash
.venv/bin/python -m compiler.qwen3.benchmark run \
  --packages artifacts/qwen3-flash --output artifacts/qwen3-benchmark
```

`run` compares HF eager with one compiled profile using identical forced tokens.
`compare` compares two compiled artifact directories for optimization work.
Both preserve rotated process order, warmups, source/package hashes, memory and
GPU telemetry. Use the [development tools guide](../../docs/development-tools.md)
for paired comparisons and the optional NVTX profiler. Shared-GPU smoke timings
are provisional; controlled performance measurements require the idle checks.
Historical GB10 speedups do not establish a speedup for another toolchain/GPU.
