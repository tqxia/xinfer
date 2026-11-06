# GB10 environment

For a non-Docker Linux ARM64 host, use this device's Bash setup script:

```bash
bash environments/gb10/setup.sh --model qwen3
bash environments/gb10/setup.sh --model qwen35

Q3_PY="$PWD/work/gb10/qwen3-venv/bin/python"
Q35_PY="$PWD/work/gb10/qwen35-venv/bin/python"
Q3_BUILD="$PWD/build/qwen3"
Q35_BUILD="$PWD/build/qwen35"
```

The host needs Python 3.12, glibc 2.28 or newer, a C++ compiler, CUDA 13.0 toolkit
and compatible NVIDIA driver. Each invocation creates a fresh independent venv
and installs Torch `2.13.0+cu130` from the
[official CUDA 13.0 index](https://download.pytorch.org/whl/cu130/torch/), which
provides a Python 3.12 Linux ARM64 wheel. It also installs Triton `3.7.1`, the
model's pinned dependencies, CMake `4.3.4` and Ninja `1.13.2`.
Setup and checking use [`qwen3_requirements.txt`](qwen3_requirements.txt) or
[`qwen35_requirements.txt`](qwen35_requirements.txt). Each file contains the full
PyTorch index, Torch, Triton, CMake, Ninja and HF requirements for that model.
The model files can select different Torch versions.
The checker derives CUDA runtime `13.0` from the Torch pin. `profile.json`
checks Linux, ARM64 and GPU capability `12.1`. All GB10 package requirements
live in this directory; the Bash script contains no package pins.

Use `--python /path/to/python3.12` to select the bootstrap interpreter and
`--venv PATH` for a different fresh venv directory. `--torch-wheel PATH` selects
a local Torch wheel instead of the official download. Existing destinations are
refused. No packages are linked or inherited from an existing environment.

Setup does not need a GPU. Follow the [shared workflow](../README.md) to run
`check.sh --python PATH`, build with the model environment's CMake and LibTorch,
and validate Triton/AOTI execution before using new packages. Native builds use
`build/qwen3/` and `build/qwen35/`.

## Validation history

The earlier native installation was
`/home/tqxia/workspace/xinfer/.venv/bin/python`, with Torch Git revision
`cf30153c4c131c8164ee7798e5022d810682e2cb`. The records below describe that build;
repeat the environment and model checks for a newly installed wheel.

The workflow was exercised on GB10 on **2026-11-01**, in a detached checkout of
`4c282ce` plus the pending dual-target changes. Both model environments were
created from the then-recorded native installation. Dependency/profile/build checks and all
16 setup/cache-export tests passed. All seven native targets and the attention
library built without Python linkage. Both full-model package pairs were
compiled afresh with AOTI input guards and default numerical/optimization settings.

| Validation | Result |
|---|---|
| Qwen3 attention | 9 operator and 9 compiled cases pass |
| Qwen3 full runtime | 26 cases / 185 logit rows; native/AOTI logits and cache bitwise |
| Qwen3 CLI | 3 fixtures / 43 generated tokens; 22 rejection checks pass |
| Qwen3.5 standalone GDN | 100 cases / 560 native calls / 1,680 tensors bitwise; both prefill branches retained |
| Qwen3.5 full model | 20 cases / 87 steps; mask/write/input/repeat checks pass |
| Qwen3.5 native runtime | 35 cases; 5,243 teacher tensors, 30 greedy rows and 288 final-state tensors bitwise; 27 rejections pass |
| Qwen3.5 end-to-end | 294 teacher steps / 14,406 tensors bitwise between native and Python AOTI |

The GDN checkpoint fixture was absent, so the standalone suite used synthetic
cases; full-model tests exercised the verified real checkpoint. The maintained
validators ran directly, without the historical mmap/receipt helpers. Guarded
testing found that contiguous singleton slices can retain noncanonical strides;
test inputs now use explicit packed copies matching the native tensor ABI.

HF numerical acceptance remains separate. Qwen3 meets its logit gates but has
17 HF cache diagnostic failures. Qwen3.5 retains 4,511 convolution-state and 15
recurrent-state end-to-end failures, and the capital fixture generates
` Paris.\n` versus HF's ` Paris\n`. Its full numerical gate remains false.
No threshold was relaxed and no performance measurement was performed.

Reports and fresh packages are retained on GB10 under
`/home/tqxia/workspace/xinfer/artifacts/gb10-20261101/main-verification/`.
Repeat the shared validation matrix for each new package/toolchain.
