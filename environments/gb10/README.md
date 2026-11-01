# GB10 native stack

The profile describes the GB10 stack validated with the shared main code:

| Component | Recorded value |
|---|---|
| Python | 3.12 |
| Torch runtime | 2.13.0+cu130 |
| Torch Git revision | cf30153c4c131c8164ee7798e5022d810682e2cb |
| Triton | 3.7.1 |
| CUDA used by Torch | 13.0 |
| GPU compute capability | 12.1 |

The recorded native Python is
`/home/tqxia/workspace/xinfer/.venv/bin/python`. Use that absolute path as
`NATIVE_PY` in the [shared workflow](../README.md). A supplied wheel/container
installation must match `profile.json`; the runtime version string alone does
not identify its exact native build. The setup helper records the actual
Torch Git revision, package locations and available installation provenance.
Ninja is in the native venv's `bin/` directory; the shared workflow adds that
directory to `PATH`. The local CUDA compiler is `/usr/local/cuda/bin/nvcc`.
Use the shared workflow's Bash entry points: `setup.sh` to create a model venv,
then `check.sh --python PATH` to inspect that model environment.

The repository does not contain the GB10 wheels or a verified public download
URL for this Torch build. Provision the same verified native stack separately
and retain the wheel hashes or container identity. Do not substitute an
unverified `pip install torch==2.13.0` recipe.

The workflow was exercised on GB10 on **2026-11-01**, in a detached checkout of
`4c282ce` plus the pending dual-target changes. Both model environments were
created from this native installation. Dependency/profile/build checks and all
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
