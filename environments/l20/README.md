# L20 environment

For the offline Docker testing environment, use the
[Docker build and launch guide](docker/README.md). For a non-Docker Linux x86-64
host, use this device's Bash setup script:

```bash
bash environments/l20/setup.sh --model qwen3
bash environments/l20/setup.sh --model qwen35

Q3_PY="$PWD/work/l20/qwen3-torch213-venv/bin/python"
Q35_PY="$PWD/work/l20/qwen35-torch213-venv/bin/python"
Q3_BUILD="$PWD/build/qwen3"
Q35_BUILD="$PWD/build/qwen35"
```

The host needs Python 3.12, glibc 2.28 or newer, a C++ compiler, CUDA toolkit and
NVIDIA driver. Each invocation creates a fresh independent venv and installs
Torch `2.13.0+cu126` from the [official CUDA 12.6 index](https://download.pytorch.org/whl/cu126/torch/),
Triton `3.7.1`, the model's pinned dependencies, CMake `4.3.4` and Ninja `1.13.2`.
Setup and checking use [`qwen3_requirements.txt`](qwen3_requirements.txt) or
[`qwen35_requirements.txt`](qwen35_requirements.txt). Each file contains the full
PyTorch index, Torch, Triton, CMake, Ninja and HF requirements for that model.
The model files can select different Torch versions.
The checker derives the required CUDA runtime `12.6` from the Torch pin.
`profile.json` checks Linux, x86-64 and GPU capability `8.9`.

Use `--python /path/to/python3.12` to select the bootstrap interpreter and
`--venv PATH` for a different fresh venv directory. `--torch-wheel PATH` selects
a local Torch wheel. Existing destinations are refused. The setup script does
not reuse system packages or an existing native environment. The older
`work/l20/qwen3-venv`, `work/l20/qwen35-venv` and other retained installations
remain available for historical comparisons.

Setup does not need a GPU. Follow the [shared workflow](../README.md) to run
`check.sh --python PATH`, build with the model environment's CMake and LibTorch,
and validate Triton/AOTI execution before using new packages. Native builds use
`build/qwen3/` and `build/qwen35/`.

The checker compares `torch.version.cuda` to the CUDA build selected by the
Torch requirement and records the local CUDA compiler separately; its minor
version can differ. The earlier
native host used nvcc `12.2.140` and driver `535.161.07`. The shared build command
sets both `CUDA_cudart_LIBRARY` and `CUDA_CUDART_LIBRARY` to the model venv's
`nvidia/cuda_runtime/lib/libcudart.so.12`, linking the runtime required by
LibTorch rather than an older host runtime.

All L20 package requirements live in this directory; the Bash scripts contain
no package pins.

## Validation history

The original full-model validation remains recorded under
[`docs/experiments/`](../../docs/experiments/). Re-run the shared validation matrix
for new packages. Qwen3.5 HF numerical acceptance remains open.

The previous Torch 2.7 workflow was exercised on 2026-11-01: both model venvs were created
from the native stack and passed pip/profile/LibTorch-build checks. All seven
native targets and the attention library built. Qwen3's nine operator and nine
compiled-attention cases passed; Qwen3.5's GDN suite passed dispatch parity for
101 cases / 566 native calls / 1,698 tensors. Its 364 existing numerical failure
records matched the previous baseline exactly. Fifteen environment/setup tests
and the CPU cache regression also passed. Reports are in
`artifacts/l20/<model>/dual-target-20261101/`; this run did not repeat full-model
export or performance measurements.

After the GB10 fixture-layout correction, the standalone GDN suite was rerun
with AOTI input guards enabled on L20. All 101 cases / 566 native calls / 1,698
tensors still passed dispatch parity, with the same 364 numerical failure
records. That report is in the run's `cond-guarded/` directory.

Torch 2.13 exposes `load_constants(..., user_managed=True)` for borrowing
externally owned constant tensors. This requires freshly exported weightless
packages (`aot_inductor.package_constants_in_so=False`) and an owner that keeps
the shared GPU tensors alive for both loaders. Both current Qwen exporters and
runtimes now use shared-weight format 2, with one archive and a retained shared
pool. Mappings are embedded in `weights.pt`; its checksum is recorded in both
package metadata files. No separate weight manifest is required.
Rebuild and re-export existing embedded-weight packages, including those built
with Torch 2.13. Keep their old binaries for rollback. The initial environment
canary below is separate from the full-model sharing validation recorded in
[`docs/experiments/shared-weights.md`](../../docs/experiments/shared-weights.md).

On **2026-11-02**, the Torch 2.13 stack passed 10 direct Triton GPU cases and
two Inductor cases. Python AOTI and native C++ each loaded two freshly compiled
weightless packages with one shared 32 MiB GPU tensor. Both loaders together
added zero measured CUDA memory, and both observed an in-place tensor mutation
without reinjection. The native probe also checked the loaders' extracted
constant pointers against the shared tensor's actual storage pointer. Reports
are in `artifacts/l20/torch213-upgrade-20261102/`.

The Python probe calls `load_constants(constants, check_full_update=True,
user_managed=True)`; its C++ companion calls
`load_constants(constants, false, true, true)`. Reproduction sources are retained
under the run's `probes/` directory. The probe also returns a weight tensor as a
diagnostic output, which AOTI clones; that output pointer is not the loader's
internal constant pointer. The native test checks the latter directly.

Both versioned model environments passed dependency, profile and LibTorch-build
checks. All seven native executables and the attention library built against
the new installation. The 22 tooling/CPU-cache unit tests passed, as did Qwen3's
nine attention operator and nine compiled-attention cases. Qwen3.5 GDN passed
dispatch and JIT/Python AOTI/native bitwise parity for 101 cases, 566 native
calls and 1,698 tensors, including strict input guards. Its full numerical
validator still exits nonzero: the same 364 comparison/case/step/tensor failures
remain as in the Torch 2.7 baseline. No numerical limits were changed. These
checks and canaries do not establish full-model numerical or performance
results on Torch 2.13. The consolidated result is `validation-summary.json`
in the run's artifact directory.

Artifact cleanup on the same date removed 344 obsolete generated packages and
state dumps, reclaiming 87.2 GiB (116.8 GiB before cleanup, 29.6 GiB immediately
after). Current package baselines, references, reports, source/graph snapshots
and checkpoints were retained. The exact deletion inventory is
`artifacts/maintenance/artifact-cleanup-20261102.json`. The new upgrade
validation artifacts were generated after that measurement.
