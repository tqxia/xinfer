# L20 native stack

For the offline Docker testing environment, use the
[Docker build and launch guide](docker/README.md).

The profile selects the Torch 2.13 CUDA 12.6 stack on the Linux x86-64 host
`kikoxia-any4.devcloud.woa.com`, SSH port `36000`:

| Component | Recorded value |
|---|---|
| Python | 3.12 |
| Torch runtime | 2.13.0+cu126 |
| Triton | 3.7.1 |
| CUDA used by Torch | 12.6 |
| Local nvcc | 12.2.140 |
| NVIDIA driver | 535.161.07 |
| GPU compute capability | 8.9 |

The selected native Python is
`/data/workspace/develop/qwen3-aoti-poc/work/l20/native-torch213/bin/python`.
Use it as `NATIVE_PY` in the [shared workflow](../README.md). The model venvs use
explicit versioned paths; native builds use each model's default directory:

```bash
NATIVE_PY="$PWD/work/l20/native-torch213/bin/python"
Q3_PY="$PWD/work/l20/qwen3-torch213-venv/bin/python"
Q35_PY="$PWD/work/l20/qwen35-torch213-venv/bin/python"
Q3_BUILD="$PWD/build/qwen3"
Q35_BUILD="$PWD/build/qwen35"
CMAKE=/data/workspace/abo/.venv/bin/cmake
```

The host has no `cmake` command on its default `PATH`; the shared workflow
selects the existing CMake executable above. Ninja is `/usr/local/bin/ninja`.
Using these tools does not change the preserved ABO Python environment.

The setup helper reads selected native packages and does not modify the source
environment. Use `setup.sh --output PATH` for these fresh model venvs, followed
by `check.sh --python PATH`. The default setup paths `work/l20/qwen3-venv` and
`work/l20/qwen35-venv`, the old `work/l20/build`, and
`/data/workspace/abo/.venv` retain Torch 2.7 for rollback. They are not selected
by the current profile; do not overwrite or rename them.
The earlier Torch 2.13 build at `work/l20/build-torch213` is also retained;
new builds use `build/qwen3/` and `build/qwen35/`.

The separate native installation was provisioned with the following command
from the repository root. Run it only when the destination does not exist;
`venv` does not protect an existing environment from modification.

```bash
/data/workspace/abo/.venv/bin/python -m venv work/l20/native-torch213
mkdir -p work/l20/tmp-torch213
env TMPDIR="$PWD/work/l20/tmp-torch213" PIP_NO_CACHE_DIR=1 \
  work/l20/native-torch213/bin/python -m pip install \
  --disable-pip-version-check --progress-bar off \
  torch==2.13.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

Keep the native installation's package freeze and environment reports with its
build. The CUDA driver and local compiler remain host components; validate
actual Triton and AOTI GPU execution after an installation change.

The profile's CUDA version is `torch.version.cuda`. The environment report
records the local CUDA compiler separately; its minor version can differ, as
in this existing stack. Retain the compiler and CMake build reports with each
deployment. For Torch 2.13, the shared build command sets both
`CUDA_cudart_LIBRARY` and `CUDA_CUDART_LIBRARY` to the wheel installation's
`nvidia/cuda_runtime/lib/libcudart.so.12`. This keeps host nvcc 12.2 while linking
the CUDA 12.6 runtime required by LibTorch; the host's older runtime produces
undefined references when linking the new build.

[`qwen3.lock`](qwen3.lock) preserves the complete original L20 Qwen3 freeze for
provenance. It includes model and native dependencies and is not a cross-target
installer. New model environments use the selected native stack plus their
model's `compiler/<model>/requirements.txt`.

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
