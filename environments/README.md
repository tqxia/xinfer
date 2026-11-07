# L20 and GB10 environments

Both targets use the same compiler, adapters, kernels and C++ runtime on `main`.
Each model has an independent Python environment, native build, compilation
cache and deployment packages. Compile on the target host with the Python
installation selected for that model.

This directory owns device-specific Bash setup scripts and shared environment
checks. Use `l20/setup.sh` or `gb10/setup.sh` for a non-Docker environment, then
`check.sh` before build or export. The scripts locate the repository root, so
they can be called from any working directory. Relative option paths are
resolved from the repository root. The [L20 Docker guide](l20/docker/README.md)
provides a separate offline image workflow. Its complete `native.lock`,
`qwen3.lock` and `qwen35.lock` files are resolved snapshots for the offline
image, including image-specific tools. Refresh those snapshots and the wheel
payload from the device/model requirement inputs when changing their pins;
setup does not regenerate the Docker locks.

| Target | Python | Torch runtime | Triton | Torch CUDA | GPU capability |
|---|---|---|---|---|---|
| [l20](l20/README.md) | 3.12 | 2.13.0+cu126 | 3.7.1 | 12.6 | 8.9 |
| [gb10](gb10/README.md) | 3.12 | 2.13.0+cu130 | 3.7.1 | 13.0 | 12.1 |

Both setup and environment checking read the selected model/device file:
`environments/<target>/<model>_requirements.txt`. Each file is self-contained:
it specifies the official PyTorch package index and all Torch, Triton,
CMake, Ninja and HF pins. The Torch version can vary independently for each
model/device. All package requirements live under `environments/`.
Qwen3 and Qwen3.5 use separate venvs because they require Transformers 4.51.3 and
5.20.2 respectively. The JSON profiles contain only operating system, CPU
architecture and GPU capability. Python 3.12 is the setup prerequisite and
default interpreter. Keep the actual environment
report with deployment records.
The Qwen3 Flash adapter rejects Transformers versions other than 4.51.3 before
loading its private HF implementation or modifying model modules. Both
benchmark eager baselines use ordinary HF dynamic-cache forwards; their reports
record `eager_baseline: hf_dynamic_cache`.

## Create model environments

The host needs Python 3.12, Linux with glibc 2.28 or newer, a C++ compiler, a
compatible CUDA toolkit and NVIDIA driver. Setup installs Torch, Triton, model
dependencies, CMake 4.3.4 and Ninja 1.13.2 into a fresh venv. It does not require
a visible GPU. GPU and build compatibility are checked separately.

```bash
TARGET=l20                         # or gb10
ENV_SUFFIX=-venv
if [ "$TARGET" = l20 ]; then
  ENV_SUFFIX=-torch213-venv
fi
RUN=baseline-001                    # choose a fresh name for each export/test run

for MODEL in qwen3 qwen35; do
  MODEL_ENV="$PWD/work/$TARGET/$MODEL$ENV_SUFFIX"
  bash "environments/$TARGET/setup.sh" --model "$MODEL" --venv "$MODEL_ENV"
  bash environments/check.sh --python "$MODEL_ENV/bin/python" \
    --target "$TARGET" --model "$MODEL" \
    --output "artifacts/$TARGET/$MODEL/$RUN/environment.json"
done
```

Both setup scripts accept `--model qwen3|qwen35`, `--python PATH` to select a
Python 3.12 executable, `--venv PATH`, and `--torch-wheel PATH` to install a
local Torch wheel instead of downloading the pinned wheel from the official
PyTorch index. A local wheel replaces only the Torch requirement; the remaining
requirements in the selected file still apply. The default Python executable is
`python3.12`. Default venvs are
`work/l20/<model>-torch213-venv` and `work/gb10/<model>-venv`.

Setup requires a new venv directory and refuses to overwrite an existing
venv. Each new environment gets its own installed Torch and dependencies;
packages from an existing Python installation are not linked or inherited.
The script installs the selected requirements file in one pip invocation,
followed by `pip check`.
Qwen3.5 also requires `fla` and `causal_conv1d` to be absent, as checked by the
environment command.

For `check.sh`, put `--python PATH` first to select the model environment to
inspect. Remaining options are forwarded to the checker unchanged. Checks
collect the actual Python/platform, Torch runtime and distribution/Git versions,
Triton, CUDA, C++ ABI, GPU and LibTorch location. They validate the installed
native packages, including CMake and Ninja, and the HF packages against the
same requirements file used by setup. Required Torch runtime and CUDA build
versions come from that file's Torch pin. Reports include the `requirements`
map separately from the hardware profile. Failures exit nonzero and are still
written to the requested JSON report. Keep the recorded build provenance as well.

## Build native runtimes for each model

Each model's native build uses LibTorch from that model's Python environment.
Build directories are `build/qwen3/` and `build/qwen35/`. CMake derives the C++
standard from the selected Torch build. Use a fresh build directory when
changing Torch installations. CMake and Ninja are installed in each model
venv's `bin/`; the C++ compiler and CUDA toolkit remain host prerequisites.

```bash
Q3_PY="$PWD/work/$TARGET/qwen3$ENV_SUFFIX/bin/python"
Q35_PY="$PWD/work/$TARGET/qwen35$ENV_SUFFIX/bin/python"
Q3_BUILD="$PWD/build/qwen3"
Q35_BUILD="$PWD/build/qwen35"
NVCC=/usr/local/cuda/bin/nvcc         # compatible toolkit used for this build

for MODEL in qwen3 qwen35; do
  MODEL_ENV="$PWD/work/$TARGET/$MODEL$ENV_SUFFIX"
  MODEL_PY="$MODEL_ENV/bin/python"
  CMAKE="$MODEL_ENV/bin/cmake"
  BUILD="$PWD/build/$MODEL"
  TORCH_PREFIX=$("$MODEL_PY" -c 'import torch; print(torch.utils.cmake_prefix_path)')
  CUDA_ARCH=$("$MODEL_PY" -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')
  CUDA_RUNTIME_FLAGS=()
  if [ "$TARGET" = l20 ]; then
    CUDA_RUNTIME=$("$MODEL_PY" -c 'from pathlib import Path; import torch; p = Path(torch.__file__).resolve().parent.parent / "nvidia/cuda_runtime/lib/libcudart.so.12"; assert p.is_file(), p; print(p)')
    CUDA_RUNTIME_FLAGS=("-DCUDA_cudart_LIBRARY=$CUDA_RUNTIME" "-DCUDA_CUDART_LIBRARY=$CUDA_RUNTIME")
  fi
  PATH="$MODEL_ENV/bin:$PATH" "$CMAKE" -S . -B "$BUILD" -G Ninja \
    -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" -DCMAKE_CUDA_COMPILER="$NVCC" \
    "${CUDA_RUNTIME_FLAGS[@]}" \
    -DTORCH_CUDA_ARCH_LIST="$CUDA_ARCH" -DCMAKE_BUILD_TYPE=Release \
    -DXINFER_BUILD_VALIDATION=ON -DXINFER_BUILD_BENCHMARK=ON
  PATH="$MODEL_ENV/bin:$PATH" "$CMAKE" --build "$BUILD" --target \
    xinfer_aoti xinfer_validate xinfer_benchmark qwen_runtime_contract \
    xinfer_state_contract xinfer_generation_contract -j2
  bash environments/check.sh --python "$MODEL_PY" \
    --target "$TARGET" --model "$MODEL" --build "$BUILD" \
    --output "artifacts/$TARGET/$MODEL/$RUN/build-environment.json"
done
```

The build produces `xinfer_aoti`, `xinfer_benchmark` and `xinfer_validate` for both
model families. Model selection comes from the package metadata; there is no
separate full-model build switch. Kernel export/AOTI checks run through
`compiler.qwen35.tests.test_kernels_aoti`; they need no native kernel runner.
Each model's Python tools default to executables and the attention library in
its `build/<model>/` directory. Existing native-path options and
`XINFER_ATTENTION_OPS_LIBRARY` optionally select another build or installed runtime.

`--build` verifies that CMake selected the active environment's LibTorch. Build
before either full-model export so Python can register the shared native
FlashAttention operator. Deploy `libxinfer_attention_ops.so` with the shared
full-model executables. Qwen3.5 packages its GDN kernels and needs no separate
GDN custom-op library.

For L20, the two CMake runtime variables select CUDA 12.6 from the model
installation for both Torch and CUDA discovery. This also supports a host
compiler with an older compatible CUDA minor version; linking Torch 2.13 to an
older host `libcudart` can leave unresolved runtime symbols. GB10 uses its CUDA
13.0 runtime selection; the extra argument array is empty there.

## Prepare inputs and export

Set explicit checkpoint paths on each host. Both model families use
`model_manifest.json`, defined once by `MODEL_MANIFEST_NAME` in
`compiler/common/artifacts.py`. Copying a complete verified checkpoint directory
between targets is supported. The shared `tools.prepare` command downloads the
pinned checkpoint and creates this manifest. Qwen3 keeps the original HF files;
Qwen3.5 extracts text weights from its multimodal source. Preparation uses CPU
and the matching model environment, with no CUDA or inference required.

For retained Qwen3.5 text checkpoints, rename `manifest.json` to
`model_manifest.json`. The JSON contents stay unchanged. Full-model package
metadata embeds the text checkpoint's manifest contents, so this filename
migration requires no metadata conversion or recompilation of existing packages.

Qwen3 tools default to `work/models/qwen3-0.6b` when run from the repository
root. Place the complete verified directory there or symlink an existing
checkpoint; `--model-dir` overrides this location for each command.

```bash
Q3_MODEL="$PWD/work/models/qwen3-0.6b"
Q35_MODEL="$PWD/work/models/qwen35-0.8b-text"
Q3_ROOT="$PWD/artifacts/$TARGET/qwen3/$RUN"
Q35_ROOT="$PWD/artifacts/$TARGET/qwen35/$RUN"
Q3_PACKAGES="$Q3_ROOT/packages"
Q35_PACKAGES="$Q35_ROOT/packages"
```

For first setup, download before enabling offline mode. Skip these commands when
reusing complete verified directories copied from another target. Download
refuses an existing output path and publishes the directory only after success.
`--cache-dir` selects a Hub cache; otherwise the Hub's configured cache is used.

```bash
"$Q3_PY" -m tools.prepare download --model Qwen/Qwen3-0.6B --output "$Q3_MODEL"
"$Q35_PY" -m tools.prepare download --model Qwen/Qwen3.5-0.8B --output "$Q35_MODEL"
```

For copied or retained directories, request an explicit integrity check as
needed. Successful download already verifies the checkpoint; model loading and
export do not repeat checkpoint manifest or checksum verification:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_COMPILE_THREADS=2
export AOTI_RUNTIME_CHECK_INPUTS=1
# Optional explicit verification for copied or retained checkpoints.
"$Q3_PY" -m tools.prepare verify --model-dir "$Q3_MODEL"
"$Q35_PY" -m tools.prepare verify --model-dir "$Q35_MODEL"

export TORCHINDUCTOR_CACHE_DIR="$PWD/work/$TARGET/cache/$RUN/qwen3"
"$Q3_PY" -m compiler.qwen3.export --model-dir "$Q3_MODEL" --output "$Q3_PACKAGES"
export TORCHINDUCTOR_CACHE_DIR="$PWD/work/$TARGET/cache/$RUN/qwen35"
"$Q35_PY" -m compiler.qwen35.export --model-dir "$Q35_MODEL" --output "$Q35_PACKAGES"
```

Serialize compilation and GPU validation. Each export invocation compiles both
entries sequentially and requires a new or empty output directory.
The default numerical settings and algorithms are shared across targets; this
workflow leaves optional Inductor tuning off. Package pairs must have matching
metadata, numerical settings and checkpoint identity. Re-export and rebuild for each
native stack/GPU; keep all existing runtime compatibility checks enabled.
Both model exporters write `prefill.pt2`, `decode.pt2`, their two metadata JSON
files, and one shared `weights.pt`. Deploy all five files together. The archive
embeds the exact constant mappings and tensor views in `extra/weights.json`;
there is no separate weight manifest. Both entry metadata files hold the archive
size and SHA-256 checksum. Each export writes the archive once, records its
checksum in both metadata files, and publishes the complete directory after
both entries succeed. One entry alone is an incomplete bundle. Dependency
versions, full compiler options and graph/timing diagnostics are written to
export logs. The deployment sidecars use metadata schema 5 with
`runtime_abi_version: 1` and retain the state, compatibility, integrity, checkpoint
and numerical contract. The runtime ABI defines the fixed batch/dtype, Flash
attention, cache/extent representation and model input/output rules. Current
Python/native runtimes require shared-weight format 2
and Torch 2.13 or newer; re-export embedded-weight and earlier shared bundles.
Keep old Torch 2.7 package pairs with their original runtime. Rebuild and
re-export for Torch 2.13; a version match does not establish compatibility with
older generated code. The old 2.7 loader extracts Qwen3.5's libraries into `/tmp`
regardless of `TMPDIR`, so retain about 3 GB plus margin there for rollback runs,
in addition to compile-cache space.

New exports default to the model's declared context ceiling: 40,960 slots for
Qwen3-0.6B and 262,144 for both Qwen3.5 checkpoints. `--max-context-length` chooses
an explicit capacity within that ceiling before GPU state loading.
Their compiled prefill calls remain bounded to 2–512 tokens, with one-token
decode. The shared runner chunks long prompts automatically and validates the
complete prompt plus generation budget before execution. Schema 5 records
per-call `token_range` and `cache_length`; no separate prompt-cap option exists.
The native request uses `--max-output-tokens`. Existing 1,024-slot bundles require a separate
schema 5 sidecar conversion, retaining compiled bounds, or a fresh export for
larger capacity. Any additional input policy belongs to the calling application. See the
[metadata migration contract](../docs/package-metadata.md). Ordinary long-context
exports, C++ execution and full 16,384-slot inference-memory probes completed on
L20 for Qwen3-0.6B, Qwen3.5-0.8B and Qwen3.5-4B. Native/Python AOTI parity
passed for all three; Qwen3 passed its existing HF numerical gates. Qwen3.5 HF
acceptance remains open. The new profile has not been validated on GB10.

## Install and relocate the runtime

After building, install the project's runtime component into a fresh prefix.
Either model's build contains the shared executables supporting both models;
the example below installs from `build/qwen3/`.
GNUInstallDirs selects executable and library directories, typically `bin/`
and `lib/` or `lib64/`. `xinfer_aoti` is always included;
`xinfer_benchmark` and `xinfer_validate` are included when their build options
are enabled. The focused `qwen_runtime_contract` helper remains a build-only
target. `CMAKE_INSTALL_BINDIR` and `CMAKE_INSTALL_LIBDIR` can customize
the layout and must be relative paths. The following commands read the configured
library directory from the build cache and use the default `bin/` directory.

```bash
BUILD="$Q3_BUILD"
PREFIX="$PWD/work/$TARGET/runtime-install"
MOVED_PREFIX="$PWD/work/$TARGET/runtime-relocated"
INSTALL_LIBDIR=$(sed -n 's/^CMAKE_INSTALL_LIBDIR:[^=]*=//p' "$BUILD/CMakeCache.txt")
"$CMAKE" --install "$BUILD" --prefix "$PREFIX" --component Runtime
mv "$PREFIX" "$MOVED_PREFIX"
"$MOVED_PREFIX/bin/xinfer_aoti" --help

"$Q3_PY" -m tools.prepare encode --model-dir "$Q3_MODEL" \
  --prompt "What is the capital of France?" --output "$Q3_ROOT/install-prompt.ids"
"$MOVED_PREFIX/bin/xinfer_aoti" "$Q3_PACKAGES" \
  "$Q3_ROOT/install-prompt.ids" "$Q3_ROOT/install-generated.ids" --max-output-tokens 8

# Python export/validation can use the installed attention library too.
export XINFER_ATTENTION_OPS_LIBRARY="$MOVED_PREFIX/$INSTALL_LIBDIR/libxinfer_attention_ops.so"
```

Installed executables find the attention library relative to `$ORIGIN`, so the
prefix can move on the same compatible native stack. The selected LibTorch and
CUDA libraries remain external; the installed RPATH retains their configured
runtime paths. `XINFER_INSTALL_RUNTIME_DIRS` can override these paths at configure
time with a semicolon-separated list; its defaults are the selected Torch library
and CUDA runtime directories. CMake also retains external linked-library
directories. A different host must provide a compatible GPU/package contract and
the matching native stack at those library paths; relocating that external
stack is a separate step. Deploy each complete five-file model package directory
alongside the installed runtime.

## Benchmark tools

Use the same metadata-selected command with each model's Python environment:

```bash
"$Q3_PY" -m tools.benchmark run \
  --model-dir "$Q3_MODEL" --packages "$Q3_PACKAGES" \
  --native "$Q3_BUILD/xinfer_benchmark" --output "$Q3_ROOT/benchmark"
"$Q35_PY" -m tools.benchmark run \
  --model-dir "$Q35_MODEL" --packages "$Q35_PACKAGES" \
  --native "$Q35_BUILD/xinfer_benchmark" --output "$Q35_ROOT/benchmark"
"$Q35_PY" -m tools.benchmark check-idle --output "$Q35_ROOT/idle.json"
```

Both families also support `compare`, using `--baseline-packages`,
`--candidate-packages` and `--output`. Native prefill timing covers all prompt
chunks; the ordinary HF benchmark processes the whole prompt in one call, and
reports record that partition difference. Numerical validation uses matched
partitions instead. Use a fresh output path for each invocation.
GPU selection,
fixed-token timing, defaults and reports are shared. `run` reports `hf` and
`cpp_aoti` profiles using the plain HF dynamic-cache baseline. Workload limits
come from package metadata. Benchmarks prepare their own tokens and do not
consume numerical validation reports. See the
[benchmark protocol](../docs/development-tools.md#benchmark-hf-and-native-aoti)
for all commands and measurement boundaries.

## Validation matrix

Run the [CPU tooling unit tests](../docs/development-tools.md#cpu-unit-tests)
with the matching model environments before GPU validation. Preparation,
benchmark and validation tests live under their owning `tools/<area>/tests/`;
compiler/model/operator tests remain under `compiler/*/tests/`.

Run the following on both targets for each new package pair. Use a GPU-idle
window for performance measurements. Functional checks can establish correctness
without making a timing claim.

| Model | L20 | GB10 |
|---|---|---|
| Qwen3-0.6B | Native/Python AOTI parity, HF logit limits, state/token diagnostics | Same gates |
| Qwen3.5-0.8B | CPU cache regression, GDN dispatch, native parity, HF/FP32 report | Same gates |

```bash
CUDA_VISIBLE_DEVICES='' "$Q35_PY" -m unittest \
  environments.tests.test_check environments.tests.test_setup \
  environments.tests.test_entrypoints compiler.qwen35.tests.test_text_cache \
  compiler.common.tests.test_weights
"$Q35_PY" -m unittest compiler.qwen35.tests.test_delta compiler.qwen35.tests.test_kernels
"$Q35_PY" -m unittest compiler.qwen35.tests.test_kernels_aoti

AOTI_RUNTIME_CHECK_INPUTS=1 "$Q3_PY" -m tools.validation.validate \
  --model-dir "$Q3_MODEL" --packages "$Q3_PACKAGES" \
  --native "$Q3_BUILD/xinfer_validate" --output "$Q3_ROOT/runtime-validation"
"$Q3_PY" -m tools.validation.qwen3.validate_runtime \
  --model-dir "$Q3_MODEL" --packages "$Q3_PACKAGES" \
  --native "$Q3_BUILD/xinfer_aoti" --output "$Q3_ROOT/native-smoke"

"$Q35_PY" -m tools.validation.validate \
  --model-dir "$Q35_MODEL" --packages "$Q35_PACKAGES" \
  --native "$Q35_BUILD/xinfer_validate" --output "$Q35_ROOT/validation"
"$Q35_PY" -m tools.validation.qwen35.validate_runtime \
  --model-dir "$Q35_MODEL" --packages "$Q35_PACKAGES" \
  --binary "$Q35_BUILD/xinfer_validate" --cli "$Q35_BUILD/xinfer_aoti" \
  --output "$Q35_ROOT/runtime-validation"
```

The common numerical command reads the model family from package metadata and
runs HF → eager → Python AOTI → C++ AOTI on matched inputs. Both models generate
HF references during that run, with no saved-reference prerequisite. Qwen3 also
cross-checks manual HF greedy decoding against HF `generate()`. Eager includes
adapters and package-selected linear packing. The runtime commands check CLI
and deployment contracts separately. The kernel unit and AOTI tests use synthetic inputs without a checkpoint.
The AOTI module compiles BF16/FP32 prefill/decode packages for both supported
geometries in temporary directories that are cleaned up automatically.

Inspect every `summary.json`, including after a nonzero validator exit. Require
finite outputs, legal dynamic boundaries, correct state carry/reset/masking and
bitwise native/Python AOTI agreement within each target. Qwen3 retains its bounded
HF logit gates and separate cache/token diagnostics. Qwen3.5's HF numerical
acceptance remains open; passing kernel/AOTI tests and runtime
`native_runtime_passed` do not replace the full numerical report. The matrix
does not require cross-target bitwise equality or waive existing thresholds.

The target guides record the model validation runs. The earlier shared Bash
setup/check entry points were checked on both L20 and GB10: 21 tooling tests, fresh setup
for both model environments, dependency checks, LibTorch build matching and
failure-status checks passed from outside the repository. These reports are in
`artifacts/<target>/environment-refactor-20261101/` in each validation checkout;
this refactor did not repeat model compilation. Retain environment reports,
package/native identities and validation summaries for each run.
