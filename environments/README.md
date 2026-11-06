# L20 and GB10 environments

Both targets use the same compiler, adapters, kernels and C++ runtime on `main`.
Each target has its own native stack, build, compilation cache and deployment
packages. Compile on the target host with its selected Torch installation.

This directory owns project setup and toolchain checks for both Python and C++.
Use `setup.sh` and `check.sh` as the entry points. Their Python implementations
inspect installed packages and Torch; they have no dependency on compiler helpers.
The scripts locate the repository root, so they can be called from any working
directory. All relative option paths are resolved from the repository root.

| Target | Python | Torch runtime | Triton | Torch CUDA | GPU capability |
|---|---|---|---|---|---|
| [l20](l20/README.md) | 3.12 | 2.13.0+cu126 | 3.7.1 | 12.6 | 8.9 |
| [gb10](gb10/README.md) | 3.12 | 2.13.0+cu130 | 3.7.1 | 13.0 | 12.1 |

The JSON profiles define the checked native versions. They describe tested
stacks, not downloadable wheel bundles. Keep the exact wheel/container identity
with deployment records. Qwen3 and Qwen3.5 have isolated HF environments because
they require Transformers 4.51.3 and 5.20.2 respectively. Their requirements files
contain model dependencies; the selected native stack supplies Torch/Triton/CUDA.
The Qwen3 Flash adapter rejects Transformers versions other than 4.51.3 before
loading its private HF implementation or modifying model modules. Both
benchmark eager baselines use ordinary HF dynamic-cache forwards; their reports
record `eager_baseline: hf_dynamic_cache`.

## Create model environments

Run the following workflow from the repository root. First provision the native
stack described by the chosen target, then set `NATIVE_PY` to its Python executable. The profile-specific
README gives the recorded installation on each existing host.

```bash
TARGET=l20                         # or gb10
NATIVE_PY=/absolute/path/to/native-stack/bin/python
CMAKE=cmake
# L20 uses versioned Python paths so the Torch 2.7 environments remain available.
ENV_SUFFIX=-venv
if [ "$TARGET" = l20 ]; then
  NATIVE_PY="$PWD/work/l20/native-torch213/bin/python"
  CMAKE=/data/workspace/abo/.venv/bin/cmake
  ENV_SUFFIX=-torch213-venv
fi
RUN=baseline-001                    # choose a fresh name for each export/test run
export PATH="$(dirname "$NATIVE_PY"):$PATH"  # host tools, including Ninja if installed there

for MODEL in qwen3 qwen35; do
  MODEL_ENV="$PWD/work/$TARGET/$MODEL$ENV_SUFFIX"
  bash environments/setup.sh \
    --target "$TARGET" --model "$MODEL" --native-python "$NATIVE_PY" \
    --output "$MODEL_ENV"
  "$MODEL_ENV/bin/python" -m pip install \
    -c "$MODEL_ENV/native-constraints.txt" \
    -r "compiler/$MODEL/requirements.txt"
  "$MODEL_ENV/bin/python" -m pip check
  bash environments/check.sh --python "$MODEL_ENV/bin/python" \
    --target "$TARGET" --model "$MODEL" \
    --output "artifacts/$TARGET/$MODEL/$RUN/environment.json"
done
```

`setup.sh` uses host `python3` to bootstrap the setup helper; `--native-python`
selects the installation used to probe the native stack and create each venv.
For `check.sh`, put `--python PATH` first to select the model environment to
inspect. Remaining options are forwarded to the checker unchanged.

The setup helper creates absent venv directories, links only the native packages
and their selected dependency closure, and records source provenance. It does
not expose the source environment's unrelated packages. Linked native packages
are read-only dependencies by convention; do not upgrade or uninstall them
through either model venv. The generated constraints keep pip from silently
replacing their recorded versions. Qwen3.5 also requires `fla` and
`causal_conv1d` to be absent, as checked by the environment command.

Environment checks collect the actual Python/platform, Torch runtime and
distribution/Git versions, Triton, CUDA, C++ ABI, GPU and LibTorch location.
Failures exit nonzero and are still written to the requested JSON report. A
profile match establishes the declared environment contract; keep the recorded
build provenance as well.

## Build native runtimes for each model

Both model venvs use the same selected LibTorch. Each model has its own native
build directory, `build/qwen3/` or `build/qwen35/`. CMake derives the C++ standard
from that Torch build. Use a fresh build directory when changing native stacks.
The host also needs CMake 3.24 or newer, Ninja, a C++ compiler and a compatible
CUDA toolkit. The setup helper does not install these executables.

```bash
Q3_PY="$PWD/work/$TARGET/qwen3$ENV_SUFFIX/bin/python"
Q35_PY="$PWD/work/$TARGET/qwen35$ENV_SUFFIX/bin/python"
Q3_BUILD="$PWD/build/qwen3"
Q35_BUILD="$PWD/build/qwen35"
TORCH_PREFIX=$("$Q3_PY" -c 'import torch; print(torch.utils.cmake_prefix_path)')
CUDA_ARCH=$("$Q3_PY" -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')
NVCC=/usr/local/cuda/bin/nvcc         # compatible toolkit used for this build
CUDA_RUNTIME_FLAGS=()
if [ "$TARGET" = l20 ]; then
  CUDA_RUNTIME=$("$Q3_PY" -c 'from pathlib import Path; import torch; p = Path(torch.__file__).resolve().parent.parent / "nvidia/cuda_runtime/lib/libcudart.so.12"; assert p.is_file(), p; print(p)')
  CUDA_RUNTIME_FLAGS=("-DCUDA_cudart_LIBRARY=$CUDA_RUNTIME" "-DCUDA_CUDART_LIBRARY=$CUDA_RUNTIME")
fi

for MODEL in qwen3 qwen35; do
  BUILD="$PWD/build/$MODEL"
  "$CMAKE" -S . -B "$BUILD" -G Ninja \
    -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" -DCMAKE_CUDA_COMPILER="$NVCC" \
    "${CUDA_RUNTIME_FLAGS[@]}" \
    -DTORCH_CUDA_ARCH_LIST="$CUDA_ARCH" -DCMAKE_BUILD_TYPE=Release \
    -DXINFER_BUILD_VALIDATION=ON -DXINFER_BUILD_BENCHMARK=ON
  "$CMAKE" --build "$BUILD" --target \
    xinfer_aoti xinfer_validate xinfer_benchmark qwen35_fused qwen_runtime_contract -j2
  bash environments/check.sh --python "$PWD/work/$TARGET/$MODEL$ENV_SUFFIX/bin/python" \
    --target "$TARGET" --model "$MODEL" --build "$BUILD" \
    --output "artifacts/$TARGET/$MODEL/$RUN/build-environment.json"
done
```

The build produces `xinfer_aoti`, `xinfer_benchmark` and `xinfer_validate` for both
model families. Model selection comes from the package metadata; there is no
separate full-model build switch. `qwen35_fused` remains a focused GDN validation
tool, built with `XINFER_BUILD_VALIDATION`.
Each model's Python tools default to executables and the attention library in
its `build/<model>/` directory. Existing native-path options and
`XINFER_ATTENTION_OPS_LIBRARY` optionally select another build or installed runtime.

`--build` verifies that CMake selected the active environment's LibTorch, resolving
symlinks to the shared native installation. Build before either full-model
export so Python can register the shared native FlashAttention operator. Deploy
`libxinfer_attention_ops.so` with the shared full-model executables.
Qwen3.5 packages its GDN kernels and needs no separate GDN custom-op library;
its standalone GDN validation targets do not use the attention library.

L20 keeps its host CUDA 12.2 compiler and explicitly links the CUDA 12.6 runtime
from the selected wheel installation. The two CMake variables select that
runtime for the Torch and CUDA discovery paths. Linking Torch 2.13 against the host's
older `libcudart` leaves unresolved runtime symbols. GB10 uses its existing
CUDA runtime selection; the extra argument array is empty there.

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
export logs. The deployment sidecars use metadata schema 3 with
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

## Install and relocate the runtime

After building, install the project's runtime component into a fresh prefix.
Either model's build contains the shared executables supporting both models;
the example below installs from `build/qwen3/`.
GNUInstallDirs selects executable and library directories, typically `bin/`
and `lib/` or `lib64/`. `xinfer_aoti` is always included;
`xinfer_benchmark` and `xinfer_validate` are included when their build options
are enabled. Focused helpers `qwen35_fused` and `qwen_runtime_contract` remain
build-only targets. `CMAKE_INSTALL_BINDIR` and `CMAKE_INSTALL_LIBDIR` can customize
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
  "$Q3_ROOT/install-prompt.ids" "$Q3_ROOT/install-generated.ids" --max-new-tokens 8

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

## Validation matrix

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
"$Q35_PY" -m compiler.qwen35.tests.validate_cond \
  --binary "$Q35_BUILD/qwen35_fused" --output "$Q35_ROOT/cond-validation"

# Generate independent HF fixtures once for this run, or supply verified saved fixtures.
"$Q3_PY" -m compiler.qwen3.tests.validate \
  --model-dir "$Q3_MODEL" --output "$Q3_ROOT/reference"
AOTI_RUNTIME_CHECK_INPUTS=1 "$Q3_PY" -m compiler.qwen3.tests.validate_cpp \
  --model-dir "$Q3_MODEL" --packages "$Q3_PACKAGES" --reference "$Q3_ROOT/reference" \
  --native "$Q3_BUILD/xinfer_validate" --output "$Q3_ROOT/runtime-validation"
"$Q3_PY" -m compiler.qwen3.tests.smoke_cpp \
  --packages "$Q3_PACKAGES" --reference "$Q3_ROOT/reference" \
  --native "$Q3_BUILD/xinfer_aoti" --output "$Q3_ROOT/native-smoke"

"$Q35_PY" -m compiler.qwen35.tests.validate_model \
  --model-dir "$Q35_MODEL" --packages "$Q35_PACKAGES" --output "$Q35_ROOT/model-validation"
"$Q35_PY" -m compiler.qwen35.tests.validate_runtime \
  --model-dir "$Q35_MODEL" --packages "$Q35_PACKAGES" \
  --binary "$Q35_BUILD/xinfer_validate" --cli "$Q35_BUILD/xinfer_aoti" \
  --output "$Q35_ROOT/runtime-validation"
"$Q35_PY" -m compiler.qwen35.tests.validate_end_to_end \
  --model-dir "$Q35_MODEL" --packages "$Q35_PACKAGES" \
  --native "$Q35_BUILD/xinfer_validate" --output "$Q35_ROOT/end-to-end"
```

The standalone GDN suite uses synthetic inputs and adds a checkpoint case when
its optional block fixture exists at `work/qwen35/0.8b/block`; select another
verified fixture with `--block-dir`. Its report states whether that case ran.

Inspect every `summary.json`, including after a nonzero validator exit. Require
finite outputs, legal dynamic boundaries, correct state carry/reset/masking and
bitwise native/Python AOTI agreement within each target. Qwen3 retains its bounded
HF logit gates and separate cache/token diagnostics. Qwen3.5's HF numerical
acceptance remains open; GDN `dispatch_passed` and runtime
`native_runtime_passed` do not replace the full numerical report. The matrix
does not require cross-target bitwise equality or waive existing thresholds.

The target guides record the model validation runs. The Bash entry-point
refactor was also checked on both L20 and GB10: 21 tooling tests, fresh setup
for both model environments, dependency checks, LibTorch build matching and
failure-status checks passed from outside the repository. These reports are in
`artifacts/<target>/environment-refactor-20261101/` in each validation checkout;
this refactor did not repeat model compilation. Retain environment reports,
package/native identities and validation summaries for each run.
