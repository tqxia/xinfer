# L20 and GB10 environments

This guide owns native setup, environment checks, builds and installation.
Use the model guides for [Qwen3](../compiler/qwen3/README.md) and
[Qwen3.5](../compiler/qwen35/README.md) export details, and
[development tools](../docs/development-tools.md) for preparation, validation and
benchmarks. [Validation status](../docs/validation-status.md) records what has
actually been tested; a supported environment is not a blanket model-acceptance claim.

## Requirements and targets

Each model has an independent Python environment, native build and compiled
packages. Compile on the target host against that model's selected LibTorch.
The host needs Python 3.12, Linux with glibc 2.28 or newer, a C++ compiler,
a compatible CUDA toolkit and NVIDIA driver. The export linker path uses Linux GCC.

| Target | CPU architecture | Torch runtime | Torch CUDA | GPU capability |
| --- | --- | --- | --- | --- |
| L20 | x86-64 | 2.13.0+cu126 | 12.6 | 8.9 |
| GB10 | ARM64 | 2.13.0+cu130 | 13.0 | 12.1 |

The authoritative pins are the self-contained
[L20 Qwen3](l20/qwen3_requirements.txt), [L20 Qwen3.5](l20/qwen35_requirements.txt),
[GB10 Qwen3](gb10/qwen3_requirements.txt) and [GB10 Qwen3.5](gb10/qwen35_requirements.txt)
requirements files. Setup and checking read the same selected file, including
the PyTorch index, Torch, Triton, CMake, Ninja and model dependencies. Versions
can differ between models. JSON profiles contain OS, CPU architecture and GPU
capability checks; they do not duplicate package pins.

Qwen3 and Qwen3.5 currently pin the same Transformers `5.20.2` stack while
retaining per-model environment and build entry points. Setup uses official
matching wheels by default and permits a local Torch wheel override. For an
offline image, use the distinct [L20 Docker workflow](l20/docker/README.md).

## Create model environments

Run these Bash commands from the repository root. Setup itself can be called
from elsewhere; relative options resolve against the repository root.

```bash
TARGET=l20                         # or gb10
ENV_SUFFIX=-venv
if [ "$TARGET" = l20 ]; then
  ENV_SUFFIX=-torch213-venv
fi
RUN=baseline-001                    # fresh report directory

for MODEL in qwen3 qwen35; do
  MODEL_ENV="$PWD/work/$TARGET/$MODEL$ENV_SUFFIX"
  bash "environments/$TARGET/setup.sh" --model "$MODEL" --venv "$MODEL_ENV"
  bash environments/check.sh --python "$MODEL_ENV/bin/python" \
    --target "$TARGET" --model "$MODEL" \
    --output "artifacts/$TARGET/$MODEL/$RUN/environment.json"
done
```

Setup always creates a fresh independent venv and refuses an existing destination.
It installs the selected requirements, then runs `pip check`; it neither links
nor inherits packages from a system installation. Setup does not require a GPU.
Both target scripts accept `--model qwen3|qwen35`, `--python PATH` (default
`python3.12`), `--venv PATH` and `--torch-wheel PATH`. A local wheel replaces only
the Torch requirement. Default venvs are `work/l20/<model>-torch213-venv` and
`work/gb10/<model>-venv`.

For `check.sh`, put `--python PATH` first. The checker verifies installed packages
against requirements, the hardware profile, Torch CUDA build and C++ ABI, and
records compiler/runtime versions. `--build` also checks CMake's selected
LibTorch. Failure reports are written before a nonzero exit. Keep the environment
report with package/build evidence.

## Build compiler kernels and native runtimes for each model

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
    xinfer_attention_ops xinfer_aoti xinfer_validate xinfer_benchmark \
    qwen_runtime_contract xinfer_state_contract xinfer_generation_contract -j2
  bash environments/check.sh --python "$MODEL_PY" \
    --target "$TARGET" --model "$MODEL" --build "$BUILD" \
    --output "artifacts/$TARGET/$MODEL/$RUN/build-environment.json"
done
```

The compiler owns the `xinfer_attention_ops` target and its native sources in
`compiler/common/kernels/`. It writes `libxinfer_attention_ops.so` into the same
`build/<model>/` directory used by export and standalone operator tests.
The root build produces `xinfer_aoti` and, when enabled, the tooling workers
`xinfer_benchmark` and `xinfer_validate` for both model families. The default
build includes compiler kernels and the generation CLI. Selecting an explicit
native target builds it independently; build `xinfer_attention_ops` separately
before export or standalone attention tests. Model selection comes from package
metadata; there is no separate full-model build switch. Kernel export/AOTI checks run
through `compiler.qwen35.tests.test_kernels_aoti`; they need no native kernel
runner.
Each model's Python tools default to executables in its `build/<model>/`
directory; existing native-path options select another runtime. Export and
standalone operator tests select the attention library from that build or
`XINFER_ATTENTION_OPS_LIBRARY`.

`--build` verifies that CMake selected the active environment's LibTorch. Build
`xinfer_attention_ops` before either full-model export so Python can register
the shared native FlashAttention operator. Both exporters embed
`libxinfer_attention_ops.so` in each `.pt2`. The native executables load it through those packages and do not
link or install a separate copy. Qwen3.5 also packages its GDN kernels.

For L20, the two CMake runtime variables select CUDA 12.6 from the model
installation for both Torch and CUDA discovery. This also supports a host
compiler with an older compatible CUDA minor version; linking Torch 2.13 to an
older host `libcudart` can leave unresolved runtime symbols. GB10 uses its CUDA
13.0 runtime selection; the extra argument array is empty there.

## Prepare inputs and export

After building, follow the [Qwen3](../compiler/qwen3/README.md#prepare-and-export)
or [Qwen3.5](../compiler/qwen35/README.md#prepare-and-export) guide with `Q3_PY` or
`Q35_PY` from the build section. Download once before enabling offline mode;
export both entries into a fresh directory. The shared
[preparation tool](../docs/development-tools.md#checkpoint-preparation-and-tokenization)
owns download, verification and token IDs. Current packaging uses a bundled
native attention dependency; see the [contract](../docs/flash-attention.md#build-and-deployment).

## Install and relocate the runtime

After building, install the project's runtime component into a fresh prefix.
Either model's build contains the shared executables supporting both models;
the example below installs from `build/qwen3/`.
GNUInstallDirs selects the executable directory, typically `bin/`.
`xinfer_aoti` is always included; `xinfer_benchmark` and `xinfer_validate` are
included when their build options are enabled. The focused
`qwen_runtime_contract` helper remains a build-only target. The attention library
is carried by each model package and is not installed in this prefix.
`CMAKE_INSTALL_BINDIR` can customize the executable layout and must be relative.
The following commands use the default `bin/` directory.

```bash
CMAKE="$(dirname "$Q3_PY")/cmake"
BUILD="$Q3_BUILD"
Q3_MODEL=work/models/qwen3-0.6b
Q3_PACKAGES=artifacts/qwen3-flash
Q3_ROOT="$PWD/work/$TARGET"
PREFIX="$PWD/work/$TARGET/runtime-install"
MOVED_PREFIX="$PWD/work/$TARGET/runtime-relocated"
"$CMAKE" --install "$BUILD" --prefix "$PREFIX" --component Runtime
mv "$PREFIX" "$MOVED_PREFIX"
"$MOVED_PREFIX/bin/xinfer_aoti" --help

"$Q3_PY" -m tools.prepare encode --model-dir "$Q3_MODEL" \
  --prompt "What is the capital of France?" --output "$Q3_ROOT/install-prompt.ids"
"$MOVED_PREFIX/bin/xinfer_aoti" "$Q3_PACKAGES" \
  "$Q3_ROOT/install-prompt.ids" "$Q3_ROOT/install-generated.ids" --max-output-tokens 8
```

The executable prefix and complete model bundle can move on the same
compatible native stack. When an AOTI loader extracts a package, the wrapper
finds its bundled attention dependency relative to `$ORIGIN`. No attention
library path or Python preload is required for package inference. The selected
LibTorch and CUDA libraries remain external; the installed RPATH retains their
configured runtime paths. `XINFER_INSTALL_RUNTIME_DIRS` can override these paths
at configure time with a semicolon-separated list; its defaults are the selected Torch library
and CUDA runtime directories. CMake also retains external linked-library
directories. A different host must provide a compatible GPU/package contract and
the matching native stack at those library paths; relocating that external
stack is a separate step. Deploy each complete five-file model package directory
alongside the installed runtime.

CPU portability is a separate constraint: generated AOTI wrappers can inherit
the build host's instruction set, including AVX-512. The runtime checks GPU and
library compatibility but not CPU ISA. Use a compatible CPU or re-export on the
destination host; moving an install prefix does not change the wrapper's ISA.

## Verify a new environment or bundle

Run the [CPU checks](../docs/development-tools.md#cpu-unit-tests) with each model's
interpreter, then its affected GPU operator/export/AOTI tests. CPU imports and
build success alone do not establish GPU compatibility. Full-model checks use
the shared [numerical chain and runtime tools](../docs/development-tools.md#python-validation-tools).
Keep native/Python bitwise parity, structural state checks and the shared
numerical limits strict; no cross-target bitwise equality is implied.

Run [benchmarks](../docs/development-tools.md#benchmark-hf-and-native-aoti)
separately on an idle device. Reports retain GPU identity, workload and package/
binary checksums. Record new results by model, device and capacity in
[validation status](../docs/validation-status.md), without treating older results
as a pass for a new build.

The Docker lock files are complete resolved snapshots for the offline image,
including image-specific tools. Refresh them and the wheel payload from the
per-device/model requirements when changing pins; native setup does not regenerate
Docker locks. See the [Docker guide](l20/docker/README.md) for offline transfer,
launch and the temporary Nsight library-path workaround.
