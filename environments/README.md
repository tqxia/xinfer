# L20 and GB10 environments

This guide owns native setup, environment checks, builds and installation.
Use the model guides for [Qwen3](../compiler/qwen3/README.md),
[Qwen3.5](../compiler/qwen35/README.md) and [Qwen3.8](../compiler/qwen38/README.md)
export details, and
[development tools](../docs/development-tools.md) for preparation, validation and
benchmarks. [Validation status](../docs/validation-status.md) records what has
actually been tested; a supported environment is not a blanket model-acceptance claim.

## Requirements and targets

All supported models (Qwen3, Qwen3.5 and Qwen3.8) share one Python 3.12 virtual
environment per target host. Compile native code on the target host against that
environment's LibTorch. The host needs Python 3.12, Linux with glibc 2.28 or
newer, a C++ compiler, a compatible CUDA toolkit and an NVIDIA driver. The
export linker path uses Linux GCC.

| Target | CPU architecture | Torch runtime | Torch CUDA | GPU capability |
| --- | --- | --- | --- | --- |
| L20 | x86-64 | 2.13.0+cu126 | 12.6 | 8.9 |
| GB10 | ARM64 | 2.13.0+cu130 | 13.0 | 12.1 |

The authoritative pins live in [requirements.txt](requirements.txt) (shared
Triton, CMake, Ninja, Transformers `5.20.2` and `kernels` pins), included by
[L20 requirements](l20/requirements.txt) and
[GB10 requirements](gb10/requirements.txt) for each target's Torch/CUDA build.
Setup and checking read the same file. Target host checks (OS, CPU
architecture, Python 3.12 and GPU capability) live in [check.py](check.py).
Setup uses official wheels by default and accepts a local Torch
wheel override. For a containerized L20 setup on the official NVIDIA CUDA 12.6
image, use the [L20 Docker workflow](l20/docker/README.md).

## Create target environment

Run these Bash commands from the repository root. Setup can also be called from
elsewhere; relative options resolve against the repository root.

```bash
TARGET=l20                         # or gb10 (auto-detected if omitted)
RUN=baseline-001                   # fresh report directory

bash environments/setup.sh --target "$TARGET"
PY="$PWD/work/$TARGET/venv/bin/python"
"$PY" -m environments.check --target "$TARGET" \
  --output "artifacts/$TARGET/$RUN/environment.json"
```

`environments/setup.sh` auto-detects `l20` (`x86_64`) or `gb10` (`aarch64`) when
`--target` is omitted. It creates a fresh independent venv at
`work/<target>/venv`, installs the target's requirements, runs `pip check`, and
removes the partial venv if installation fails. Existing destinations are
refused and preserved. Supported options are `--target l20|gb10`,
`--python PATH` (default `python3.12`), `--venv PATH`, `--torch-wheel PATH`
(replaces only `torch==`), and `--dev` (also installs `requirements-dev.txt`).
Setup does not require a GPU.

`python -m environments.check` verifies installed packages against the target
requirements, Python 3.12, OS/architecture, native tools (`c++`, `nvcc`), Torch
CUDA build, C++ ABI and GPU capability. Pass `--dev` to also check formatting
tools, and `--build PATH` to verify CMake's `Torch_DIR`. Failure reports are
written before exiting nonzero.

## Build compiler kernels and native runtimes

Use `environments/build.sh` to configure and build `xinfer_attention_ops`,
`xinfer_aoti`, `xinfer_validate`, `xinfer_benchmark` and the native contract
tests into `build/` against the active venv's LibTorch:

```bash
PY="$PWD/work/$TARGET/venv/bin/python"

bash environments/build.sh --target "$TARGET" --python "$PY"
"$PY" -m environments.check --target "$TARGET" --build build \
  --output "artifacts/$TARGET/$RUN/build-environment.json"
```

`environments/build.sh` resolves `torch.utils.cmake_prefix_path`, CUDA
architecture (live GPU capability, `TORCH_CUDA_ARCH_LIST`, or the target
default `8.9`/`12.1`), L20's Torch-supplied `libcudart.so.12` link flags, and
offline `FETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON` automatically, printing the
exact CMake commands it runs. Useful options: `--build PATH` (default `build`),
`--nvcc PATH`, `--jobs N` (default `2`), `--targets LIST` (comma-separated
subset), `--no-validation`, `--no-benchmark`.

The compiler owns the `xinfer_attention_ops` target and its native sources in
`compiler/common/kernels/`. It writes `libxinfer_attention_ops.so` into `build/`
for export and standalone operator tests.
The root build produces `xinfer_aoti` and, when enabled, the tooling workers
`xinfer_benchmark` and `xinfer_validate` for all model families. Model selection
comes from package metadata; there is no separate model build switch.
Kernel export/AOTI checks run through
`compiler.common.tests.gated_delta_net.test_aoti`; they need no native kernel
runner.

Export entry points load `build/libxinfer_attention_ops.so` by default (or
`XINFER_ATTENTION_OPS_LIBRARY`). Numerical validation and benchmark tools
default to native executables in `build/` and dispatch by package `model_type`
(`qwen3`, `qwen35` or `qwen38`); native-path options (`--native`, `--binary`,
`--cli`) select another build directory. Qwen3.8 packages use
`model_type="qwen38"` with the shared Qwen3.5 runtime and calling convention.

Build `xinfer_attention_ops` before full-model export so Python can register the
shared native FlashAttention operator. Exporters embed
`libxinfer_attention_ops.so` in each `.pt2`; native executables load it through
those packages and do not link or install a separate copy.

For L20, `build.sh` passes `-DCUDA_cudart_LIBRARY` and `-DCUDA_CUDART_LIBRARY`
pointing to Torch's CUDA 12.6 runtime so host toolchains with an older CUDA
minor version do not leave unresolved runtime symbols. GB10 uses its CUDA 13.0
runtime selection directly.

## Prepare inputs and export

After building, follow the [Qwen3](../compiler/qwen3/README.md#prepare-and-export),
[Qwen3.5](../compiler/qwen35/README.md#prepare-and-export) or
[Qwen3.8](../compiler/qwen38/README.md) guide with `PY` from above.
Download once before enabling offline mode; export both entries into a fresh
directory. The shared
[preparation tool](../docs/development-tools.md#checkpoint-preparation-and-tokenization)
owns download, verification and token IDs. Current packaging uses a bundled
native attention dependency; see the [contract](../docs/flash-attention.md#build-and-deployment).

## Install and relocate the runtime

After building, install the project's runtime component from `build/` into a
fresh prefix. The executables support all model families.
GNUInstallDirs selects the executable directory, typically `bin/`.
`xinfer_aoti` is always included; `xinfer_benchmark` and `xinfer_validate` are
included when their build options are enabled. The focused
`qwen_runtime_contract` helper remains a build-only target. The attention library
is carried by each model package and is not installed in this prefix.
`CMAKE_INSTALL_BINDIR` can customize the executable layout and must be relative.

```bash
CMAKE="$(dirname "$PY")/cmake"
BUILD="$PWD/build"
Q3_MODEL=work/models/qwen3-0.6b
Q3_PACKAGES=artifacts/qwen3-flash
Q3_ROOT="$PWD/work/$TARGET"
PREFIX="$PWD/work/$TARGET/runtime-install"
MOVED_PREFIX="$PWD/work/$TARGET/runtime-relocated"
"$CMAKE" --install "$BUILD" --prefix "$PREFIX" --component Runtime
mv "$PREFIX" "$MOVED_PREFIX"
"$MOVED_PREFIX/bin/xinfer_aoti" --help

"$PY" -m tools.prepare encode --model-dir "$Q3_MODEL" \
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
stack is a separate step. Deploy each complete four-file model package directory
alongside the installed runtime.

CPU portability is a separate constraint: generated AOTI wrappers can inherit
the build host's instruction set, including AVX-512. The runtime checks GPU and
library compatibility but not CPU ISA. Use a compatible CPU or re-export on the
destination host; moving an install prefix does not change the wrapper's ISA.

## Verify a new environment or bundle

Run the [CPU checks](../docs/development-tools.md#cpu-unit-tests) with the
target interpreter, then the affected GPU operator/export/AOTI tests. CPU
imports and build success alone do not establish GPU compatibility. Full-model
checks use the shared
[numerical chain and runtime tools](../docs/development-tools.md#python-validation-tools).
Keep native/Python bitwise parity, structural state checks and the shared
numerical limits strict; no cross-target bitwise equality is implied.

Run [benchmarks](../docs/development-tools.md#benchmark-hf-and-native-aoti)
separately on an idle device. Reports retain GPU identity, workload and package/
binary checksums. Record new results by model, device and capacity in
[validation status](../docs/validation-status.md), without treating older results
as a pass for a new build.

The [L20 Docker image](l20/docker/README.md) builds directly from
`environments/l20/requirements.txt` and `requirements-dev.txt` on
`nvidia/cuda:12.6.3-devel-ubuntu24.04` with a single `/opt/xinfer/venv` (`$PY`)
and preinstalled `nlohmann-json3-dev`.
