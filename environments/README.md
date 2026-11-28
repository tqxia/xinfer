# Target environments (L20, PRO 5000 and GB10)

Owns target venv setup, environment checks, native builds, Docker workflows and
runtime installation. See [Qwen3](../compiler/qwen3/README.md),
[Qwen3.5](../compiler/qwen35/README.md) and
[Qwen3.8](../compiler/qwen38/README.md) for export details,
[development tools](../docs/development-tools.md) for validation/benchmarks, and
[validation status](../docs/validation-status.md) for tested profiles.

## Requirements and targets

All supported models share one Python 3.12 virtual environment per target host.
Hosts require Linux (glibc >= 2.28), Python 3.12, GCC/C++20, a Rust toolchain
(`cargo`/`rustc`), a compatible CUDA toolkit and an NVIDIA driver.

| Target | CPU architecture | Torch runtime | Torch CUDA | GPU capability |
| --- | --- | --- | --- | --- |
| L20 | x86-64 | 2.13.0+cu126 | 12.6 | 8.9 |
| PRO 5000 | x86-64 | 2.13.0+cu130 | 13.0 | 12.0 |
| GB10 | ARM64 | 2.13.0+cu130 | 13.0 | 12.1 |

Dependency pins live in [requirements.txt](requirements.txt) (shared Triton,
CMake, Ninja, Transformers `5.20.2`, `huggingface-hub`, `tokenizers`,
`safetensors` and `kernels`), included by
[l20/requirements.txt](l20/requirements.txt),
[pro5000/requirements.txt](pro5000/requirements.txt) and
[gb10/requirements.txt](gb10/requirements.txt). Host checks live in
[check.py](check.py). For containerized testing, see the
[L20 Docker workflow](l20/docker/README.md) (`nvidia/cuda:12.6.3-devel-ubuntu24.04`)
or [PRO 5000 Docker workflow](pro5000/docker/README.md)
(`nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04`).

## Create target environment

```bash
TARGET=l20                         # l20, pro5000, or gb10 (auto-detected if omitted)
RUN=baseline-001                   # fresh report directory

bash environments/setup.sh --target "$TARGET"
PY="$PWD/work/$TARGET/venv/bin/python"
"$PY" -m environments.check --target "$TARGET"   --output "artifacts/$TARGET/$RUN/environment.json"
```

`setup.sh` auto-detects `gb10` (`aarch64`), `l20` (SM 8.9) or `pro5000` (SM 12.0)
when `--target` is omitted (pass `--target` explicitly on CPU-only `x86_64`
hosts), creates `work/<target>/venv`, runs `pip check`, and refuses existing
destinations (options: `--target`, `--python`, `--venv`, `--torch-wheel`,
`--dev`). `python -m environments.check` verifies packages, Python 3.12,
OS/arch, `c++`/`nvcc`/`cargo`/`rustc`, Torch CUDA, C++ ABI and GPU capability
(plus `--dev` formatters and `--build PATH` CMake `Torch_DIR`).

## Build compiler kernels and native runtimes

```bash
PY="$PWD/work/$TARGET/venv/bin/python"

bash environments/build.sh --target "$TARGET" --python "$PY"
"$PY" -m environments.check --target "$TARGET" --build build   --output "artifacts/$TARGET/$RUN/build-environment.json"
```

`build.sh` configures and builds all targets into `build/` against the venv's
LibTorch: compiler operator libraries `libxinfer_attention_ops.so` and
`libxinfer_deltanet_ops.so` (`compiler/common/kernels/`), and native Rust
executables `xinfer_aoti`, `xinfer_serve`, `xinfer_validate` and
`xinfer_benchmark` (all model families use the same binaries, dispatched via
`metadata.json`). Options: `--build PATH`, `--nvcc PATH`, `--jobs N` (default
`2`), `--targets LIST`, `--no-server`, `--no-validation`, `--no-benchmark`. Set
`CARGO_NET_OFFLINE=true` to build Rust targets against a pre-populated Cargo
cache.

Build `xinfer_attention_ops` (and `xinfer_deltanet_ops` for Qwen3.5/Qwen3.8)
before export or standalone operator tests; export loads them from `build/` by
default (override via `XINFER_ATTENTION_OPS_LIBRARY` /
`XINFER_DELTANET_OPS_LIBRARY`) and embeds them into each `.pt2`. Runtime
executables consume the embedded copies at load time and do not link separate
operator libraries.

On L20, `build.sh` points `CUDA_cudart_LIBRARY` / `CUDA_CUDART_LIBRARY` at
Torch's bundled `libcudart.so.12` so older host CUDA toolkits do not leave
unresolved symbols; PRO 5000 and GB10 use CUDA 13.0 directly.

## Prepare inputs and export

Follow the [Qwen3](../compiler/qwen3/README.md#prepare-and-export),
[Qwen3.5](../compiler/qwen35/README.md#prepare-and-export) or
[Qwen3.8](../compiler/qwen38/README.md#prepare-and-export) guide with `PY`.
Download checkpoints once before enabling offline mode (see
[preparation](../docs/development-tools.md#checkpoint-preparation-and-tokenization)).

## Install and relocate the runtime

Install the `Runtime` component into a standalone prefix (`xinfer_aoti` is
always installed; `xinfer_serve`, `xinfer_benchmark` and `xinfer_validate` are
included when enabled; `CMAKE_INSTALL_BINDIR` defaults to `bin/` and must be
relative):

```bash
CMAKE="$(dirname "$PY")/cmake"
PREFIX="$PWD/work/$TARGET/runtime-install"
MOVED_PREFIX="$PWD/work/$TARGET/runtime-relocated"

"$CMAKE" --install "$PWD/build" --prefix "$PREFIX" --component Runtime
mv "$PREFIX" "$MOVED_PREFIX"
"$MOVED_PREFIX/bin/xinfer_aoti" --help
"$MOVED_PREFIX/bin/xinfer_serve" --help
```

Installed binaries and complete model bundles can move freely on the same
compatible LibTorch/CUDA stack: embedded operator libraries resolve via
`$ORIGIN`, and `RUNPATH` retains the configured LibTorch and CUDA runtime
directories (override at configure time with semicolon-separated
`XINFER_INSTALL_RUNTIME_DIRS`). Deploy each complete model bundle directory
alongside the installed runtime.

CPU ISA compatibility is not verified at load time: compiled AOTI wrappers can
inherit the build host's instruction set (including AVX-512). Use a compatible
CPU or re-export on the destination host.

## Verify a new environment or bundle

1. Run [CPU unit tests](../docs/development-tools.md#cpu-unit-tests) (Python and
   Rust), then the affected GPU operator/export tests from the model guides.
2. Run [numerical and runtime validation](../docs/development-tools.md#python-validation-tools)
   on each new bundle.
3. Run [benchmarks](../docs/development-tools.md#benchmark-hf-and-native-aoti)
   separately on an idle GPU, and record new results in
   [validation status](../docs/validation-status.md).
