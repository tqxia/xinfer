#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: environments/build.sh [options]

Configure and build xinfer compiler kernels and native executables.

  --target l20|gb10    Target profile (default: auto-detected from uname -m)
  --python PATH        Venv Python providing Torch/CMake/Ninja (default: work/<target>/venv/bin/python)
  --build PATH         CMake build directory (default: build)
  --nvcc PATH          CUDA compiler (default: $CUDA_HOME/bin/nvcc or /usr/local/cuda/bin/nvcc)
  --jobs N             Parallel build jobs (default: 2)
  --targets LIST       Comma-separated CMake targets (default: all enabled targets)
  --no-validation      Configure with -DXINFER_BUILD_VALIDATION=OFF
  --no-benchmark       Configure with -DXINFER_BUILD_BENCHMARK=OFF
  -h, --help           Show this help

Relative paths are resolved from the repository root. When
FETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON is set in the environment, CMake is
configured offline with FETCHCONTENT_FULLY_DISCONNECTED=ON.
EOF
}

fail() {
  printf '%s\n' "$*" >&2
  exit 2
}

detect_target() {
  case "$(uname -m)" in
    x86_64|amd64) printf 'l20\n' ;;
    aarch64|arm64) printf 'gb10\n' ;;
    *) fail "Unsupported host architecture $(uname -m); pass --target l20 or gb10" ;;
  esac
}

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
target=""
python=""
build_dir="build"
nvcc="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
jobs="2"
targets=""
build_validation="ON"
build_benchmark="ON"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target|--python|--build|--nvcc|--jobs|--targets)
      [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || fail "Missing value for $1"
      case "$1" in
        --target) target="$2" ;;
        --python) python="$2" ;;
        --build) build_dir="$2" ;;
        --nvcc) nvcc="$2" ;;
        --jobs) jobs="$2" ;;
        --targets) targets="$2" ;;
      esac
      shift 2
      ;;
    --no-validation)
      build_validation="OFF"
      shift
      ;;
    --no-benchmark)
      build_benchmark="OFF"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown option: $1"
      ;;
  esac
done

if [[ -z "$target" ]]; then
  target="$(detect_target)"
fi
case "$target" in
  l20|gb10) ;;
  *) usage >&2; fail "Select --target l20 or gb10" ;;
esac
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || fail "--jobs must be a positive integer: $jobs"

cd "$repo_root"
python="${python:-work/$target/venv/bin/python}"
[[ "$python" = /* ]] || python="$repo_root/$python"
[[ -x "$python" ]] || fail "Python executable not found: $python"
[[ "$build_dir" = /* ]] || build_dir="$repo_root/$build_dir"
[[ "$nvcc" = /* ]] || nvcc="$repo_root/$nvcc"

venv_bin="$(dirname -- "$python")"
if [[ -x "$venv_bin/cmake" ]]; then
  cmake_bin="$venv_bin/cmake"
elif command -v cmake >/dev/null 2>&1; then
  cmake_bin="$(command -v cmake)"
else
  fail "cmake not found in $venv_bin or PATH"
fi

torch_prefix="$("$python" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  cuda_arch="$TORCH_CUDA_ARCH_LIST"
else
  cuda_arch="$("$python" -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability()) if torch.cuda.is_available() else print("")')"
  if [[ -z "$cuda_arch" ]]; then
    case "$target" in
      l20) cuda_arch="8.9" ;;
      gb10) cuda_arch="12.1" ;;
    esac
  fi
fi

cmake_args=(
  -S "$repo_root"
  -B "$build_dir"
  -G Ninja
  "-DCMAKE_PREFIX_PATH=$torch_prefix"
  "-DCMAKE_CUDA_COMPILER=$nvcc"
  "-DTORCH_CUDA_ARCH_LIST=$cuda_arch"
  -DCMAKE_BUILD_TYPE=Release
  "-DXINFER_BUILD_VALIDATION=$build_validation"
  "-DXINFER_BUILD_BENCHMARK=$build_benchmark"
)

if [[ "$target" == "l20" ]]; then
  cuda_runtime="$("$python" -c 'from pathlib import Path; import torch; p = Path(torch.__file__).resolve().parent.parent / "nvidia/cuda_runtime/lib/libcudart.so.12"; assert p.is_file(), f"Missing {p}"; print(p)')"
  cmake_args+=(
    "-DCUDA_cudart_LIBRARY=$cuda_runtime"
    "-DCUDA_CUDART_LIBRARY=$cuda_runtime"
  )
fi

if [[ -n "${FETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON:-}" ]]; then
  cmake_args+=(
    "-DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=$FETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON
  )
fi

build_args=(--build "$build_dir" -j "$jobs")
if [[ -n "$targets" ]]; then
  IFS=',' read -r -a target_list <<< "$targets"
  build_args+=(--target "${target_list[@]}")
fi

export PATH="$venv_bin:$PATH"
printf '+'
printf ' %q' "$cmake_bin" "${cmake_args[@]}"
printf '\n'
"$cmake_bin" "${cmake_args[@]}"

printf '+'
printf ' %q' "$cmake_bin" "${build_args[@]}"
printf '\n'
"$cmake_bin" "${build_args[@]}"
