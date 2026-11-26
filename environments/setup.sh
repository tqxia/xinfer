#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: environments/setup.sh [options]

Create a fresh Python 3.12 virtual environment for this target.

  --target l20|pro5000|gb10  Target profile (default: auto-detected from CPU/GPU)
  --python PATH              Python 3.12 executable (default: python3.12)
  --venv PATH                Fresh venv directory (default: work/<target>/venv)
  --torch-wheel PATH         Install a local Torch wheel instead of downloading it
  --dev                      Also install formatting tools from requirements-dev.txt
  -h, --help                 Show this help

Relative paths are resolved from the repository root. The host must already
provide a C++ compiler, CUDA toolkit and NVIDIA driver. GPU checks are separate.
EOF
}

fail() {
  printf '%s\n' "$*" >&2
  exit 2
}

detect_target() {
  case "$(uname -m)" in
    aarch64|arm64) printf 'gb10\n' ;;
    x86_64|amd64)
      local caps=""
      if command -v nvidia-smi >/dev/null 2>&1; then
        caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' \r' | sort -u)"
      fi
      case "$caps" in
        8.9) printf 'l20\n' ;;
        12.0) printf 'pro5000\n' ;;
        *) fail "Cannot auto-detect x86_64 target from GPU; pass --target l20 or pro5000" ;;
      esac
      ;;
    *) fail "Unsupported host architecture $(uname -m); pass --target l20, pro5000 or gb10" ;;
  esac
}

write_filtered_requirements() {
  local source_file="$1"
  local base_dir
  base_dir="$(cd -- "$(dirname -- "$source_file")" && pwd)"
  local line included
  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      torch==*) continue ;;
      "-r "*)
        included="${line#-r }"
        [[ "$included" = /* ]] || included="$base_dir/$included"
        write_filtered_requirements "$included"
        ;;
      "--requirement "*)
        included="${line#--requirement }"
        [[ "$included" = /* ]] || included="$base_dir/$included"
        write_filtered_requirements "$included"
        ;;
      *) printf '%s\n' "$line" ;;
    esac
  done < "$source_file"
}

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
target=""
python=python3.12
venv=""
torch_wheel=""
include_dev=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target|--python|--venv|--torch-wheel)
      [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || fail "Missing value for $1"
      case "$1" in
        --target) target="$2" ;;
        --python) python="$2" ;;
        --venv) venv="$2" ;;
        --torch-wheel) torch_wheel="$2" ;;
      esac
      shift 2
      ;;
    --dev)
      include_dev=1
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
  l20|pro5000|gb10) ;;
  *) usage >&2; fail "Select --target l20, pro5000 or gb10" ;;
esac

cd "$repo_root"
venv="${venv:-work/$target/venv}"
[[ "$venv" = /* ]] || venv="$repo_root/$venv"
[[ ! -e "$venv" && ! -L "$venv" ]] || fail "Refusing to overwrite $venv; choose a fresh --venv directory"

requirements="$repo_root/environments/$target/requirements.txt"
[[ -f "$requirements" ]] || fail "Requirements file not found: $requirements"

requirements_override=""
created_venv=""
cleanup() {
  local status=$?
  [[ -n "$requirements_override" ]] && rm -f "$requirements_override"
  if [[ $status -ne 0 && -n "$created_venv" ]]; then
    rm -rf "$created_venv"
  fi
}
trap cleanup EXIT

if [[ -n "$torch_wheel" ]]; then
  [[ "$torch_wheel" = /* ]] || torch_wheel="$repo_root/$torch_wheel"
  [[ -f "$torch_wheel" ]] || fail "Torch wheel does not exist: $torch_wheel"
  requirements_override="$(mktemp "${TMPDIR:-/tmp}/xinfer-requirements.XXXXXX")"
  write_filtered_requirements "$requirements" > "$requirements_override"
  requirements="$requirements_override"
fi

install_args=(-r "$requirements")
if [[ -n "$torch_wheel" ]]; then
  install_args=("$torch_wheel" "${install_args[@]}")
fi
if [[ $include_dev -eq 1 ]]; then
  install_args+=(-r "$repo_root/requirements-dev.txt")
fi

unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
created_venv="$venv"
"$python" -m venv "$venv"
model_python="$venv/bin/python"
"$model_python" -m pip install "${install_args[@]}"
"$model_python" -m pip check
printf 'Created %s\n' "$venv"
