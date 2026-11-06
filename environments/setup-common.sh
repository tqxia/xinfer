#!/usr/bin/env bash
# Shared argument handling and installation for the device setup entry points.

usage() {
  cat <<EOF
Usage: environments/$target/setup.sh --model qwen3|qwen35 [options]

Create a fresh model environment from the device and model requirements.

  --python PATH       Python 3.12 executable (default: python3.12)
  --venv PATH         Fresh venv directory (default: work/$target/<model>$env_suffix)
  --torch-wheel PATH  Install a local Torch wheel instead of downloading it
  -h, --help          Show this help

Relative paths are resolved from the repository root. The host must already
provide a C++ compiler, CUDA toolkit and NVIDIA driver. GPU checks are separate.
EOF
}

fail() {
  printf '%s\n' "$*" >&2
  exit 2
}

model=""
python=python3.12
venv=""
torch_wheel=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|--python|--venv|--torch-wheel)
      [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || fail "Missing value for $1"
      case "$1" in
        --model) model="$2" ;;
        --python) python="$2" ;;
        --venv) venv="$2" ;;
        --torch-wheel) torch_wheel="$2" ;;
      esac
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown option: $1" ;;
  esac
done
case "$model" in
  qwen3|qwen35) ;;
  *) usage >&2; fail "Select --model qwen3 or qwen35" ;;
esac

cd "$repo_root"
venv="${venv:-work/$target/$model$env_suffix}"
[[ "$venv" = /* ]] || venv="$repo_root/$venv"
[[ ! -e "$venv" && ! -L "$venv" ]] || fail "Refusing to overwrite $venv; choose a fresh --venv directory"
requirements="$repo_root/environments/$target/${model}_requirements.txt"
if [[ -n "$torch_wheel" ]]; then
  [[ -f "$torch_wheel" ]] || fail "Torch wheel does not exist: $torch_wheel"
  [[ "$torch_wheel" = /* ]] || torch_wheel="$repo_root/$torch_wheel"
  requirements_override="$(mktemp "${TMPDIR:-/tmp}/xinfer-requirements.XXXXXX")"
  trap 'rm -f "$requirements_override"' EXIT
  sed '/^torch==/d' "$requirements" > "$requirements_override"
  requirements="$requirements_override"
fi

set -- -r "$requirements"
if [[ -n "$torch_wheel" ]]; then
  set -- "$torch_wheel" "$@"
fi

unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
"$python" -m venv "$venv"
model_python="$venv/bin/python"
"$model_python" -m pip install "$@"
"$model_python" -m pip check
printf 'Created %s\n' "$venv"
