#!/usr/bin/env bash
# Fresh CPU preparation environment; native inference uses gb10/setup.sh.
set -euo pipefail
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python=python3.12
venv=work/prepare/qwen36-venv
usage() {
  echo 'Usage: environments/prepare-qwen36.sh [--python PATH] [--venv PATH]'
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python|--venv)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      if [[ "$1" == --python ]]; then python=$2; else venv=$2; fi
      shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
case "$venv" in /*) ;; *) venv="$repo_root/$venv" ;; esac
if [[ -e "$venv" || -L "$venv" ]]; then
  echo "Refusing to overwrite an existing preparation environment: $venv" >&2
  exit 2
fi
"$python" - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit("Python 3.12 is required")
PY
mkdir -p "$(dirname "$venv")"
"$python" -m venv "$venv"
"$venv/bin/python" -m pip install -r "$repo_root/environments/qwen36_prepare_requirements.txt"
"$venv/bin/python" -m pip check
printf 'Preparation environment: %s\n' "$venv"
