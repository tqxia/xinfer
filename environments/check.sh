#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: environments/check.sh --python MODEL_PYTHON [check options]\n'
}

case "${1:-}" in
  --help|-h)
    usage
    python=python3
    set -- --help
    ;;
  --python)
    if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == -* ]]; then
      usage >&2
      exit 2
    fi
    python="$2"
    shift 2
    ;;
  --python=*)
    python="${1#--python=}"
    shift
    if [[ -z "$python" ]]; then
      usage >&2
      exit 2
    fi
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
exec "$python" -m environments.check "$@"
