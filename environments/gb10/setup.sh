#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
target=gb10
env_suffix=-venv
source "$repo_root/environments/setup-common.sh"
