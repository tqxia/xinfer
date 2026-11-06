#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
target=l20
env_suffix=-torch213-venv
source "$repo_root/environments/setup-common.sh"
