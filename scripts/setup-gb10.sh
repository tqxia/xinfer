#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
mkdir -p work/wheels
.venv/bin/python -m pip download --no-deps --dest work/wheels torch==2.13.0 \
    --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pip install --find-links work/wheels -r compiler/requirements-gb10.lock
.venv/bin/python tools/setup/fix_cusparselt_metadata.py
.venv/bin/python -m pip check
.venv/bin/python tools/setup/download_model.py
