#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if (( $# < 1 || $# > 2 )); then
    echo "Usage: $0 'prompt' [max-new-tokens]" >&2
    exit 2
fi
mkdir -p work
request_dir=$(mktemp -d "$PWD/work/chat.XXXXXX")
.venv/bin/python compiler/prepare.py encode --prompt "$1" --output "$request_dir/prompt.ids"
./build/qwen_aoti artifacts/aoti-eager "$request_dir/prompt.ids" "$request_dir/generated.ids" \
    --max-new-tokens "${2:-64}"
.venv/bin/python compiler/prepare.py decode --input "$request_dir/generated.ids"
