#!/usr/bin/env bash
# Optional native headers only: FlashInfer plus its pinned CCCL submodule.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
revision=beabf5004585b03da60e927bbb803acbaa62fe36
destination=work/deps/flashinfer
if [[ ! -e "$destination" ]]; then
  mkdir -p work/deps
  git clone --depth 1 --branch v0.7.2.post1 https://github.com/flashinfer-ai/flashinfer.git "$destination"
fi
test "$(git -C "$destination" rev-parse HEAD)" = "$revision"
git -C "$destination" diff --exit-code HEAD -- include
git -C "$destination" submodule update --init --depth 1 3rdparty/cccl
test "$(git -C "$destination/3rdparty/cccl" rev-parse HEAD)" = 16bd510c9b712e82b0ab6cbb630d8e29ba1f7116
git -C "$destination/3rdparty/cccl" diff --exit-code HEAD
echo "FlashInfer source verified: $revision"
