#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
torch_cmake=$(.venv/bin/python -c 'import torch; print(torch.utils.cmake_prefix_path)')
torch_arch=$(.venv/bin/python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')
cmake -S . -B build -G Ninja \
    -DCMAKE_PREFIX_PATH="$torch_cmake" \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    -DTORCH_CUDA_ARCH_LIST="$torch_arch" \
    -DCMAKE_BUILD_TYPE=Release \
    -DQWEN_BUILD_VALIDATION=ON -DQWEN_BUILD_BENCHMARK=ON "$@"
cmake --build build -j2
