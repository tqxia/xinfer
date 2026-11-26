cmake_minimum_required(VERSION 3.24)
project(xinfer LANGUAGES CXX CUDA)

# Configure TORCH_CUDA_ARCH_LIST before find_package(Torch) so PyTorch selects SM80+
# Tensor Core targets cleanly instead of relying on its broken 12.1 autodetection.
if(NOT DEFINED TORCH_CUDA_ARCH_LIST AND "$ENV{TORCH_CUDA_ARCH_LIST}" STREQUAL "")
  execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE _xinfer_caps
    RESULT_VARIABLE _xinfer_smi_rc
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(_xinfer_smi_rc EQUAL 0 AND NOT "${_xinfer_caps}" STREQUAL "")
    string(REPLACE "\n" ";" _xinfer_caps "${_xinfer_caps}")
    list(REMOVE_DUPLICATES _xinfer_caps)
    set(TORCH_CUDA_ARCH_LIST "${_xinfer_caps}" CACHE STRING "CUDA architectures for Torch kernels")
  else()
    set(TORCH_CUDA_ARCH_LIST "8.0+PTX" CACHE STRING "CUDA architectures for Torch kernels")
  endif()
endif()

find_package(Torch 2.13 REQUIRED)
# Runtime headers require at least C++20; preserve higher standards from LibTorch.
get_target_property(XINFER_CXX_STANDARD torch CXX_STANDARD)
if(NOT XINFER_CXX_STANDARD)
  set(XINFER_CXX_STANDARD 20)
elseif(XINFER_CXX_STANDARD LESS 20)
  set(XINFER_CXX_STANDARD 20)
endif()
separate_arguments(TORCH_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
set(CMAKE_CUDA_STANDARD ${XINFER_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the CLI paths stable when building from the repository root.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
option(XINFER_BUILD_VALIDATION "Build native validation drivers" OFF)
option(XINFER_BUILD_BENCHMARK "Build fixed-token native benchmarks" OFF)
add_subdirectory(compiler)
add_subdirectory(runtime)
add_subdirectory(tools)
