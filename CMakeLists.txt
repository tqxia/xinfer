cmake_minimum_required(VERSION 3.24)
project(xinfer LANGUAGES CXX)

find_package(Torch 2.13 REQUIRED)
# Match the installed LibTorch headers, including C++20 on the GB10 toolchain.
get_target_property(XINFER_CXX_STANDARD torch CXX_STANDARD)
if(NOT XINFER_CXX_STANDARD)
  set(XINFER_CXX_STANDARD 17)
endif()
separate_arguments(TORCH_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

# Keep the CLI paths stable when building from the repository root.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
option(XINFER_BUILD_VALIDATION "Build native validation drivers" OFF)
option(XINFER_BUILD_BENCHMARK "Build fixed-token native benchmarks" OFF)
add_subdirectory(compiler)
add_subdirectory(runtime)
add_subdirectory(tools)
