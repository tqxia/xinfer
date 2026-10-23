cmake_minimum_required(VERSION 3.24)
project(qwen_aoti LANGUAGES CXX)

# Keep the CLI paths stable when building from the repository root.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_subdirectory(runtime)
