find_package(Torch 2.13 REQUIRED)

# Match the installed LibTorch headers, including C++20 on the GB10 toolchain.
get_target_property(QWEN_CXX_STANDARD torch CXX_STANDARD)
if(NOT QWEN_CXX_STANDARD)
  set(QWEN_CXX_STANDARD 17)
endif()

# Header-only JSON parsing; pinned release and checksum, no runtime dependency.
include(FetchContent)
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
FetchContent_MakeAvailable(nlohmann_json)

option(QWEN_BUILD_VALIDATION "Build native validation drivers" OFF)
option(QWEN_BUILD_BENCHMARK "Build fixed-token native benchmarks" OFF)
separate_arguments(TORCH_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
add_library(xinfer_attention_ops SHARED common/attention.cpp)
set_target_properties(xinfer_attention_ops PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_features(xinfer_attention_ops PRIVATE cxx_std_${QWEN_CXX_STANDARD})
target_compile_options(xinfer_attention_ops PRIVATE ${TORCH_FLAGS} -Wall -Wextra)
target_link_libraries(xinfer_attention_ops PRIVATE ${TORCH_LIBRARIES})
add_subdirectory(qwen3)

option(QWEN_BUILD_DELTANET "Build the isolated Qwen3.5 DeltaNet probe" OFF)
option(QWEN_BUILD_QWEN35 "Build the Qwen3.5 text generation runtime" OFF)
if(QWEN_BUILD_DELTANET OR QWEN_BUILD_QWEN35)
  add_subdirectory(qwen35)
endif()
if(QWEN_BUILD_VALIDATION AND QWEN_BUILD_QWEN35)
  add_executable(qwen_runtime_contract common/tests/runtime_contract.cpp)
  target_include_directories(qwen_runtime_contract PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_features(qwen_runtime_contract PRIVATE cxx_std_${QWEN_CXX_STANDARD})
  target_compile_options(qwen_runtime_contract PRIVATE ${TORCH_FLAGS} -Wall -Wextra)
  target_link_libraries(qwen_runtime_contract PRIVATE ${TORCH_LIBRARIES} nlohmann_json::nlohmann_json xinfer_attention_ops)
endif()
