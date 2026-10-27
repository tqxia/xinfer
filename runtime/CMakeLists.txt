find_package(Torch 2.7 REQUIRED)

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
option(QWEN_BUILD_PROFILE "Build the Qwen3 NVTX profiling driver" OFF)
option(QWEN_ENABLE_FLASH_ATTENTION "Build experimental Qwen3 valid-prefix FlashAttention" OFF)
separate_arguments(TORCH_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
add_subdirectory(qwen3)

option(QWEN_BUILD_DELTANET "Build the isolated Qwen3.5 DeltaNet probe" OFF)
if(QWEN_BUILD_DELTANET)
  add_subdirectory(qwen35)
endif()
