cmake_minimum_required(VERSION 3.24)
project(qwen_aoti LANGUAGES CXX)

find_package(Torch 2.7 REQUIRED)

# Header-only JSON parsing; pinned release and checksum, no runtime dependency.
include(FetchContent)
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
FetchContent_MakeAvailable(nlohmann_json)

add_executable(qwen_aoti main.cpp)
target_compile_features(qwen_aoti PRIVATE cxx_std_17)
separate_arguments(TORCH_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(qwen_aoti PRIVATE ${TORCH_FLAGS} -Wall -Wextra)
target_link_libraries(qwen_aoti PRIVATE ${TORCH_LIBRARIES} nlohmann_json::nlohmann_json)
