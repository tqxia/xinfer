include(GNUInstallDirs)

if(IS_ABSOLUTE "${CMAKE_INSTALL_BINDIR}")
  message(FATAL_ERROR "Use relative CMAKE_INSTALL_BINDIR for runtime relocation")
endif()
# Native stacks may live inside the source tree, where CMake's automatic
# INSTALL_RPATH_USE_LINK_PATH filtering would otherwise discard their paths.
set(XINFER_DEFAULT_INSTALL_RUNTIME_DIRS "${TORCH_INSTALL_PREFIX}/lib")
foreach(cuda_runtime IN ITEMS CUDA_cudart_LIBRARY CUDA_CUDART_LIBRARY)
  if(IS_ABSOLUTE "${${cuda_runtime}}")
    get_filename_component(cuda_runtime_dir "${${cuda_runtime}}" DIRECTORY)
    list(APPEND XINFER_DEFAULT_INSTALL_RUNTIME_DIRS "${cuda_runtime_dir}")
  endif()
endforeach()
list(REMOVE_DUPLICATES XINFER_DEFAULT_INSTALL_RUNTIME_DIRS)
set(XINFER_INSTALL_RUNTIME_DIRS "${XINFER_DEFAULT_INSTALL_RUNTIME_DIRS}" CACHE STRING
    "External LibTorch/CUDA library directories searched by installed runtimes")

# Header-only JSON parsing; pinned release and checksum, no runtime dependency.
include(FetchContent)
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
FetchContent_MakeAvailable(nlohmann_json)

option(XINFER_BUILD_VALIDATION "Build native validation drivers" OFF)
option(XINFER_BUILD_BENCHMARK "Build fixed-token native benchmarks" OFF)
add_executable(xinfer_aoti main.cpp)
set(XINFER_TARGETS xinfer_aoti)
set(XINFER_INSTALL_TARGETS xinfer_aoti)
if(XINFER_BUILD_BENCHMARK)
  add_executable(xinfer_benchmark benchmark.cpp)
  list(APPEND XINFER_TARGETS xinfer_benchmark)
  list(APPEND XINFER_INSTALL_TARGETS xinfer_benchmark)
endif()
if(XINFER_BUILD_VALIDATION)
  add_executable(xinfer_validate validate.cpp)
  add_executable(qwen_runtime_contract common/tests/runtime_contract.cpp)
  add_executable(xinfer_state_contract common/tests/state.cpp)
  add_executable(xinfer_generation_contract common/tests/generation.cpp)
  add_executable(qwen_megaqwen_package_contract qwen3/tests/megaqwen_package.cpp)
  list(APPEND XINFER_TARGETS xinfer_validate qwen_runtime_contract xinfer_state_contract xinfer_generation_contract qwen_megaqwen_package_contract)
  list(APPEND XINFER_INSTALL_TARGETS xinfer_validate)
  if(XINFER_ENABLE_MEGAQWEN)
    add_executable(qwen_attention_contract qwen3/tests/attention.cpp qwen3/tests/attention.cu)
    add_executable(qwen_megaqwen_op_contract qwen3/tests/megaqwen_op.cpp)
    set_target_properties(qwen_attention_contract PROPERTIES
      CUDA_ARCHITECTURES "${XINFER_MEGAQWEN_CUDA_ARCHITECTURES}"
      CUDA_STANDARD ${XINFER_CXX_STANDARD}
      CUDA_STANDARD_REQUIRED ON
      CUDA_RUNTIME_LIBRARY None)
    target_compile_options(qwen_attention_contract PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false>")
    target_link_libraries(qwen_attention_contract PRIVATE xinfer_megaqwen_core)
    # The prototype package supplies the op test's dispatcher registration.
    add_dependencies(qwen_megaqwen_op_contract xinfer_megaqwen_ops)
    list(APPEND XINFER_TARGETS qwen_attention_contract qwen_megaqwen_op_contract)
  endif()
endif()
foreach(target IN LISTS XINFER_TARGETS)
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_features(${target} PRIVATE cxx_std_${XINFER_CXX_STANDARD})
  target_compile_options(${target} PRIVATE ${TORCH_FLAGS} "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>")
  target_link_libraries(${target} PRIVATE ${TORCH_LIBRARIES} nlohmann_json::nlohmann_json)
  set_target_properties(${target} PROPERTIES BUILD_RPATH_USE_ORIGIN TRUE)
endforeach()

foreach(target IN LISTS XINFER_INSTALL_TARGETS)
  set_target_properties(${target} PROPERTIES
    INSTALL_RPATH "$ORIGIN;${XINFER_INSTALL_RUNTIME_DIRS}"
    INSTALL_RPATH_USE_LINK_PATH TRUE)
endforeach()
install(TARGETS ${XINFER_INSTALL_TARGETS}
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT Runtime)
