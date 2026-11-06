find_package(Torch 2.13 REQUIRED)
include(GNUInstallDirs)

if(IS_ABSOLUTE "${CMAKE_INSTALL_BINDIR}" OR IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
  message(FATAL_ERROR "Use relative CMAKE_INSTALL_BINDIR and CMAKE_INSTALL_LIBDIR for runtime relocation")
endif()
file(RELATIVE_PATH XINFER_INSTALL_LIB_FROM_BIN "${CMAKE_INSTALL_FULL_BINDIR}" "${CMAKE_INSTALL_FULL_LIBDIR}")

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

# Match the installed LibTorch headers, including C++20 on the GB10 toolchain.
get_target_property(XINFER_CXX_STANDARD torch CXX_STANDARD)
if(NOT XINFER_CXX_STANDARD)
  set(XINFER_CXX_STANDARD 17)
endif()

# Header-only JSON parsing; pinned release and checksum, no runtime dependency.
include(FetchContent)
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
FetchContent_MakeAvailable(nlohmann_json)

option(XINFER_BUILD_VALIDATION "Build native validation drivers" OFF)
option(XINFER_BUILD_BENCHMARK "Build fixed-token native benchmarks" OFF)
option(XINFER_ENABLE_MEGAQWEN "Enable optional cooperative Qwen3-0.6B decode" OFF)
separate_arguments(TORCH_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
if(XINFER_ENABLE_MEGAQWEN)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(XINFER_MEGAQWEN_CUDA_ARCHITECTURES "89" CACHE STRING
      "CUDA architectures for MegaQwen (SM89 is the validated target)")
  add_library(xinfer_decode_backend SHARED qwen3/megaqwen/backend.cpp qwen3/megaqwen/kernels.cu)
  set_target_properties(xinfer_decode_backend PROPERTIES
    OUTPUT_NAME xinfer_megaqwen
    CUDA_ARCHITECTURES "${XINFER_MEGAQWEN_CUDA_ARCHITECTURES}"
    CUDA_STANDARD ${XINFER_CXX_STANDARD}
    CUDA_STANDARD_REQUIRED ON
    CUDA_RUNTIME_LIBRARY None
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH "$ORIGIN;${XINFER_INSTALL_RUNTIME_DIRS}"
    INSTALL_RPATH_USE_LINK_PATH TRUE)
  target_include_directories(xinfer_decode_backend PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_compile_options(xinfer_decode_backend PRIVATE ${TORCH_FLAGS}
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false>")
  if(CUDA_cudart_LIBRARY)
    target_link_libraries(xinfer_decode_backend PRIVATE "${CUDA_cudart_LIBRARY}")
  else()
    target_link_libraries(xinfer_decode_backend PRIVATE CUDA::cudart)
  endif()
else()
  add_library(xinfer_decode_backend STATIC qwen3/megaqwen/stub.cpp)
  target_compile_options(xinfer_decode_backend PRIVATE ${TORCH_FLAGS} -Wall -Wextra)
endif()
target_compile_features(xinfer_decode_backend PRIVATE cxx_std_${XINFER_CXX_STANDARD})
target_link_libraries(xinfer_decode_backend PRIVATE ${TORCH_LIBRARIES} nlohmann_json::nlohmann_json)
add_library(xinfer_attention_ops SHARED common/attention.cpp)
set_target_properties(xinfer_attention_ops PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH "$ORIGIN;${XINFER_INSTALL_RUNTIME_DIRS}"
  INSTALL_RPATH_USE_LINK_PATH TRUE)
target_compile_features(xinfer_attention_ops PRIVATE cxx_std_${XINFER_CXX_STANDARD})
target_compile_options(xinfer_attention_ops PRIVATE ${TORCH_FLAGS} -Wall -Wextra)
target_link_libraries(xinfer_attention_ops PRIVATE ${TORCH_LIBRARIES})
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
  list(APPEND XINFER_TARGETS xinfer_validate qwen_runtime_contract xinfer_state_contract xinfer_generation_contract)
  list(APPEND XINFER_INSTALL_TARGETS xinfer_validate)
  if(XINFER_ENABLE_MEGAQWEN)
    add_executable(qwen_hybrid_contract qwen3/tests/hybrid.cpp)
    add_executable(qwen_attention_contract qwen3/tests/attention.cpp)
    target_compile_definitions(xinfer_decode_backend PRIVATE XINFER_MEGAQWEN_TESTING)
    target_compile_definitions(qwen_attention_contract PRIVATE XINFER_MEGAQWEN_TESTING)
    list(APPEND XINFER_TARGETS qwen_hybrid_contract qwen_attention_contract)
  endif()
endif()
foreach(target IN LISTS XINFER_TARGETS)
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_features(${target} PRIVATE cxx_std_${XINFER_CXX_STANDARD})
  target_compile_options(${target} PRIVATE ${TORCH_FLAGS} -Wall -Wextra)
  target_link_libraries(${target} PRIVATE ${TORCH_LIBRARIES} nlohmann_json::nlohmann_json
    xinfer_attention_ops xinfer_decode_backend)
  set_target_properties(${target} PROPERTIES BUILD_RPATH_USE_ORIGIN TRUE)
endforeach()

foreach(target IN LISTS XINFER_INSTALL_TARGETS)
  set_target_properties(${target} PROPERTIES
    INSTALL_RPATH "$ORIGIN/${XINFER_INSTALL_LIB_FROM_BIN};${XINFER_INSTALL_RUNTIME_DIRS}"
    INSTALL_RPATH_USE_LINK_PATH TRUE)
endforeach()
install(TARGETS ${XINFER_INSTALL_TARGETS} xinfer_attention_ops
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT Runtime
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Runtime)
if(XINFER_ENABLE_MEGAQWEN)
  install(TARGETS xinfer_decode_backend
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Runtime)
endif()
