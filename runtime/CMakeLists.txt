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
set_target_properties(xinfer_attention_ops PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH "$ORIGIN;${XINFER_INSTALL_RUNTIME_DIRS}"
  INSTALL_RPATH_USE_LINK_PATH TRUE)
target_compile_features(xinfer_attention_ops PRIVATE cxx_std_${QWEN_CXX_STANDARD})
target_compile_options(xinfer_attention_ops PRIVATE ${TORCH_FLAGS} -Wall -Wextra)
target_link_libraries(xinfer_attention_ops PRIVATE ${TORCH_LIBRARIES})
add_executable(xinfer_aoti main.cpp)
set(XINFER_TARGETS xinfer_aoti)
set(XINFER_INSTALL_TARGETS xinfer_aoti)
if(QWEN_BUILD_BENCHMARK)
  add_executable(xinfer_benchmark benchmark.cpp)
  list(APPEND XINFER_TARGETS xinfer_benchmark)
  list(APPEND XINFER_INSTALL_TARGETS xinfer_benchmark)
endif()
if(QWEN_BUILD_VALIDATION)
  add_executable(xinfer_validate validate.cpp)
  add_executable(qwen_runtime_contract common/tests/runtime_contract.cpp)
  list(APPEND XINFER_TARGETS xinfer_validate qwen_runtime_contract)
  list(APPEND XINFER_INSTALL_TARGETS xinfer_validate)
endif()
foreach(target IN LISTS XINFER_TARGETS)
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_features(${target} PRIVATE cxx_std_${QWEN_CXX_STANDARD})
  target_compile_options(${target} PRIVATE ${TORCH_FLAGS} -Wall -Wextra)
  target_link_libraries(${target} PRIVATE ${TORCH_LIBRARIES} nlohmann_json::nlohmann_json xinfer_attention_ops)
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

if(QWEN_BUILD_VALIDATION)
  add_subdirectory(qwen35)
endif()
