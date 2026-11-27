include(GNUInstallDirs)

if(IS_ABSOLUTE "${CMAKE_INSTALL_BINDIR}")
  message(FATAL_ERROR "Use relative CMAKE_INSTALL_BINDIR for runtime relocation")
endif()

find_program(XINFER_CARGO cargo REQUIRED)

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

function(xinfer_add_rust_shim target source)
  add_library(${target} STATIC ${source})
  target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/runtime/core")
  target_compile_features(${target} PRIVATE cxx_std_${XINFER_CXX_STANDARD})
  target_compile_options(${target} PRIVATE ${TORCH_FLAGS} -Wall -Wextra -fPIC)
  target_link_libraries(${target} PRIVATE ${TORCH_LIBRARIES})
  set_target_properties(${target} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endfunction()

xinfer_add_rust_shim(xinfer_torch_shim core/shim/torch_shim.cpp)

set(_xinfer_cudart "")
foreach(_cand IN ITEMS CUDA_cudart_LIBRARY CUDA_CUDART_LIBRARY)
  if(IS_ABSOLUTE "${${_cand}}")
    set(_xinfer_cudart "${${_cand}}")
    break()
  endif()
endforeach()
if("${_xinfer_cudart}" STREQUAL "" AND DEFINED CUDAToolkit_LIBRARY_DIR)
  set(_xinfer_cudart "${CUDAToolkit_LIBRARY_DIR}/libcudart.so")
endif()
if("${_xinfer_cudart}" STREQUAL "")
  message(FATAL_ERROR "Unable to resolve libcudart library path for Rust linking")
endif()

set(_xinfer_rpath_list "$ORIGIN" ${XINFER_INSTALL_RUNTIME_DIRS})
list(REMOVE_DUPLICATES _xinfer_rpath_list)
string(JOIN ":" _xinfer_rpath ${_xinfer_rpath_list})

file(GLOB _xinfer_nvidia_dirs LIST_DIRECTORIES true "${TORCH_INSTALL_PREFIX}/../nvidia/*/lib")
set(_xinfer_rpath_link_list)
foreach(_dir IN LISTS _xinfer_nvidia_dirs)
  if(IS_DIRECTORY "${_dir}")
    list(APPEND _xinfer_rpath_link_list "${_dir}")
  endif()
endforeach()
get_filename_component(_xinfer_cudart_dir "${_xinfer_cudart}" DIRECTORY)
list(APPEND _xinfer_rpath_link_list "${TORCH_INSTALL_PREFIX}/lib" "${_xinfer_cudart_dir}")
list(REMOVE_DUPLICATES _xinfer_rpath_link_list)
string(JOIN ":" _xinfer_rpath_link ${_xinfer_rpath_link_list})

set(XINFER_RUST_LINK_CONFIG "${CMAKE_BINARY_DIR}/xinfer_rust_link.env" CACHE INTERNAL
    "Link configuration consumed by Cargo build scripts")
file(WRITE "${XINFER_RUST_LINK_CONFIG}"
  "SHIM_DIR=${CMAKE_BINARY_DIR}\n"
  "TORCH_LIB_DIR=${TORCH_INSTALL_PREFIX}/lib\n"
  "CUDART_LIB=${_xinfer_cudart}\n"
  "RPATH=${_xinfer_rpath}\n"
  "RPATH_LINK=${_xinfer_rpath_link}\n")

function(xinfer_add_rust_executable target package)
  set(_deps xinfer_torch_shim ${ARGN})
  set(_cargo_args build
    --manifest-path "${PROJECT_SOURCE_DIR}/Cargo.toml"
    --locked --release
    --package "${package}"
    --bin "${target}")
  add_custom_target(${target} ALL
    COMMAND ${CMAKE_COMMAND} -E env
      "XINFER_LINK_CONFIG=${XINFER_RUST_LINK_CONFIG}"
      "CARGO_TARGET_DIR=${CMAKE_BINARY_DIR}/rust"
      "${XINFER_CARGO}" ${_cargo_args}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_BINARY_DIR}/rust/release/${target}" "${CMAKE_BINARY_DIR}/${target}"
    DEPENDS ${_deps}
    VERBATIM)
  install(PROGRAMS "${CMAKE_BINARY_DIR}/${target}"
    DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT Runtime)
endfunction()

xinfer_add_rust_executable(xinfer_aoti xinfer-runtime)
if(XINFER_BUILD_SERVER)
  xinfer_add_rust_executable(xinfer_serve xinfer-server)
endif()
