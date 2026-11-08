enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
set(XINFER_MEGAQWEN_CUDA_ARCHITECTURES "89" CACHE STRING
    "CUDA architectures for MegaQwen (SM89 is the validated target)")

# Compile decode kernels for the operator library. Attention tests include the
# same device arithmetic from attention.cuh in their own test translation unit.
# Runtime executables load the operator through their compiled packages.
add_library(xinfer_megaqwen_core STATIC kernels.cu)
set_target_properties(xinfer_megaqwen_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_ARCHITECTURES "${XINFER_MEGAQWEN_CUDA_ARCHITECTURES}"
  CUDA_STANDARD ${XINFER_CXX_STANDARD}
  CUDA_STANDARD_REQUIRED ON
  CUDA_RUNTIME_LIBRARY None)
target_include_directories(xinfer_megaqwen_core PUBLIC "${XINFER_COMPILER_ROOT}"
  ${CUDAToolkit_INCLUDE_DIRS})
target_compile_options(xinfer_megaqwen_core PRIVATE ${TORCH_FLAGS}
  "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false>")
target_link_libraries(xinfer_megaqwen_core PUBLIC ${TORCH_LIBRARIES})
if(CUDA_cudart_LIBRARY)
  target_link_libraries(xinfer_megaqwen_core PUBLIC "${CUDA_cudart_LIBRARY}")
else()
  target_link_libraries(xinfer_megaqwen_core PUBLIC CUDA::cudart)
endif()

add_library(xinfer_megaqwen_ops SHARED op.cpp shim.cpp)
set_target_properties(xinfer_megaqwen_ops PROPERTIES
  OUTPUT_NAME xinfer_megaqwen
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH_USE_ORIGIN TRUE)
target_include_directories(xinfer_megaqwen_ops PUBLIC "${XINFER_COMPILER_ROOT}")
target_compile_features(xinfer_megaqwen_ops PRIVATE cxx_std_${XINFER_CXX_STANDARD})
target_compile_options(xinfer_megaqwen_ops PRIVATE ${TORCH_FLAGS} -Wall -Wextra)
target_link_libraries(xinfer_megaqwen_ops PRIVATE xinfer_megaqwen_core)
