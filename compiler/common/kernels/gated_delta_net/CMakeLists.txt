add_library(xinfer_deltanet_ops SHARED deltanet.cu deltanet.cpp deltanet_shim.cpp)
set_target_properties(xinfer_deltanet_ops PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH_USE_ORIGIN TRUE
  CUDA_ARCHITECTURES OFF)
target_compile_features(xinfer_deltanet_ops PRIVATE cxx_std_${XINFER_CXX_STANDARD})
target_compile_options(xinfer_deltanet_ops PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:${TORCH_FLAGS} -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_link_libraries(xinfer_deltanet_ops PRIVATE ${TORCH_LIBRARIES})
