add_library(xinfer_attention_ops SHARED attention.cpp attention_shim.cpp)
set_target_properties(xinfer_attention_ops PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH_USE_ORIGIN TRUE)
target_compile_features(xinfer_attention_ops PRIVATE cxx_std_${XINFER_CXX_STANDARD})
target_compile_options(xinfer_attention_ops PRIVATE ${TORCH_FLAGS} -Wall -Wextra)
target_link_libraries(xinfer_attention_ops PRIVATE ${TORCH_LIBRARIES})
