add_executable(qwen35_fused tests/fused.cpp)
target_compile_features(qwen35_fused PRIVATE cxx_std_${XINFER_CXX_STANDARD})
target_compile_options(qwen35_fused PRIVATE ${TORCH_FLAGS} -Wall -Wextra)
target_link_libraries(qwen35_fused PRIVATE ${TORCH_LIBRARIES} nlohmann_json::nlohmann_json)
