add_executable(qwen35_block block_main.cpp)
target_compile_features(qwen35_block PRIVATE cxx_std_17)
target_compile_options(qwen35_block PRIVATE ${TORCH_FLAGS} -Wall -Wextra)
target_link_libraries(qwen35_block PRIVATE ${TORCH_LIBRARIES} nlohmann_json::nlohmann_json)
