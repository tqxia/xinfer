// Execute different token lengths through one AOTI package, without Python.
#include "../../common/io.h"
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/version.h>

#include <iostream>

extern "C" void xinfer_delta_ops_link();
using namespace xinfer;

at::Tensor read_tensor(std::istream& file, const std::vector<int64_t>& shape, at::ScalarType dtype) {
  auto cpu = at::empty(shape, at::TensorOptions().dtype(dtype));
  file.read(static_cast<char*>(cpu.mutable_data_ptr()), cpu.nbytes());
  require(bool(file), "Truncated input fixture");
  return cpu.to(at::kCUDA);
}

int main(int argc, char** argv) {
  try {
    require(argc == 4, "Usage: qwen35_dynamic_block PACKAGE_DIR FIXTURE_DIR OUTPUT_DIR");
    const fs::path packages(argv[1]), fixtures(argv[2]), output(argv[3]);
    require(!fs::exists(output), "Output directory already exists");
    const auto meta = read_json(packages / "prefill.json");
    const auto version = meta.at("torch").get<std::string>();
    require(version.substr(0, version.find('+')) == TORCH_VERSION, "Torch version mismatch");
    require(meta.at("cxx11_abi") == bool(_GLIBCXX_USE_CXX11_ABI), "C++ ABI mismatch");
    auto hidden_shape = meta.at("inputs").at(0).at("shape").get<std::vector<int64_t>>();
    const auto conv_shape = meta.at("inputs").at(1).at("shape").get<std::vector<int64_t>>();
    const auto state_shape = meta.at("inputs").at(2).at("shape").get<std::vector<int64_t>>();
    c10::InferenceMode inference;
    at::set_num_threads(1);
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    xinfer_delta_ops_link();
    torch::inductor::AOTIModelPackageLoader loader((packages / "prefill.pt2").string());
    const auto manifest = read_json(fixtures / "cases.json");
    Json report = {{"cases", Json::array()}, {"package_loads", 1}};
    for (const auto& item : manifest.at("cases")) {
      const auto name = item.at("name").get<std::string>();
      require(!name.empty() && fs::path(name).filename() == name, "Invalid case name");
      std::ifstream initial(fixtures / (name + "-state.bin"), std::ios::binary);
      auto conv = read_tensor(initial, conv_shape, at::kBFloat16);
      auto state = read_tensor(initial, state_shape, at::kFloat);
      require(initial.peek() == EOF, "Extra state fixture data");
      int step = 0;
      for (const auto& length : item.at("lengths")) {
        const auto tokens = length.get<int64_t>();
        require(tokens >= meta.at("min_tokens").get<int64_t>() &&
                tokens <= meta.at("max_tokens").get<int64_t>(), "Token length outside exported range");
        hidden_shape[1] = tokens;
        const auto stem = name + "-" + std::to_string(step++);
        std::ifstream file(fixtures / (stem + ".bin"), std::ios::binary);
        auto hidden = read_tensor(file, hidden_shape, at::kBFloat16);
        require(file.peek() == EOF, "Extra hidden fixture data");
        const std::vector<at::Tensor> inputs{hidden, conv, state};
        const std::vector<at::Tensor> before{hidden.clone(), conv.clone(), state.clone()};
        auto result = loader.run(inputs);
        require(result.size() == 3, "Expected output and two states");
        auto destination = output_file(output / (stem + ".bin"), true);
        for (size_t i = 0; i < result.size(); ++i) {
          require(at::equal(inputs[i], before[i]), "Package mutated an input");
          require(result[i].sizes() == inputs[i].sizes() && result[i].scalar_type() == inputs[i].scalar_type(),
                  "Output shape/dtype mismatch");
          require(at::isfinite(result[i]).all().item<bool>(), "Non-finite output/state");
          auto cpu = result[i].to(at::kCPU).contiguous();
          destination.write(static_cast<const char*>(cpu.const_data_ptr()), cpu.nbytes());
        }
        require(bool(destination), "Failed to write native results");
        conv = result[1];
        state = result[2];
      }
      report["cases"].push_back({{"name", name}, {"steps", step}, {"inputs_unchanged", true}});
    }
    std::ifstream maps("/proc/self/maps");
    for (std::string line; std::getline(maps, line);)
      require(line.find("libpython") == std::string::npos && line.find("libtorch_python") == std::string::npos,
              "Python library loaded");
    report["no_python_library"] = true;
    output_file(output / "native.json") << report.dump(2) << '\n';
    std::cout << report.dump(2) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
