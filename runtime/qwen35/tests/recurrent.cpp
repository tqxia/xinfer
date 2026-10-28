// Run packaged recurrent Triton fixtures with no Python runtime or custom-op library.
#include "../../common/io.h"
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <iostream>
#include <map>

using namespace xinfer;
using Tensor = at::Tensor;

Tensor read_tensor(std::istream& file, const std::vector<int64_t>& shape, at::ScalarType dtype) {
  auto cpu = at::empty(shape, at::TensorOptions().dtype(dtype));
  file.read(static_cast<char*>(cpu.mutable_data_ptr()), cpu.nbytes());
  require(bool(file), "Truncated fixture");
  return cpu.to(at::kCUDA);
}

int main(int argc, char** argv) {
  try {
    require(argc == 4, "Usage: qwen35_recurrent FIXTURES OUTPUT PACKAGES");
    const fs::path fixtures(argv[1]), output(argv[2]);
    require(!fs::exists(output), "Output directory already exists");
    c10::InferenceMode inference;
    at::set_num_threads(1);
    std::map<std::string, std::unique_ptr<torch::inductor::AOTIModelPackageLoader>> packages;
    for (const std::string dtype : {"bf16", "fp32"})
      for (const std::string mode : {"decode", "prefill"}) {
        const auto name = dtype + "-" + mode;
        packages[name] = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
            (fs::path(argv[3]) / (name + ".pt2")).string());
      }
    Json report = {{"calls", 0}, {"packages_loaded", packages.size()}, {"inputs_unchanged", true}};
    const auto manifest = read_json(fixtures / "cases.json");
    for (const auto& item : manifest.at("cases")) {
      const auto name = item.at("name").get<std::string>();
      require(!name.empty() && fs::path(name).filename() == name, "Invalid fixture name");
      const auto dtype_name = item.at("dtype").get<std::string>();
      require(dtype_name == "bf16" || dtype_name == "fp32", "Invalid fixture dtype");
      const auto dtype = dtype_name == "bf16" ? at::kBFloat16 : at::kFloat;
      std::ifstream initial(fixtures / (name + "-state.bin"), std::ios::binary);
      auto state = read_tensor(initial, {1, 16, 128, 128}, at::kFloat);
      require(initial.peek() == EOF, "Extra initial state data");
      int step = 0;
      for (const auto& length : item.at("lengths")) {
        const auto tokens = length.get<int64_t>();
        require(tokens >= 1 && tokens <= 512, "Token length must be 1..512");
        const auto stem = name + "-" + std::to_string(step++);
        std::ifstream file(fixtures / (stem + ".bin"), std::ios::binary);
        std::vector<Tensor> inputs;
        for (int i = 0; i < 3; ++i) inputs.push_back(read_tensor(file, {1, tokens, 16, 128}, dtype));
        inputs.push_back(read_tensor(file, {1, tokens, 16}, at::kFloat));
        inputs.push_back(read_tensor(file, {1, tokens, 16}, dtype));
        inputs.push_back(state);
        require(file.peek() == EOF, "Extra fixture data");
        std::vector<Tensor> before;
        for (const auto& x : inputs) before.push_back(x.clone());
        const auto result = packages.at(dtype_name + (tokens == 1 ? "-decode" : "-prefill"))->run(inputs);
        for (size_t i = 0; i < inputs.size(); ++i)
          require(at::equal(inputs[i], before[i]), "Input mutation");
        require(result.size() == 2 && result[0].sizes() == inputs[0].sizes() &&
                result[0].scalar_type() == dtype && result[1].sizes() == state.sizes() &&
                result[1].scalar_type() == at::kFloat, "Output shape/dtype mismatch");
        auto destination = output_file(output / (stem + ".bin"), true);
        for (const auto& x : result) {
          require(at::isfinite(x).all().item<bool>(), "Non-finite result");
          auto cpu = x.cpu().contiguous();
          destination.write(static_cast<const char*>(cpu.const_data_ptr()), cpu.nbytes());
        }
        require(bool(destination), "Failed to write result");
        state = result[1];
        report["calls"] = report["calls"].get<int>() + 1;
      }
    }
    std::ifstream maps("/proc/self/maps");
    for (std::string line; std::getline(maps, line);) {
      require(line.find("libpython") == std::string::npos && line.find("libtorch_python") == std::string::npos,
              "Python library loaded");
      require(line.find("libxinfer_delta_ops") == std::string::npos &&
              line.find("libxinfer_recurrent_ops") == std::string::npos, "Custom-op library loaded");
    }
    report["no_python_library"] = true;
    report["no_custom_op_library"] = true;
    output_file(output / "native.json") << report.dump(2) << '\n';
    std::cout << report.dump(2) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
