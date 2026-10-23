// DeltaNet fixture runner. State crosses package calls as ordinary LibTorch tensors.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/version.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;
using Json = nlohmann::json;
using Loader = torch::inductor::AOTIModelPackageLoader;

void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}

Json read_json(const fs::path& path) {
  std::ifstream file(path);
  require(file.is_open(), "Cannot open " + path.string());
  return Json::parse(file);
}

at::Tensor read_hidden(const fs::path& path, int64_t tokens) {
  auto cpu = at::empty({1, tokens, 2560}, at::TensorOptions().dtype(at::kBFloat16));
  require(fs::file_size(path) == cpu.nbytes(), "Incorrect fixture size: " + path.string());
  std::ifstream file(path, std::ios::binary);
  file.read(static_cast<char*>(cpu.mutable_data_ptr()), cpu.nbytes());
  require(bool(file), "Failed to read " + path.string());
  return cpu.to(at::kCUDA);
}

void write_tensor(const fs::path& path, const at::Tensor& tensor) {
  auto cpu = tensor.to(at::kCPU).contiguous();
  std::ofstream file(path, std::ios::binary);
  file.write(static_cast<const char*>(cpu.const_data_ptr()), cpu.nbytes());
  file.close();
  require(bool(file), "Failed to write " + path.string());
}

int main(int argc, char** argv) {
  try {
    require(argc == 4, "Usage: qwen35_block PACKAGE_DIR FIXTURE_DIR OUTPUT_DIR");
    const fs::path packages(argv[1]), fixtures(argv[2]), output(argv[3]);
    require(!fs::exists(output), "Output directory already exists");
    const auto prefill_meta = read_json(packages / "prefill.json");
    const auto decode_meta = read_json(packages / "decode.json");
    for (const auto* key : {"checkpoint", "torch", "cuda", "cxx11_abi", "sm", "hf_source_sha256",
                           "source_sha256", "parameter_dtypes", "inductor_configs", "kernel_path", "tf32"})
      require(prefill_meta.at(key) == decode_meta.at(key), std::string("Package metadata differs: ") + key);
    for (const auto* kind : {"prefill", "decode"}) {
      const auto meta = read_json(packages / (std::string(kind) + ".json"));
      const auto version = meta.at("torch").get<std::string>();
      require(version.substr(0, version.find('+')) == TORCH_VERSION, "Torch version mismatch");
      require(meta.at("cxx11_abi") == bool(_GLIBCXX_USE_CXX11_ABI), "C++ ABI mismatch");
      require(meta.at("kind") == kind && meta.at("tokens") == (std::string(kind) == "prefill" ? 64 : 1),
              "Unexpected entry point");
    }
    c10::InferenceMode inference;
    at::set_num_threads(1);
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    Loader prefill((packages / "prefill.pt2").string());
    Loader decode((packages / "decode.pt2").string());
    fs::create_directories(output);
    const auto manifest = read_json(fixtures / "cases.json");
    const auto bf16 = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
    Json report = {{"torch", TORCH_VERSION}, {"cxx11_abi", bool(_GLIBCXX_USE_CXX11_ABI)},
                   {"cases", Json::array()}};
    for (const auto& item : manifest.at("cases")) {
      const auto name = item.at("name").get<std::string>();
      require(!name.empty() && fs::path(name).filename() == name, "Invalid case name");
      auto conv = at::zeros({1, 8192, 4}, bf16);
      auto state = at::zeros({1, 32, 128, 128}, bf16.dtype(at::kFloat));
      size_t index = 0;
      for (const auto& step : item.at("steps")) {
        const auto kind = step.at("kind").get<std::string>();
        require(kind == "prefill" || kind == "decode", "Unknown entry point");
        auto x = read_hidden(fixtures / step.at("file").get<std::string>(), kind == "prefill" ? 64 : 1);
        const std::vector<at::Tensor> inputs{x, conv, state};
        const std::vector<at::Tensor> before{x.clone(), conv.clone(), state.clone()};
        auto result = (kind == "prefill" ? prefill : decode).run(inputs);
        require(result.size() == 3, "Expected output, next_conv, next_recurrent");
        for (size_t i = 0; i < inputs.size(); ++i) {
          require(at::equal(inputs[i], before[i]), "Package mutated an input");
          require(result[i].sizes() == inputs[i].sizes() && result[i].scalar_type() == inputs[i].scalar_type(),
                  "Output shape/dtype mismatch");
          require(at::isfinite(result[i]).all().item<bool>(), "Non-finite output/state");
          write_tensor(output / (name + "-" + std::to_string(index) + "-" + std::to_string(i) + ".bin"), result[i]);
        }
        conv = result[1];
        state = result[2];
        ++index;
      }
      report["cases"].push_back({{"name", name}, {"steps", index}, {"inputs_unchanged", true}});
    }
    std::ifstream maps("/proc/self/maps");
    for (std::string line; std::getline(maps, line);)
      require(line.find("libpython") == std::string::npos, "Python library loaded");
    report["no_python_library"] = true;
    std::ofstream file(output / "native.json");
    file << report.dump(2) << '\n';
    file.close();
    require(bool(file), "Failed to write native report");
    std::cout << report.dump(2) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
