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
#include <limits>
#include <stdexcept>
#include <utility>
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

// Package inputs are contiguous: hidden BF16, convolution BF16, recurrent FP32.
std::vector<std::vector<int64_t>> input_shapes(const Json& meta) {
  const auto& inputs = meta.at("inputs");
  require(inputs.is_array() && inputs.size() == 3, "Expected three input descriptions");
  std::vector<std::vector<int64_t>> shapes;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto shape = inputs[i].at("shape").get<std::vector<int64_t>>();
    const auto stride = inputs[i].at("stride").get<std::vector<int64_t>>();
    require(shape.size() == (i == 2 ? 4 : 3) && shape[0] == 1, "Unexpected input rank/batch");
    require(stride.size() == shape.size(), "Unexpected input stride rank");
    require(inputs[i].at("dtype") == (i == 2 ? "torch.float32" : "torch.bfloat16"),
            "Unexpected input dtype");
    int64_t contiguous = 1;
    for (size_t d = shape.size(); d-- > 0;) {
      require(shape[d] > 0 && stride[d] == contiguous, "Expected positive contiguous input shape");
      require(shape[d] <= std::numeric_limits<int64_t>::max() / contiguous, "Input shape overflow");
      contiguous *= shape[d];
    }
    shapes.push_back(std::move(shape));
  }
  require(meta.at("tokens") == shapes[0][1], "Input tokens disagree with metadata");
  return shapes;
}

at::Tensor read_hidden(const fs::path& path, const std::vector<int64_t>& shape) {
  auto cpu = at::empty(shape, at::TensorOptions().dtype(at::kBFloat16));
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

// Never pad hidden tokens: padding can change the convolution and recurrent state.
std::vector<at::Tensor> run_sequence(Loader& prefill, Loader& decode, const at::Tensor& hidden,
                                    at::Tensor conv, at::Tensor state) {
  std::vector<at::Tensor> outputs;
  for (int64_t offset = 0; offset < hidden.size(1);) {
    const int64_t length = hidden.size(1) - offset >= 64 ? 64 : 1;
    auto piece = hidden.narrow(1, offset, length).contiguous();
    auto result = (length == 64 ? prefill : decode).run({piece, conv, state});
    require(result.size() == 3, "Expected output, next_conv, next_recurrent");
    outputs.push_back(result[0]);
    conv = result[1];
    state = result[2];
    offset += length;
  }
  return {outputs.size() == 1 ? outputs[0] : at::cat(outputs, 1), conv, state};
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
    const auto prefill_shapes = input_shapes(prefill_meta);
    const auto decode_shapes = input_shapes(decode_meta);
    require(prefill_shapes[0][2] == decode_shapes[0][2], "Package hidden dimensions differ");
    require(prefill_shapes[1] == decode_shapes[1] && prefill_shapes[2] == decode_shapes[2],
            "Package state shapes differ");
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
                   {"checkpoint", prefill_meta.at("checkpoint")},
                   {"prefill_shapes", prefill_shapes}, {"decode_shapes", decode_shapes},
                   {"cases", Json::array()}};
    for (const auto& item : manifest.at("cases")) {
      const auto name = item.at("name").get<std::string>();
      require(!name.empty() && fs::path(name).filename() == name, "Invalid case name");
      auto conv = at::zeros(prefill_shapes[1], bf16);
      auto state = at::zeros(prefill_shapes[2], bf16.dtype(at::kFloat));
      size_t index = 0;
      for (const auto& step : item.at("steps")) {
        const auto kind = step.at("kind").get<std::string>();
        require(kind == "prefill" || kind == "decode", "Unknown entry point");
        const auto tokens = step.value("tokens", int64_t(kind == "prefill" ? 64 : 1));
        require(tokens >= 1 && tokens <= 512, "Expected 1 <= tokens <= 512");
        require(kind != "decode" || tokens == 1, "Decode requires one token");
        auto shape = prefill_shapes[0];
        shape[1] = tokens;
        auto x = read_hidden(fixtures / step.at("file").get<std::string>(), shape);
        const std::vector<at::Tensor> inputs{x, conv, state};
        const std::vector<at::Tensor> before{x.clone(), conv.clone(), state.clone()};
        auto result = run_sequence(prefill, decode, x, conv, state);
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
      require(line.find("libpython") == std::string::npos && line.find("libtorch_python") == std::string::npos,
              "Python library loaded");
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
