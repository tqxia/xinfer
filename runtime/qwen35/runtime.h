#pragma once
#include "../common/io.h"
#include "../common/generation.h"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/InferenceMode.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/cuda.h>
#include <torch/version.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen35 {
namespace fs = std::filesystem;
using Json = nlohmann::json;
using Loader = torch::inductor::AOTIModelPackageLoader;
constexpr int64_t max_prompt = 512;

using xinfer::require;
using xinfer::integer;
inline Json read_json(const fs::path& path) {
  return xinfer::read_json(path, "Cannot open ");
}

inline void write_tensor(std::ostream& file, const at::Tensor& value) {
  const auto cpu = value.cpu().contiguous();
  file.write(static_cast<const char*>(cpu.const_data_ptr()), cpu.nbytes());
  require(bool(file), "Failed to write tensor");
}

struct TensorSpec {
  std::vector<int64_t> shape;
  at::ScalarType dtype;
};

struct Spec {
  Json metadata;
  int64_t capacity, vocab;
  std::vector<int64_t> eos;
  std::vector<TensorSpec> state;

  explicit Spec(const fs::path& packages) {
    metadata = read_json(packages / "decode.json");
    const auto prefill = read_json(packages / "prefill.json");
    for (const auto* key : {"checkpoint", "torch", "cuda", "cxx11_abi", "sm", "layer_types", "state",
                           "cache_length", "vocabulary", "hidden_size", "logits", "input_order", "output_order",
                           "input_mutation", "generation_config", "inductor_configs", "attention_backend",
                           "source_sha256", "hf_source_sha256", "tf32", "parameter_dtypes", "rope_frequency_dtype"})
      require(metadata.at(key) == prefill.at(key), std::string("Package metadata differs: ") + key);
    require(metadata.value("eager_numerics", false) == prefill.value("eager_numerics", false),
            "Package numerical settings differ");
    require(metadata.at("kind") == "decode" && metadata.at("tokens") == 1 && prefill.at("kind") == "prefill",
            "Invalid package entry points");
    require(metadata.at("delta_backend") == prefill.at("delta_backend"), "Package DeltaNet backends differ");
    require(metadata.at("delta_backend") == "triton_op", "Unsupported DeltaNet backend");
    require(prefill.at("tokens").is_null() && prefill.at("token_range") == Json::array({2, max_prompt}) &&
            metadata.at("token_range") == Json::array({1, 1}), "Expected dynamic T=2..512/T=1 packages");
    const auto version = metadata.at("torch").get<std::string>();
    require(version.substr(0, version.find('+')) == TORCH_VERSION, "Torch version mismatch");
    require(metadata.at("cxx11_abi") == bool(_GLIBCXX_USE_CXX11_ABI), "C++ ABI mismatch");
    require(metadata.at("attention_backend") == "efficient_sdpa" && metadata.at("tf32") == false &&
            metadata.at("input_mutation") == false, "Unsupported attention or mutation contract");
    capacity = metadata.at("cache_length");
    vocab = metadata.at("vocabulary");
    require(capacity == 1024 && vocab > 0, "Unsupported context capacity or vocabulary");
    require(metadata.at("logits") == Json({{"shape", {1, vocab}}, {"dtype", "bfloat16"}}),
            "Unsupported logits contract");
    const auto kinds = metadata.at("layer_types").get<std::vector<std::string>>();
    require(!kinds.empty() && metadata.at("state").size() == 2 * kinds.size(), "Invalid state count");
    Json inputs = {"input_ids", "cache_position"}, outputs = {"last_logits"};
    for (size_t i = 0; i < 2 * kinds.size(); ++i) {
      const auto& item = metadata.at("state").at(i);
      const bool linear = kinds[i / 2] == "linear_attention";
      require(linear || kinds[i / 2] == "full_attention", "Unsupported layer type");
      const std::string kind = linear ? (i % 2 ? "recurrent" : "conv") : (i % 2 ? "value" : "key");
      const std::string name = "layer_" + std::to_string(i / 2) + "." + kind;
      const bool fp32 = kind == "recurrent";
      require(item.at("layer") == i / 2 && item.at("kind") == kind && item.at("name") == name &&
              item.at("dtype") == (fp32 ? "float32" : "bfloat16"), "Invalid state order or dtype");
      auto shape = item.at("shape").get<std::vector<int64_t>>();
      require(shape.size() == (kind == "conv" ? 3 : 4) && shape[0] == 1, "Invalid state shape");
      int64_t elements = 1;
      for (auto dim : shape) {
        require(dim > 0 && dim <= std::numeric_limits<int64_t>::max() / elements / 4, "Invalid state dimension");
        elements *= dim;
      }
      require(linear || shape[2] == capacity, "KV capacity mismatch");
      if (!linear && i % 2) require(shape == state.back().shape, "K/V shapes differ");
      state.push_back({std::move(shape), fp32 ? at::kFloat : at::kBFloat16});
      inputs.push_back(name);
      outputs.push_back(name);
    }
    require(metadata.at("input_order") == inputs && metadata.at("output_order") == outputs, "Invalid tensor ABI order");
    const auto stop = metadata.at("generation_config").at("eos_token_id");
    eos = stop.is_array() ? stop.get<std::vector<int64_t>>() : std::vector<int64_t>{stop.get<int64_t>()};
    require(!eos.empty(), "Missing EOS tokens");
    for (auto id : eos) require(id >= 0 && id < vocab, "Invalid EOS token");
    for (const auto* kind : {"prefill", "decode"}) {
      const auto& meta = std::string(kind) == "prefill" ? prefill : metadata;
      require(fs::file_size(packages / (std::string(kind) + ".pt2")) == meta.at("package_bytes").get<uintmax_t>(),
              "Package file size mismatch");
    }
  }

  void check_tokens(const std::vector<int64_t>& ids) const {
    require(!ids.empty() && ids.size() <= max_prompt, "Expected 1..512 input tokens");
    for (auto id : ids) require(id >= 0 && id < vocab, "Token ID out of range");
  }

  void check_request(const std::vector<int64_t>& ids, int64_t limit) const {
    check_tokens(ids);
    require(limit > 0 && limit <= capacity - static_cast<int64_t>(ids.size()),
            "Generation budget must be positive and prompt + budget must fit the context");
  }
};

inline std::vector<int64_t> read_ids(const fs::path& path, const Spec& spec) {
  std::ifstream file(path);
  require(file.is_open(), "Cannot open token IDs");
  std::vector<int64_t> ids;
  for (std::string word; file >> word;) {
    ids.push_back(integer(word));
    require(ids.size() <= max_prompt, "Expected 1..512 input tokens");
  }
  require(file.eof(), "Failed to read token IDs");
  spec.check_tokens(ids);
  return ids;
}

inline void configure(const Spec& spec) {
  require(torch::cuda::is_available(), "CUDA device 0 is required");
  const auto* device = at::cuda::getDeviceProperties(0);
  require(spec.metadata.at("sm") == Json::array({device->major, device->minor}), "GPU architecture mismatch");
  at::set_num_threads(1);
  at::globalContext().setAllowTF32CuBLAS(false);
  at::globalContext().setAllowTF32CuDNN(false);
}

class Runtime {
 public:
  Runtime(const fs::path& packages, const Spec& spec)
      : spec_(spec),
        prefill_((packages / "prefill.pt2").string()), decode_((packages / "decode.pt2").string()) {
    reset();
  }

  void reset() {
    state_.clear();
    for (const auto& item : spec_.state)
      state_.push_back(at::zeros(item.shape, at::TensorOptions().device(at::Device(at::kCUDA, 0)).dtype(item.dtype)));
    position_ = 0;
    prefill_calls_ = decode_calls_ = 0;
  }

  at::Tensor step(const std::vector<int64_t>& tokens) {
    spec_.check_tokens(tokens);
    require(static_cast<int64_t>(tokens.size()) <= spec_.capacity - position_, "Context capacity exceeded");
    const int64_t length = tokens.size();
    const auto options = at::TensorOptions().device(at::Device(at::kCUDA, 0)).dtype(at::kLong);
    std::vector<at::Tensor> inputs{at::tensor(tokens, at::kLong).view({1, length}).to(options),
                                   at::arange(position_, position_ + length, options)};
    inputs.insert(inputs.end(), state_.begin(), state_.end());
    auto result = (length > 1 ? prefill_ : decode_).run(inputs);
    ++(length > 1 ? prefill_calls_ : decode_calls_);
    require(result.size() == 1 + state_.size(), "Incorrect package output count");
    require(result[0].sizes() == at::IntArrayRef({1, spec_.vocab}) && result[0].scalar_type() == at::kBFloat16,
            "Incorrect logits shape/dtype");
    for (size_t i = 0; i < state_.size(); ++i)
      require(result[i + 1].sizes() == state_[i].sizes() && result[i + 1].scalar_type() == state_[i].scalar_type(),
              "Incorrect state shape/dtype");
    state_.assign(result.begin() + 1, result.end());
    position_ += length;
    return result[0];
  }

  std::vector<int64_t> generate(const std::vector<int64_t>& prompt, int64_t limit, std::ostream* trace = nullptr) {
    spec_.check_request(prompt, limit);
    return xinfer::generate(*this, prompt, limit, spec_.eos, trace);
  }

  int64_t position() const { return position_; }
  int64_t prefill_calls() const { return prefill_calls_; }
  int64_t decode_calls() const { return decode_calls_; }
  const std::vector<at::Tensor>& state() const { return state_; }

 private:
  const Spec& spec_;
  Loader prefill_, decode_;
  std::vector<at::Tensor> state_;
  int64_t position_ = 0;
  int64_t prefill_calls_ = 0, decode_calls_ = 0;
};
}  // namespace qwen35
