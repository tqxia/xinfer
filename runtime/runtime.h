#pragma once

#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/cuda.h>
#include <torch/version.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen {

namespace fs = std::filesystem;
using Json = nlohmann::json;
using Loader = torch::inductor::AOTIModelPackageLoader;

inline void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

inline int64_t integer(const std::string& text) {
  int64_t value = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  require(result.ec == std::errc{} && result.ptr == text.data() + text.size(),
          "Invalid integer: " + text);
  return value;
}

inline Json read_json(const fs::path& path) {
  std::ifstream file(path);
  require(file.is_open(), "Cannot open metadata: " + path.string());
  return Json::parse(file);
}

struct Spec {
  int64_t layers, cache_length, max_prompt, vocab;
  std::vector<int64_t> cache_shape, eos;

  explicit Spec(const fs::path& directory) {
    const auto prefill = read_json(directory / "prefill.json");
    const auto decode = read_json(directory / "decode.json");
    require(prefill.value("eager_numerics", false) == decode.value("eager_numerics", false) &&
            prefill.value("triton_enable_fp_fusion", true) == decode.value("triton_enable_fp_fusion", true),
            "Package numerical settings differ");
    for (const auto* key : {"model_revision", "torch_version", "torch_cuda_version",
                           "cxx11_abi", "cuda_compute_capability", "batch_size", "dtype",
                           "layers", "kv_heads", "head_dim", "cache_length", "vocab_size",
                           "cache_shape", "cache_stride", "flat_input_count", "eos_token_ids"}) {
      require(prefill.at(key) == decode.at(key), std::string("Package metadata differs: ") + key);
    }
    const auto version = decode.at("torch_version").get<std::string>();
    require(version.substr(0, version.find('+')) == TORCH_VERSION, "LibTorch version differs from package");
    require(decode.at("cxx11_abi") == bool(_GLIBCXX_USE_CXX11_ABI), "LibTorch C++ ABI differs from package");
    require(decode.at("batch_size") == 1 && decode.at("dtype") == "bfloat16", "Expected batch=1 and BF16");
    require(prefill.at("entry_point") == "prefill" && decode.at("entry_point") == "decode",
            "Incorrect package entry points");
    require(decode.at("token_length") == Json::array({1, 1}) && prefill.at("token_length").at(0) == 2,
            "Unsupported token-length contract");
    layers = decode.at("layers");
    cache_length = decode.at("cache_length");
    max_prompt = prefill.at("token_length").at(1);
    vocab = decode.at("vocab_size");
    cache_shape = decode.at("cache_shape").get<std::vector<int64_t>>();
    eos = decode.at("eos_token_ids").get<std::vector<int64_t>>();
    const int64_t heads = decode.at("kv_heads"), dim = decode.at("head_dim");
    require(layers > 0 && cache_length > 0 && max_prompt >= 2 && max_prompt <= cache_length && vocab > 0,
            "Invalid model dimensions");
    require(cache_shape == std::vector<int64_t>{1, heads, cache_length, dim} && heads > 0 && dim > 0,
            "Invalid KV cache shape");
    require(decode.at("cache_stride") == Json::array({heads * cache_length * dim, cache_length * dim, dim, 1}),
            "Expected contiguous KV cache");
    require(decode.at("flat_input_count") == 2 + 2 * layers, "Invalid native input count");
    require(!eos.empty() && std::all_of(eos.begin(), eos.end(), [this](int64_t id) { return id >= 0 && id < vocab; }),
            "Invalid EOS token IDs");
  }
};

inline std::vector<int64_t> read_ids(const fs::path& path, const Spec& spec) {
  std::ifstream file(path);
  require(file.is_open(), "Cannot open token IDs: " + path.string());
  std::vector<int64_t> ids;
  for (std::string word; file >> word;) {
    const int64_t id = integer(word);
    require(id >= 0 && id < spec.vocab, "Token ID out of range: " + word);
    ids.push_back(id);
    require(static_cast<int64_t>(ids.size()) <= spec.max_prompt, "Prompt exceeds maximum token length");
  }
  require(file.eof(), "Failed to read token IDs: " + path.string());
  require(!ids.empty(), "Prompt must not be empty");
  return ids;
}

inline std::ofstream output_file(const fs::path& path, bool binary = false) {
  if (path.has_parent_path()) fs::create_directories(path.parent_path());
  std::ofstream file(path, std::ios::out | (binary ? std::ios::binary : std::ios::openmode{}));
  require(file.is_open(), "Cannot open output: " + path.string());
  return file;
}

inline void write_logits(std::ostream& stream, const at::Tensor& logits) {
  const auto row = logits.to(at::kCPU, at::kFloat).contiguous();
  stream.write(reinterpret_cast<const char*>(row.const_data_ptr<float>()), row.numel() * sizeof(float));
  require(bool(stream), "Failed to write logits trace");
}

class Runtime {
 public:
  Runtime(const fs::path& directory, const Spec& spec)
      : spec_(spec), prefill_((directory / "prefill.pt2").string()),
        decode_((directory / "decode.pt2").string()) {
    // LibTorch allocations are aligned; keep these exact buffers for both loaders.
    for (int64_t i = 0; i < 2 * spec.layers; ++i)
      cache_.push_back(at::zeros(spec.cache_shape, at::TensorOptions().device(device_).dtype(at::kBFloat16)));
  }

  void reset(double fill = 0) {
    for (auto& tensor : cache_) tensor.fill_(fill);
    position_ = 0;
  }

  const std::vector<at::Tensor>& cache() const { return cache_; }
  int64_t position() const { return position_; }

  at::Tensor step(const std::vector<int64_t>& tokens) {
    const auto count = static_cast<int64_t>(tokens.size());
    require(count > 0 && count <= spec_.max_prompt, "Step token count is outside the package range");
    require(count <= spec_.cache_length - position_, "Step exceeds KV cache capacity");
    require(std::all_of(tokens.begin(), tokens.end(), [this](int64_t id) { return id >= 0 && id < spec_.vocab; }),
            "Step token ID out of range");
    const auto options = at::TensorOptions().device(device_).dtype(at::kLong);
    std::vector<at::Tensor> inputs{
        count == 1 ? at::full({1, 1}, tokens[0], options) : at::tensor(tokens, at::kLong).unsqueeze(0).to(device_),
        count == 1 ? at::full({1}, position_, options) : at::arange(position_, position_ + count, options)};
    inputs.insert(inputs.end(), cache_.begin(), cache_.end());
    // T=1 also handles a one-token prompt at position zero.
    auto outputs = (count == 1 ? decode_ : prefill_).run(inputs);
    require(outputs.size() == 1 && outputs[0].sizes() == at::IntArrayRef({1, spec_.vocab}) &&
            outputs[0].scalar_type() == at::kBFloat16 && outputs[0].device() == device_,
            "Unexpected compiled output");
    position_ += count;
    return outputs[0];
  }

  std::vector<int64_t> generate(const std::vector<int64_t>& prompt, int64_t limit, std::ostream* trace) {
    require(!prompt.empty() && static_cast<int64_t>(prompt.size()) <= spec_.max_prompt,
            "Prompt token count is outside the package range");
    require(limit > 0 && limit <= spec_.cache_length - static_cast<int64_t>(prompt.size()),
            "Generation budget exceeds KV cache capacity");
    reset();
    auto logits = step(prompt);
    std::vector<int64_t> generated;
    for (int64_t index = 0; index < limit; ++index) {
      // Row i predicts generated[i]; same format in the CLI and validation driver.
      if (trace) write_logits(*trace, logits);
      const int64_t token = logits.argmax(-1).item<int64_t>();
      generated.push_back(token);
      if (std::find(spec_.eos.begin(), spec_.eos.end(), token) != spec_.eos.end() || index + 1 == limit) break;
      logits = step({token});
    }
    return generated;
  }

 private:
  const Spec& spec_;
  const c10::Device device_{c10::kCUDA, 0};
  Loader prefill_, decode_;
  std::vector<at::Tensor> cache_;
  int64_t position_ = 0;
};

}  // namespace qwen
