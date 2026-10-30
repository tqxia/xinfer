#pragma once
#include "../common/io.h"
#include "../common/generation.h"

#include "custom_ops/attention.h"

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

using xinfer::require;
using xinfer::integer;
using xinfer::read_json;
using xinfer::output_file;
using xinfer::write_logits;

struct Spec {
  int64_t layers, cache_length, max_prompt, vocab;
  std::vector<int64_t> cache_shape, cache_stride, eos;

  explicit Spec(const fs::path& directory) {
    const auto prefill = read_json(directory / "prefill.json");
    const auto decode = read_json(directory / "decode.json");
    const auto backend = decode.value("attention_backend", std::string("efficient"));
    const auto cache_input = decode.value("cache_input", std::string("full_capacity"));
    require(prefill.value("attention_backend", std::string("efficient")) == backend &&
            prefill.value("cache_input", std::string("full_capacity")) == cache_input,
            "Package attention/cache settings differ");
    require(backend == "flash",
            "Legacy efficient packages are unsupported; re-export Qwen3 packages with the current exporter");
    require(cache_input == "full_capacity_with_extent", "Unsupported attention/cache settings");
    require(prefill.contains("custom_attention") && decode.contains("custom_attention"),
            "Flash packages require the custom-op contract; re-export Qwen3 packages with the current exporter");
    const Json contract = {{"name", "xinfer::flash_attention"}, {"abi_version", 1}, {"read_only", true}};
    require(prefill.at("custom_attention") == contract && decode.at("custom_attention") == contract,
            "Unsupported custom attention contract");
    require(xinfer::attention_ops_abi_version() == 1, "Unsupported native attention ABI");
    require(prefill.value("cache_padding", 0) == decode.value("cache_padding", 0),
            "Package cache padding settings differ");
    require(decode.value("cache_padding", 0) == 0, "Unsupported cache padding");
    require(!prefill.value("eager_numerics", false) && !decode.value("eager_numerics", false),
            "Legacy eager-numerics packages are unsupported; re-export Qwen3 packages with the current exporter");
    require(prefill.at("triton_enable_fp_fusion") == decode.at("triton_enable_fp_fusion") &&
            prefill.at("inductor_configs") == decode.at("inductor_configs"),
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
    cache_stride = decode.at("cache_stride").get<std::vector<int64_t>>();
    require(prefill.at("export_options") == decode.at("export_options"), "Package export settings differ");
    require(decode.at("export_options") == Json({{"backed_size_oblivious", false}}),
            "Unsupported Flash export settings");
    require(cache_shape == std::vector<int64_t>{1, 8, 1024, 128}, "Unsupported Flash cache shape");
    require(prefill.at("valid_cache_length") == decode.at("valid_cache_length") &&
            decode.at("valid_cache_length") == Json::array({1, cache_length}),
            "Unsupported valid-prefix length contract");
    require(prefill.at("kv_extent") == decode.at("kv_extent") &&
            decode.at("kv_extent") == Json({{"device", "cpu"}, {"dtype", "uint8"},
              {"shape", "[valid_length + 1]"}, {"data_read", false}}), "Unsupported KV extent contract");
    eos = decode.at("eos_token_ids").get<std::vector<int64_t>>();
    const int64_t heads = decode.at("kv_heads"), dim = decode.at("head_dim");
    require(layers > 0 && cache_length > 0 && max_prompt >= 2 && max_prompt <= cache_length && vocab > 0,
            "Invalid model dimensions");
    require(cache_shape == std::vector<int64_t>{1, heads, cache_length, dim} && heads > 0 && dim > 0,
            "Invalid KV cache shape");
    require(cache_stride == std::vector<int64_t>{heads * cache_length * dim, cache_length * dim, dim, 1},
            "Unexpected KV cache strides");
    require(decode.at("flat_input_count") == 3 + 2 * layers, "Invalid native input count");
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

class Runtime {
 public:
  Runtime(const fs::path& directory, const Spec& spec)
      : spec_(spec), prefill_((directory / "prefill.pt2").string()),
        decode_((directory / "decode.pt2").string()) {
    // LibTorch allocations are aligned; keep these exact buffers for both loaders.
    for (int64_t i = 0; i < 2 * spec.layers; ++i)
      cache_.push_back(at::zeros(spec.cache_shape,
          at::TensorOptions().device(device_).dtype(at::kBFloat16)));
    extent_ = at::empty({spec.cache_length + 1}, at::TensorOptions().device(at::kCPU).dtype(at::kByte));
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
    // Only shape metadata is consumed: no CPU data read and no CUDA scalar sync.
    // +1 keeps the input extent >=2 even for a one-token initial prompt.
    inputs.push_back(extent_.narrow(0, 0, position_ + count + 1));
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
    return xinfer::generate(*this, prompt, limit, spec_.eos, trace);
  }

 private:
  const Spec& spec_;
  const c10::Device device_{c10::kCUDA, 0};
  Loader prefill_, decode_;
  std::vector<at::Tensor> cache_;
  at::Tensor extent_;
  int64_t position_ = 0;
};

}  // namespace qwen
