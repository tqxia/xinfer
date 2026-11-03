#pragma once
#include "../common/attention.h"
#include "../common/runner.h"

namespace qwen {
namespace fs = std::filesystem;
using Json = nlohmann::json;
using xinfer::require;
using xinfer::integer;
using xinfer::read_json;
using xinfer::read_ids;
using xinfer::output_file;
using xinfer::write_logits;

struct Spec : xinfer::SpecBase {
  int64_t layers;
  std::vector<int64_t> cache_shape, cache_stride;

  explicit Spec(const fs::path& directory) : SpecBase(xinfer::ModelType::Qwen3) {
    const auto prefill = read_json(directory / "prefill.json");
    const auto decode = read_json(directory / "decode.json");
    xinfer::check_model_type(prefill, decode, model_type);
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
    xinfer::check_pair(prefill, decode, {"model_revision", "torch_version", "torch_cuda_version",
                                      "cxx11_abi", "cuda_compute_capability", "batch_size", "dtype",
                                      "layers", "kv_heads", "head_dim", "cache_length", "vocab_size",
                                      "cache_shape", "cache_stride", "flat_input_count", "eos_token_ids"});
    xinfer::check_toolchain(decode, "torch_version", "LibTorch version differs from package",
                            "LibTorch C++ ABI differs from package");
    require(decode.at("batch_size") == 1 && decode.at("dtype") == "bfloat16", "Expected batch=1 and BF16");
    require(prefill.at("entry_point") == "prefill" && decode.at("entry_point") == "decode",
            "Incorrect package entry points");
    require(decode.at("token_length") == Json::array({1, 1}) && prefill.at("token_length").at(0) == 2,
            "Unsupported token-length contract");
    layers = decode.at("layers");
    capacity = decode.at("cache_length");
    max_prompt = prefill.at("token_length").at(1);
    vocab = decode.at("vocab_size");
    cache_shape = decode.at("cache_shape").get<std::vector<int64_t>>();
    cache_stride = decode.at("cache_stride").get<std::vector<int64_t>>();
    compute_capability = decode.at("cuda_compute_capability").get<std::vector<int64_t>>();
    require(prefill.at("export_options") == decode.at("export_options"), "Package export settings differ");
    require(decode.at("export_options") == Json({{"backed_size_oblivious", false}}),
            "Unsupported Flash export settings");
    require(cache_shape == std::vector<int64_t>{1, 8, 1024, 128}, "Unsupported Flash cache shape");
    require(prefill.at("valid_cache_length") == decode.at("valid_cache_length") &&
            decode.at("valid_cache_length") == Json::array({1, capacity}),
            "Unsupported valid-prefix length contract");
    require(prefill.at("kv_extent") == decode.at("kv_extent") &&
            decode.at("kv_extent") == Json({{"device", "cpu"}, {"dtype", "uint8"},
              {"shape", "[valid_length + 1]"}, {"data_read", false}}), "Unsupported KV extent contract");
    eos = decode.at("eos_token_ids").get<std::vector<int64_t>>();
    const int64_t heads = decode.at("kv_heads"), dim = decode.at("head_dim");
    require(layers > 0 && capacity > 0 && max_prompt >= 2 && max_prompt <= capacity && vocab > 0,
            "Invalid model dimensions");
    require(cache_shape == std::vector<int64_t>{1, heads, capacity, dim} && heads > 0 && dim > 0,
            "Invalid KV cache shape");
    require(cache_stride == std::vector<int64_t>{heads * capacity * dim, capacity * dim, dim, 1},
            "Unexpected KV cache strides");
    require(decode.at("flat_input_count") == 3 + 2 * layers, "Invalid native input count");
    require(!eos.empty() && std::all_of(eos.begin(), eos.end(), [this](int64_t id) { return id >= 0 && id < vocab; }),
            "Invalid EOS token IDs");
    for (int64_t i = 0; i < 2 * layers; ++i) {
      const std::string kind = i % 2 ? "value" : "key";
      state.push_back({"layer_" + std::to_string(i / 2) + "." + kind, kind, cache_shape,
                       at::kBFloat16, true});
    }
    xinfer::check_package_files(directory, prefill, decode, false, model_type);
    shared_weights = decode.at("shared_weights");
  }
};

class StatePolicy {
 public:
  using Spec = qwen::Spec;

  StatePolicy(const Spec& spec, c10::Device device) : specs_(spec.state), device_(device) {}

  void reset(double fill = 0) {
    if (tensors_.empty())
      for (const auto& item : specs_)
        tensors_.push_back(at::empty(item.shape, at::TensorOptions().device(device_).dtype(item.dtype)));
    for (auto& tensor : tensors_) tensor.fill_(fill);
  }

  const std::vector<at::Tensor>& tensors() const { return tensors_; }

  void accept(const std::vector<at::Tensor>& outputs) {
    require(outputs.size() == 1, "Incorrect package output count");
  }

 private:
  std::vector<xinfer::StateSpec> specs_;
  c10::Device device_;
  std::vector<at::Tensor> tensors_;
};

using Runtime = xinfer::Runner<StatePolicy>;
}  // namespace qwen
