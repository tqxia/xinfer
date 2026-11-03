#pragma once
#include "../common/attention.h"
#include "../common/runner.h"

#include <limits>

namespace qwen35 {
namespace fs = std::filesystem;
using Json = nlohmann::json;
using xinfer::require;
using xinfer::integer;
using xinfer::read_json;
using xinfer::read_ids;
using xinfer::configure;
constexpr int64_t max_prompt = 512;

struct Spec : xinfer::SpecBase {
  Json metadata;

  explicit Spec(const fs::path& packages) : SpecBase(xinfer::ModelType::Qwen35) {
    metadata = read_json(packages / "decode.json");
    const auto prefill = read_json(packages / "prefill.json");
    xinfer::check_model_type(prefill, metadata, model_type);
    require(!metadata.value("eager_numerics", false) && !prefill.value("eager_numerics", false),
            "Legacy eager-numerics packages are unsupported; re-export Qwen3.5 packages with the current exporter");
    xinfer::check_optional_pair(prefill, metadata,
                                {"attention_backend", "input_mutation", "cache_input", "kv_extent",
                                 "custom_attention", "export_options"});
    const Json attention = {{"name", "xinfer::flash_attention"}, {"abi_version", 1}, {"read_only", true}};
    const Json extent = {{"device", "cpu"}, {"dtype", "uint8"}, {"shape", "[valid_length + 1]"}, {"data_read", false}};
    require(metadata.value("attention_backend", Json()) == "flash" && metadata.value("input_mutation", Json()) == "kv_only" &&
            metadata.value("cache_input", Json()) == "full_capacity_with_extent" &&
            metadata.value("kv_extent", Json()) == extent && metadata.value("custom_attention", Json()) == attention &&
            metadata.value("export_options", Json()) == Json({{"backed_size_oblivious", false}}),
            "Legacy attention or state contract is unsupported; re-export Qwen3.5 packages with the current exporter");
    require(xinfer::attention_ops_abi_version() == 1, "Unsupported native attention ABI");
    xinfer::check_pair(prefill, metadata,
                       {"checkpoint", "torch", "cuda", "cxx11_abi", "sm", "layer_types", "state",
                        "cache_length", "vocabulary", "hidden_size", "logits", "input_order", "output_order",
                        "generation_config", "inductor_configs", "source_sha256", "hf_source_sha256",
                        "tf32", "parameter_dtypes", "rope_frequency_dtype"});
    require(metadata.value("linear_packing", Json()) == prefill.value("linear_packing", Json()),
            "Package linear-packing settings differ");
    require(metadata.at("kind") == "decode" && metadata.at("tokens") == 1 && prefill.at("kind") == "prefill",
            "Invalid package entry points");
    require(metadata.at("delta_backend") == prefill.at("delta_backend"), "Package DeltaNet backends differ");
    require(metadata.at("delta_backend") == "triton_op", "Unsupported DeltaNet backend");
    // Graph counts differ by entry; the dispatch policy must match across a pair.
    // Missing fields on both sides retain support for earlier package pairs.
    xinfer::check_optional_pair(prefill, metadata,
                                {"delta_preparation", "delta_operator", "delta_algorithm", "delta_short_threshold",
                                 "delta_chunk_size", "delta_transform_tile", "delta_dot_precision"},
                                "Package DeltaNet policy differs: ");
    require(prefill.at("tokens").is_null() && prefill.at("token_range") == Json::array({2, max_prompt}) &&
            metadata.at("token_range") == Json::array({1, 1}), "Expected dynamic T=2..512/T=1 packages");
    xinfer::check_toolchain(metadata, "torch");
    require(metadata.at("tf32") == false, "Unsupported numerical contract");
    capacity = metadata.at("cache_length");
    vocab = metadata.at("vocabulary");
    compute_capability = metadata.at("sm").get<std::vector<int64_t>>();
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
      state.push_back({name, kind, std::move(shape), fp32 ? at::kFloat : at::kBFloat16, !linear});
      inputs.push_back(name);
      outputs.push_back(name);
    }
    inputs.push_back("kv_extent");
    require(metadata.at("input_order") == inputs && metadata.at("output_order") == outputs, "Invalid tensor ABI order");
    const auto stop = metadata.at("generation_config").at("eos_token_id");
    eos = stop.is_array() ? stop.get<std::vector<int64_t>>() : std::vector<int64_t>{stop.get<int64_t>()};
    check_eos();
    xinfer::check_package_files(packages, prefill, metadata, true, model_type);
    shared_weights = metadata.at("shared_weights");
  }
};

class StatePolicy {
 public:
  using Spec = qwen35::Spec;

  StatePolicy(const Spec& spec, c10::Device device) : specs_(spec.state), device_(device) {}

  void reset(double fill = 0) {
    tensors_.clear();
    for (const auto& item : specs_) {
      auto tensor = at::zeros(item.shape, at::TensorOptions().device(device_).dtype(item.dtype));
      if (item.inplace && fill != 0) tensor.fill_(fill);
      tensors_.push_back(std::move(tensor));
    }
  }

  const std::vector<at::Tensor>& tensors() const { return tensors_; }

  void accept(const std::vector<at::Tensor>& outputs) {
    require(outputs.size() == 1 + tensors_.size(), "Incorrect package output count");
    for (size_t i = 0; i < tensors_.size(); ++i) {
      const auto& next = outputs[i + 1];
      require(next.sizes() == tensors_[i].sizes() && next.scalar_type() == tensors_[i].scalar_type() &&
                  next.device() == device_, "Incorrect state shape/dtype/device");
      if (specs_[i].inplace)
        require(next.data_ptr() == tensors_[i].data_ptr() && next.strides() == tensors_[i].strides(),
                "KV outputs must alias caller-owned buffers");
    }
    tensors_.assign(outputs.begin() + 1, outputs.end());
  }

 private:
  std::vector<xinfer::StateSpec> specs_;
  c10::Device device_;
  std::vector<at::Tensor> tensors_;
};

using Runtime = xinfer::Runner<StatePolicy>;
}  // namespace qwen35
