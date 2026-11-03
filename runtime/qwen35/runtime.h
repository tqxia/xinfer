#pragma once
#include "../common/attention.h"
#include "../common/runner.h"

#include <limits>

namespace qwen35 {
namespace fs = std::filesystem;
using Json = nlohmann::json;
using xinfer::configure;
using xinfer::integer;
using xinfer::read_ids;
using xinfer::read_json;
using xinfer::require;
constexpr int64_t max_prompt = 512;

struct Spec : xinfer::SpecBase {
  explicit Spec(const fs::path &packages) : Spec(packages, xinfer::SpecBase(packages)) {}

  Spec(const fs::path &packages, const xinfer::SpecBase &parsed) : SpecBase(parsed) {
    require(model_type == xinfer::ModelType::Qwen35, "Unsupported model type");
    const auto &specific = metadata.at("model_specific");
    require(specific.value("delta_backend", Json()) == "triton_op", "Unsupported DeltaNet backend");
    require(specific.contains("layer_types") && specific.at("layer_types").is_array() &&
                !specific.at("layer_types").empty() &&
                state.size() == 2 * specific.at("layer_types").size(),
            "Invalid state count");
    std::vector<std::string> kinds;
    for (const auto &kind : specific.at("layer_types")) {
      require(kind.is_string() && (kind == "linear_attention" || kind == "full_attention"),
              "Unsupported layer type");
      kinds.push_back(kind.get<std::string>());
    }
    Json outputs = {"last_logits"};
    for (size_t i = 0; i < state.size(); ++i) {
      const bool linear = kinds[i / 2] == "linear_attention";
      const std::string kind = linear ? (i % 2 ? "recurrent" : "conv") : (i % 2 ? "value" : "key");
      const auto &item = state[i];
      require(item.layer == int64_t(i / 2) && item.kind == kind, "Invalid state order or dtype");
      if (!linear && i % 2)
        require(item.shape == state[i - 1].shape, "K/V shapes differ");
      outputs.push_back(item.name);
    }
    require(metadata.at("output_order") == outputs, "Invalid tensor ABI order");
    validate_files(packages);
  }
};

class StatePolicy {
public:
  using Spec = qwen35::Spec;

  StatePolicy(const Spec &spec, c10::Device device) : specs_(spec.state), device_(device) {}

  void reset(double fill = 0) {
    tensors_.clear();
    for (const auto &item : specs_) {
      auto tensor = at::zeros(item.shape, at::TensorOptions().device(device_).dtype(item.dtype));
      if (item.inplace && fill != 0)
        tensor.fill_(fill);
      tensors_.push_back(std::move(tensor));
    }
  }

  const std::vector<at::Tensor> &tensors() const { return tensors_; }

  void accept(const std::vector<at::Tensor> &outputs) {
    require(outputs.size() == 1 + tensors_.size(), "Incorrect package output count");
    for (size_t i = 0; i < tensors_.size(); ++i) {
      const auto &next = outputs[i + 1];
      require(next.sizes() == tensors_[i].sizes() &&
                  next.scalar_type() == tensors_[i].scalar_type() && next.device() == device_,
              "Incorrect state shape/dtype/device");
      if (specs_[i].inplace)
        require(next.data_ptr() == tensors_[i].data_ptr() &&
                    next.strides() == tensors_[i].strides(),
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
} // namespace qwen35
