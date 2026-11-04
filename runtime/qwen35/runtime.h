#pragma once
#include "../common/attention.h"
#include "../common/runner.h"

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
  explicit Spec(const fs::path &packages) : Spec(xinfer::SpecBase(packages)) {}

  explicit Spec(const xinfer::SpecBase &parsed) : SpecBase(parsed) {
    require(model_type == xinfer::ModelType::Qwen35, "Unsupported model type");
    const auto &specific = metadata.at("model_specific");
    require(specific.value("delta_backend", Json()) == "triton_op", "Unsupported DeltaNet backend");
    require(!state.empty() && state.size() % 2 == 0, "Invalid state count");
    for (size_t i = 0; i < state.size(); i += 2) {
      const auto &first = state[i];
      const auto &second = state[i + 1];
      const bool linear = first.kind == "conv";
      require((linear || first.kind == "key") && second.kind == (linear ? "recurrent" : "value") &&
                  first.layer == int64_t(i / 2) && second.layer == first.layer,
              "Invalid state order or dtype");
      if (!linear)
        require(first.shape == second.shape, "K/V shapes differ");
    }
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
