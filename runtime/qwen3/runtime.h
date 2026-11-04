#pragma once
#include "../common/attention.h"
#include "../common/runner.h"

namespace qwen {
namespace fs = std::filesystem;
using Json = nlohmann::json;
using xinfer::integer;
using xinfer::output_file;
using xinfer::read_ids;
using xinfer::read_json;
using xinfer::require;
using xinfer::write_logits;

struct Spec : xinfer::SpecBase {
  explicit Spec(const fs::path &directory) : Spec(xinfer::SpecBase(directory)) {}

  explicit Spec(const xinfer::SpecBase &parsed) : SpecBase(parsed) {
    require(model_type == xinfer::ModelType::Qwen3, "Unsupported model type");
    xinfer::metadata_check(metadata.at("model_specific").empty(), "model_specific");
    xinfer::metadata_check(!state.empty() && state.size() % 2 == 0, "state");
    const std::vector<int64_t> shape{1, 8, 1024, 128};
    for (size_t i = 0; i < state.size(); ++i) {
      const auto &item = state[i];
      xinfer::metadata_check(item.layer == int64_t(i / 2) &&
                                 item.kind == (i % 2 ? "value" : "key") &&
                                 item.dtype == at::kBFloat16 && item.inplace,
                             "state");
      require(item.shape == shape, "Unsupported Flash cache shape");
    }
  }
};

class StatePolicy {
public:
  using Spec = qwen::Spec;

  StatePolicy(const Spec &spec, c10::Device device) : specs_(spec.state), device_(device) {}

  void reset(double fill = 0) {
    if (tensors_.empty())
      for (const auto &item : specs_)
        tensors_.push_back(
            at::empty(item.shape, at::TensorOptions().device(device_).dtype(item.dtype)));
    for (auto &tensor : tensors_)
      tensor.fill_(fill);
  }

  const std::vector<at::Tensor> &tensors() const { return tensors_; }

  void accept(const std::vector<at::Tensor> &outputs) {
    require(outputs.size() == 1, "Incorrect package output count");
  }

private:
  std::vector<xinfer::StateSpec> specs_;
  c10::Device device_;
  std::vector<at::Tensor> tensors_;
};

using Runtime = xinfer::Runner<StatePolicy>;
} // namespace qwen
