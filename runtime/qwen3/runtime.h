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
  explicit Spec(const fs::path &directory) : Spec(directory, xinfer::SpecBase(directory)) {}

  Spec(const fs::path &directory, const xinfer::SpecBase &parsed) : SpecBase(parsed) {
    require(model_type == xinfer::ModelType::Qwen3, "Unsupported model type");
    const auto &specific = metadata.at("model_specific");
    for (const auto *field :
         {"layers", "kv_heads", "head_dim", "cache_shape", "cache_stride", "cache_padding"})
      xinfer::metadata_check(specific.contains(field), "model_specific");
    for (const auto *field : {"layers", "kv_heads", "head_dim", "cache_padding"})
      xinfer::metadata_check(
          xinfer::metadata_integer(specific.at(field), std::numeric_limits<int64_t>::min()),
          "model_specific");
    const auto layers = specific.at("layers").get<int64_t>();
    const auto heads = specific.at("kv_heads").get<int64_t>(),
               dim = specific.at("head_dim").get<int64_t>();
    require(layers > 0 && layers <= std::numeric_limits<int64_t>::max() / 2 && heads > 0 && dim > 0,
            "Invalid model dimensions");
    require(specific.at("cache_padding") == 0, "Unsupported cache padding");
    for (const auto *field : {"cache_shape", "cache_stride"}) {
      xinfer::metadata_check(specific.at(field).is_array(), "model_specific");
      for (const auto &value : specific.at(field))
        xinfer::metadata_check(xinfer::metadata_integer(value, std::numeric_limits<int64_t>::min()),
                               "model_specific");
    }
    const auto shape = specific.at("cache_shape").get<std::vector<int64_t>>();
    const auto stride = specific.at("cache_stride").get<std::vector<int64_t>>();
    require(shape == std::vector<int64_t>{1, 8, 1024, 128}, "Unsupported Flash cache shape");
    require(shape == std::vector<int64_t>{1, heads, capacity, dim}, "Invalid KV cache shape");
    require(stride == std::vector<int64_t>{8 * 1024 * 128, 1024 * 128, 128, 1},
            "Unexpected KV cache strides");
    xinfer::metadata_check(state.size() == size_t(2 * layers), "state");
    for (size_t i = 0; i < state.size(); ++i) {
      const auto &item = state[i];
      xinfer::metadata_check(item.layer == int64_t(i / 2) &&
                                 item.kind == (i % 2 ? "value" : "key") && item.shape == shape &&
                                 item.stride == stride && item.dtype == at::kBFloat16 &&
                                 item.inplace,
                             "state");
    }
    xinfer::metadata_check(metadata.at("output_order") == Json::array({"last_logits"}),
                           "output_order");
    validate_files(directory);
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
