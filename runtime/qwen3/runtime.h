#pragma once
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
    const std::vector<int64_t> shape{1, 8, capacity, 128};
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

using Runtime = xinfer::Runner<Spec>;
} // namespace qwen
