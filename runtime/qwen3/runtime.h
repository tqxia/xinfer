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
    const auto &extra = metadata.at("model_specific");
    compiled_megaqwen =
        extra.contains("decode_backend") && extra.at("decode_backend") == "megaqwen";
    xinfer::metadata_check(
        extra.empty() || (compiled_megaqwen && xinfer::metadata_keys(extra, {"decode_backend"})),
        "model_specific");
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
    if (compiled_megaqwen) {
      require(state.size() == 56 && vocab == 151936,
              "MegaQwen decode requires Qwen3-0.6B geometry");
      require(capacity <= 40960, "MegaQwen context capacity exceeds 40960");
    }
  }
};

using Runtime = xinfer::Runner<Spec>;
} // namespace qwen
