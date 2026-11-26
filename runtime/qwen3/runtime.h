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
    xinfer::metadata_check(!state.empty() && state.size() % 2 == 0, "state");

    for (size_t i = 0; i < state.size(); i += 2) {
      const auto &key = state[i];
      const auto &value = state[i + 1];
      xinfer::metadata_check(key.layer == int64_t(i / 2) && value.layer == key.layer &&
                                 key.kind == "key" && value.kind == "value" &&
                                 key.dtype == at::kBFloat16 && value.dtype == at::kBFloat16,
                             "state");
      require(key.shape == value.shape, "K/V shapes differ");
    }
  }
};

using Runtime = xinfer::Runner<Spec>;

} // namespace qwen
