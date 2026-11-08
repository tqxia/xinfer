#pragma once
#include "../common/runner.h"

namespace qwen35 {
namespace fs = std::filesystem;
using Json = nlohmann::json;
using xinfer::integer;
using xinfer::read_ids;
using xinfer::read_json;
using xinfer::require;

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

using Runtime = xinfer::Runner<Spec>;
} // namespace qwen35
