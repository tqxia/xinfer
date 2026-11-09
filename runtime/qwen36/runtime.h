#pragma once
#include "../common/attention.h"
#include "../common/runner.h"

namespace qwen36 {
namespace fs = std::filesystem;
using Json = nlohmann::json;
using xinfer::require;

struct Spec : xinfer::SpecBase {
  explicit Spec(const fs::path &packages) : Spec(xinfer::SpecBase(packages)) {}

  explicit Spec(const xinfer::SpecBase &parsed) : SpecBase(parsed) {
    require(model_type == xinfer::ModelType::Qwen36, "Unsupported model type");
    const auto &checkpoint = metadata.at("checkpoint");
    require(checkpoint.at("model_id") == "Qwen/Qwen3.6-35B-A3B-FP8" &&
                checkpoint.at("revision") == "c653bba46dfa06e9658dcc0b0dc87fe4685f14a2" &&
                vocab == 248320 && eos == std::vector<int64_t>{248046, 248044} &&
                capacity <= 262144,
            "Unsupported Qwen3.6 checkpoint or execution bounds");
    const auto &specific = metadata.at("model_specific");
    const Json kernel = {{"repo", "kernels-community/finegrained-fp8"},
                         {"revision", "03ae31933313f38fae2b050dbe2731024386c369"}};
    require(specific.value("delta_backend", Json()) == "triton_op" &&
                specific.value("attention_backend", Json()) == "flash_attention" &&
                specific.value("fp8_backend", Json()) == "triton_op" &&
                specific.value("fp8_format", Json()) == "e4m3fn" &&
                specific.value("fp8_block_size", Json()) == Json::array({128, 128}) &&
                specific.value("fp8_scale_dtype", Json()) == "bfloat16" &&
                xinfer::metadata_equal(specific.value("fp8_kernel", Json()), kernel) &&
                xinfer::metadata_equal(specific.value("linear_packing", Json()),
                                       Json{{"enabled", false}}),
            "Unsupported Qwen3.6 FP8 backend");
    require(state.size() == 80, "Qwen3.6 packages require all 40 decoder layers");
    for (size_t layer = 0; layer < 40; ++layer) {
      const bool attention = layer % 4 == 3;
      const auto &first = state[2 * layer];
      const auto &second = state[2 * layer + 1];
      const std::vector<int64_t> first_shape =
          attention ? std::vector<int64_t>{1, 2, capacity, 256} : std::vector<int64_t>{1, 8192, 4};
      const std::vector<int64_t> second_shape =
          attention ? first_shape : std::vector<int64_t>{1, 32, 128, 128};
      require(first.layer == int64_t(layer) && second.layer == int64_t(layer) &&
                  first.kind == (attention ? "key" : "conv") &&
                  second.kind == (attention ? "value" : "recurrent") &&
                  first.shape == first_shape && second.shape == second_shape,
              "Unsupported Qwen3.6 state geometry");
    }
  }
};

using Runtime = xinfer::Runner<Spec>;
} // namespace qwen36
