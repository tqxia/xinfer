#pragma once
#include "spec.h"
#include "generation.h"
#include "state.h"
#include "decode.h"

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <utility>
#include <vector>

namespace xinfer {

// Model specs validate the ABI; common state storage enforces its update rules.
template <class ModelSpec> class Runner {
public:
  using Spec = ModelSpec;
  using Loader = torch::inductor::AOTIModelPackageLoader;

  Runner(const fs::path &packages, const Spec &spec,
         DecodeBackendKind backend = DecodeBackendKind::Aoti)
      : Runner(spec, spec.validate_files(packages), backend) {}

  void reset(double fill = 0) {
    state_.reset(fill);
    position_ = prefill_calls_ = decode_calls_ = 0;
    megaqwen_calls_ = 0;
  }

  at::Tensor step(const std::vector<int64_t> &tokens) {
    spec_.check_step_tokens(tokens);
    const auto length = static_cast<int64_t>(tokens.size());
    require(length <= spec_.capacity - position_, "Context capacity exceeded");
    // The first call always goes through AOTI, including one-token prompts.
    // Appended multi-token chunks also use AOTI against the same state storage.
    if (native_decode_ && length == 1 && position_ > 0) {
      auto logits = native_decode_->run(tokens.front(), position_);
      check_output(logits);
      state_.accept({logits}); // Qwen3 mutates its caller-owned KV and returns logits only.
      ++position_;
      ++decode_calls_;
      ++megaqwen_calls_;
      return logits;
    }
    const auto options = at::TensorOptions().device(device_).dtype(at::kLong);
    const auto ids = length == 1 ? at::full({1, 1}, tokens.front(), options)
                                 : at::tensor(tokens, at::kLong).view({1, length}).to(options);
    std::vector<at::Tensor> inputs{ids, at::arange(position_, position_ + length, options)};
    const auto &tensors = state_.tensors();
    inputs.insert(inputs.end(), tensors.begin(), tensors.end());
    // Prefix length comes from CPU shape metadata, never CUDA tensor data.
    inputs.push_back(extent_.narrow(0, 0, position_ + length + 1));
    auto outputs = (length == 1 ? decode_ : prefill_).run(inputs);
    require(!outputs.empty(), "Unexpected compiled output");
    check_output(outputs[0]);
    state_.accept(outputs);
    position_ += length;
    ++(length == 1 ? decode_calls_ : prefill_calls_);
    return outputs[0];
  }

  at::Tensor prefill(const std::vector<int64_t> &prompt) {
    // Validate the complete request before its first chunk can change state.
    spec_.check_tokens(prompt);
    require(static_cast<int64_t>(prompt.size()) <= spec_.capacity - position_,
            "Context capacity exceeded");
    return xinfer::prefill(*this, prompt, spec_.max_prefill);
  }

  std::vector<int64_t> generate(const std::vector<int64_t> &prompt, int64_t limit,
                                std::ostream *trace = nullptr) {
    spec_.check_request(prompt, limit);
    return xinfer::generate(*this, prompt, limit, spec_.eos, trace);
  }

  const std::vector<at::Tensor> &state() const { return state_.tensors(); }
  const std::vector<at::Tensor> &cache() const { return state(); }
  const Spec &spec() const { return spec_; }
  int64_t position() const { return position_; }
  int64_t prefill_calls() const { return prefill_calls_; }
  int64_t decode_calls() const { return decode_calls_; }
  int64_t megaqwen_calls() const { return megaqwen_calls_; }
  Json decode_info() const {
    return native_decode_ ? native_decode_->info() : Json{{"backend", "aoti"}};
  }

private:
  static const Spec &checked_spec(const Spec &spec, DecodeBackendKind backend) {
    if (backend == DecodeBackendKind::MegaQwen) {
      require(spec.model_type == ModelType::Qwen3, "MegaQwen decode requires Qwen3-0.6B");
      require(megaqwen_decode_available(),
              "MegaQwen decode is not enabled; rebuild with XINFER_ENABLE_MEGAQWEN=ON");
      const auto &checkpoint = spec.metadata.at("checkpoint");
      require(checkpoint.at("model_id") == "Qwen/Qwen3-0.6B" &&
                  checkpoint.at("revision") == "c1899de289a04d12100db370d81485cdf75e47ca",
              "MegaQwen decode requires the supported pinned Qwen3-0.6B checkpoint");
    }
    return spec;
  }

  void check_output(const at::Tensor &output) const {
    require(output.defined() && output.sizes() == at::IntArrayRef({1, spec_.vocab}) &&
                output.scalar_type() == at::kBFloat16 && output.device() == device_,
            "Unexpected compiled output");
  }

  Runner(const Spec &spec, ValidatedBundle bundle, DecodeBackendKind backend)
      : spec_(checked_spec(spec, backend)), device_(configured_device(spec_)),
        weights_(std::move(bundle), spec_.shared_weights),
        prefill_(weights_.package_path("prefill").string()),
        decode_(weights_.package_path("decode").string()), state_(spec_.state, device_) {
    weights_.load(prefill_, "prefill");
    weights_.load(decode_, "decode");
    extent_ =
        at::empty({spec_.capacity + 1}, at::TensorOptions().device(at::kCPU).dtype(at::kByte));
    reset();
    if (backend == DecodeBackendKind::MegaQwen)
      native_decode_ =
          make_megaqwen_decode(weights_, state_.tensors(), spec_.capacity, spec_.vocab);
  }

  const Spec spec_;
  const c10::Device device_;
  // Borrowing loaders are destroyed before their canonical weight owner.
  SharedWeights weights_;
  Loader prefill_, decode_;
  StatePolicy state_;
  at::Tensor extent_;
  int64_t position_ = 0, prefill_calls_ = 0, decode_calls_ = 0;
  int64_t megaqwen_calls_ = 0;
  // Destroy the borrowing backend before StatePolicy and SharedWeights.
  std::unique_ptr<NativeDecode> native_decode_;
};

} // namespace xinfer
