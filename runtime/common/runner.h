#pragma once
#include "spec.h"
#include "generation.h"

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <vector>

namespace xinfer {

// Policies own state storage and enforce the model's output/mutation ABI.
template <class StatePolicy> class Runner {
public:
  using Spec = typename StatePolicy::Spec;
  using Loader = torch::inductor::AOTIModelPackageLoader;

  Runner(const fs::path &packages, const Spec &spec)
      : spec_(spec), device_(checked_device(spec_)), weights_(packages, spec_.shared_weights),
        prefill_((packages / "prefill.pt2").string()), decode_((packages / "decode.pt2").string()),
        state_(spec_, device_) {
    configure(spec_);
    weights_.load(prefill_, "prefill");
    weights_.load(decode_, "decode");
    extent_ =
        at::empty({spec_.capacity + 1}, at::TensorOptions().device(at::kCPU).dtype(at::kByte));
    reset();
  }

  void reset(double fill = 0) {
    state_.reset(fill);
    position_ = prefill_calls_ = decode_calls_ = 0;
  }

  at::Tensor step(const std::vector<int64_t> &tokens) {
    spec_.check_tokens(tokens);
    const auto length = static_cast<int64_t>(tokens.size());
    require(length <= spec_.capacity - position_, "Context capacity exceeded");
    const auto options = at::TensorOptions().device(device_).dtype(at::kLong);
    const auto ids = length == 1 ? at::full({1, 1}, tokens.front(), options)
                                 : at::tensor(tokens, at::kLong).view({1, length}).to(options);
    std::vector<at::Tensor> inputs{ids, at::arange(position_, position_ + length, options)};
    const auto &tensors = state_.tensors();
    inputs.insert(inputs.end(), tensors.begin(), tensors.end());
    // Prefix length comes from CPU shape metadata, never CUDA tensor data.
    inputs.push_back(extent_.narrow(0, 0, position_ + length + 1));
    auto outputs = (length == 1 ? decode_ : prefill_).run(inputs);
    require(!outputs.empty() && outputs[0].sizes() == at::IntArrayRef({1, spec_.vocab}) &&
                outputs[0].scalar_type() == at::kBFloat16 && outputs[0].device() == device_,
            "Unexpected compiled output");
    state_.accept(outputs);
    position_ += length;
    ++(length == 1 ? decode_calls_ : prefill_calls_);
    return outputs[0];
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

private:
  const Spec spec_;
  const c10::Device device_;
  // Borrowing loaders are destroyed before their canonical weight owner.
  SharedWeights weights_;
  Loader prefill_, decode_;
  StatePolicy state_;
  at::Tensor extent_;
  int64_t position_ = 0, prefill_calls_ = 0, decode_calls_ = 0;
};

} // namespace xinfer
