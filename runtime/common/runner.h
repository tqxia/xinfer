#pragma once

#include "spec.h"
#include "generation.h"
#include "state.h"

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
         std::optional<int64_t> allocated_batch_size = std::nullopt)
      : Runner(spec, validate_bundle(packages, spec, allocated_batch_size),
               resolve_allocated_batch_size(spec, allocated_batch_size)) {}

  void reset(double fill = 0) {
    state_.reset(fill);
    position_ = prefill_calls_ = decode_calls_ = 0;
    positions_.assign(allocated_batch_size_, 0);
  }

  at::Tensor step(const std::vector<int64_t> &tokens) { return step_batch({tokens}); }

  at::Tensor step_batch(const std::vector<std::vector<int64_t>> &rows) {
    const auto length = spec_.check_batch_tokens(rows);
    const auto batch_size = static_cast<int64_t>(rows.size());
    require(batch_size <= allocated_batch_size_, "Batch size exceeds allocated state");
    for (size_t row = 0; row < rows.size(); ++row) {
      require(static_cast<int64_t>(rows[row].size()) <= spec_.capacity - positions_[row],
              "Context capacity exceeded");
      require(positions_[row] + rows[row].size() > 0, "Each row must start with tokens");
    }

    std::vector<int64_t> tokens(rows.size() * length, 0), positions(tokens.size(), 0);
    for (size_t row = 0; row < rows.size(); ++row)
      for (size_t token = 0; token < rows[row].size(); ++token) {
        tokens[row * length + token] = rows[row][token];
        positions[row * length + token] = positions_[row] + token;
      }
    const auto options = at::TensorOptions().device(device_).dtype(at::kLong);
    auto cache_positions = at::tensor(positions, at::kLong).view({batch_size, length}).to(options);
    std::vector<at::Tensor> inputs{
        at::tensor(tokens, at::kLong).view({batch_size, length}).to(options), cache_positions};
    const auto &tensors = state_.tensors();
    // Prefix views preserve the allocation and unused rows' state.
    for (const auto &tensor : tensors)
      inputs.push_back(tensor.narrow(0, 0, batch_size));
    std::vector<int32_t> kv_lengths, query_lengths;
    for (size_t row = 0; row < rows.size(); ++row) {
      kv_lengths.push_back(positions_[row] + rows[row].size());
      query_lengths.push_back(rows[row].size());
    }
    inputs.push_back(at::tensor(kv_lengths, at::kInt));
    inputs.push_back(at::tensor(query_lengths, at::kInt));

    auto output = run_inputs(inputs, length, batch_size);
    for (size_t row = 0; row < rows.size(); ++row)
      positions_[row] += rows[row].size();
    position_ = *std::max_element(positions_.begin(), positions_.end());
    return output;
  }

private:
  at::Tensor run_inputs(const std::vector<at::Tensor> &inputs, int64_t length, int64_t batch_size) {
    auto outputs = (length == 1 ? decode_ : prefill_).run(inputs);
    require(!outputs.empty() && outputs[0].sizes() == at::IntArrayRef({batch_size, spec_.vocab}) &&
                outputs[0].scalar_type() == at::kBFloat16 && outputs[0].device() == device_,
            "Unexpected compiled output");

    state_.accept(outputs, batch_size);
    ++(length == 1 ? decode_calls_ : prefill_calls_);
    return outputs[0];
  }

public:
  at::Tensor prefill(const std::vector<int64_t> &prompt) {
    // Validate the complete request before its first chunk can change state.
    spec_.check_tokens(prompt);
    require(static_cast<int64_t>(prompt.size()) <= spec_.capacity - positions_[0],
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
  int64_t allocated_batch_size() const { return allocated_batch_size_; }
  int64_t position() const { return position_; }
  const std::vector<int64_t> &positions() const { return positions_; }
  int64_t prefill_calls() const { return prefill_calls_; }
  int64_t decode_calls() const { return decode_calls_; }

private:
  static int64_t resolve_allocated_batch_size(const Spec &spec,
                                              std::optional<int64_t> allocated_batch_size) {
    const auto batch_size = allocated_batch_size.value_or(spec.max_batch_size);
    require(batch_size >= 1 && batch_size <= spec.max_batch_size, "Invalid allocated batch size");
    return batch_size;
  }

  static ValidatedBundle validate_bundle(const fs::path &packages, const Spec &spec,
                                         std::optional<int64_t> allocated_batch_size) {
    resolve_allocated_batch_size(spec, allocated_batch_size);
    return spec.validate_files(packages);
  }

  Runner(const Spec &spec, ValidatedBundle bundle, int64_t allocated_batch_size)
      : spec_(spec), allocated_batch_size_(allocated_batch_size), device_(configured_device(spec_)),
        weights_(std::move(bundle), spec_.shared_weights),
        prefill_(weights_.package_path("prefill").string()),
        decode_(weights_.package_path("decode").string()),
        state_(spec_.state, device_, allocated_batch_size_) {
    weights_.load(prefill_, "prefill");
    weights_.load(decode_, "decode");

    reset();
  }

  const Spec spec_;
  const int64_t allocated_batch_size_;
  const c10::Device device_;

  // Borrowing loaders are destroyed before their canonical weight owner.
  SharedWeights weights_;
  Loader prefill_, decode_;
  StatePolicy state_;
  std::vector<int64_t> positions_;
  int64_t position_ = 0, prefill_calls_ = 0, decode_calls_ = 0;
};

} // namespace xinfer
