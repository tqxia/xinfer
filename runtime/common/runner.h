#pragma once

#include "spec.h"
#include "generation.h"
#include "state.h"

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <algorithm>
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

    const int64_t count = batch_size * length;
    // Keep cache_positions 16-byte aligned for AOTI's compiled input alignment.
    const int64_t stride = (count + 1) & ~int64_t{1};
    auto *token_data = host_token_pos_.data_ptr<int64_t>();
    auto *position_data = token_data + stride;
    std::fill_n(token_data, stride + count, int64_t{0});
    auto *kv_data = host_lengths_.data_ptr<int32_t>();
    auto *query_data = kv_data + allocated_batch_size_;
    for (size_t row = 0; row < rows.size(); ++row) {
      const auto row_tokens = static_cast<int64_t>(rows[row].size());
      for (int64_t token = 0; token < row_tokens; ++token) {
        token_data[row * length + token] = rows[row][token];
        position_data[row * length + token] = positions_[row] + token;
      }
      kv_data[row] = static_cast<int32_t>(positions_[row] + row_tokens);
      query_data[row] = static_cast<int32_t>(row_tokens);
    }

    // One coalesced pageable H2D copy avoids stalling the GPU stream while
    // keeping staging buffers safe to reuse on the next step_batch call.
    auto staged = device_token_pos_.narrow(0, 0, stride + count);
    staged.copy_(host_token_pos_.narrow(0, 0, stride + count));
    inputs_.clear();
    inputs_.push_back(staged.narrow(0, 0, count).view({batch_size, length}));
    inputs_.push_back(staged.narrow(0, stride, count).view({batch_size, length}));
    // Prefix views preserve the allocation and unused rows' state.
    for (const auto &tensor : state_.tensors())
      inputs_.push_back(tensor.narrow(0, 0, batch_size));
    inputs_.push_back(host_lengths_.select(0, 0).narrow(0, 0, batch_size));
    inputs_.push_back(host_lengths_.select(0, 1).narrow(0, 0, batch_size));

    auto output = run_inputs(inputs_, length, batch_size);
    inputs_.clear();
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
        state_(spec_.state, device_, allocated_batch_size_),
        host_token_pos_(
            at::empty({2 * ((allocated_batch_size_ * spec_.max_prefill + 1) & ~int64_t{1})},
                      at::TensorOptions().device(at::kCPU).dtype(at::kLong))),
        device_token_pos_(
            at::empty({2 * ((allocated_batch_size_ * spec_.max_prefill + 1) & ~int64_t{1})},
                      at::TensorOptions().device(device_).dtype(at::kLong))),
        host_lengths_(at::empty({2, allocated_batch_size_},
                                at::TensorOptions().device(at::kCPU).dtype(at::kInt))) {
    weights_.load(prefill_, "prefill");
    weights_.load(decode_, "decode");
    inputs_.reserve(state_.tensors().size() + 4);

    reset();
  }

  const Spec spec_;
  const int64_t allocated_batch_size_;
  const c10::Device device_;

  // Borrowing loaders are destroyed before their canonical weight owner.
  SharedWeights weights_;
  Loader prefill_, decode_;
  StatePolicy state_;
  at::Tensor host_token_pos_, device_token_pos_, host_lengths_;
  std::vector<at::Tensor> inputs_;
  std::vector<int64_t> positions_;
  int64_t position_ = 0, prefill_calls_ = 0, decode_calls_ = 0;
};

} // namespace xinfer
