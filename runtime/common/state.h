#pragma once

#include "spec.h"

#include <cstdint>

namespace xinfer {

// Caller-owned in-place state buffers for logits-only compiled packages.
class StatePolicy {
public:
  StatePolicy(const std::vector<StateSpec> &specs, c10::Device device,
              std::optional<int64_t> allocated_batch_size = std::nullopt)
      : specs_(specs), device_(device) {
    require(!specs_.empty(), "Empty state specification");
    require(!specs_[0].shape.empty() && specs_[0].shape[0] >= 1, "Invalid state batch shape");
    const auto package_batch_size = specs_[0].shape[0];
    allocated_batch_size_ = allocated_batch_size.value_or(package_batch_size);
    require(allocated_batch_size_ >= 1 && allocated_batch_size_ <= package_batch_size,
            "Invalid allocated batch size");

    for (const auto &item : specs_) {
      require(!item.shape.empty() && item.shape[0] == package_batch_size,
              "Inconsistent state batch size");
    }
  }

  int64_t allocated_batch_size() const { return allocated_batch_size_; }

  void reset(double fill = 0) {
    if (tensors_.empty())
      tensors_.resize(specs_.size());

    for (size_t i = 0; i < specs_.size(); ++i) {
      const auto &item = specs_[i];
      auto shape = item.shape;
      shape[0] = allocated_batch_size_;
      auto &tensor = tensors_[i];
      const auto options = at::TensorOptions().device(device_).dtype(item.dtype);
      if (!tensor.defined())
        tensor = at::empty(shape, options);
      // Keep base pointers 16-byte aligned so AOTI's alignment guards never clone in-place inputs.
      require((reinterpret_cast<uintptr_t>(tensor.data_ptr()) & 15) == 0,
              "State buffer must be 16-byte aligned");
      tensor.fill_(item.is_kv() ? fill : 0);
    }
  }

  const std::vector<at::Tensor> &tensors() const { return tensors_; }

  void accept(const std::vector<at::Tensor> &outputs, int64_t batch_size) {
    require(!tensors_.empty(), "State must be reset before execution");
    require(outputs.size() == 1, "Incorrect package output count");
    require(batch_size >= 1 && batch_size <= allocated_batch_size_, "Invalid state batch size");
  }

private:
  std::vector<StateSpec> specs_;
  c10::Device device_;
  int64_t allocated_batch_size_ = 1;
  std::vector<at::Tensor> tensors_;
};

} // namespace xinfer
