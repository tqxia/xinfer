#pragma once

#include "spec.h"

namespace xinfer {

// Storage ownership comes from state metadata; output positions come from the ABI.
class StatePolicy {
public:
  StatePolicy(const std::vector<StateSpec> &specs, c10::Device device)
      : specs_(specs), device_(device) {
    for (const auto &item : specs_) {
      require(item.inplace || item.output_index.has_value(), "Replacement state must be returned");
      if (item.output_index) {
        require(*item.output_index == output_count_, "Invalid state output order");
        ++output_count_;
      }
    }
  }

  void reset(double fill = 0) {
    if (tensors_.empty())
      tensors_.resize(specs_.size());

    for (size_t i = 0; i < specs_.size(); ++i) {
      const auto &item = specs_[i];
      auto &tensor = tensors_[i];
      const auto options = at::TensorOptions().device(device_).dtype(item.dtype);
      if (item.inplace) {
        if (!tensor.defined())
          tensor = at::empty(item.shape, options);
        tensor.fill_(fill);
      } else {
        tensor = at::zeros(item.shape, options);
      }
    }
  }

  const std::vector<at::Tensor> &tensors() const { return tensors_; }

  void accept(const std::vector<at::Tensor> &outputs, int64_t batch_size) {
    require(outputs.size() == output_count_, "Incorrect package output count");
    require(batch_size >= 1 && batch_size <= specs_[0].shape[0], "Invalid state batch size");

    // Validate the complete return before replacing any functional state.
    for (size_t i = 0; i < specs_.size(); ++i) {
      const auto &item = specs_[i];
      if (!item.output_index)
        continue;

      const auto &next = outputs[*item.output_index];
      auto shape = item.shape;
      shape[0] = batch_size;
      require(next.defined() && next.sizes() == at::IntArrayRef(shape) &&
                  next.scalar_type() == item.dtype && next.device() == device_ &&
                  next.is_contiguous(),
              "Incorrect state shape/dtype/device/layout");
      if (item.inplace)
        require(next.const_data_ptr() == tensors_[i].const_data_ptr() &&
                    next.strides() == tensors_[i].strides(),
                "KV outputs must alias caller-owned buffers");
    }

    for (size_t i = 0; i < specs_.size(); ++i) {
      if (specs_[i].inplace)
        continue;
      const auto &next = outputs[*specs_[i].output_index];
      const auto remaining = specs_[i].shape[0] - batch_size;
      // Preserve omitted rows and the original functional input. The full-batch
      // path adopts the package output directly; smaller batches retain the tail.
      tensors_[i] =
          remaining ? at::cat({next, tensors_[i].narrow(0, batch_size, remaining)}, 0) : next;
    }
  }

private:
  std::vector<StateSpec> specs_;
  c10::Device device_;
  std::vector<at::Tensor> tensors_;
  size_t output_count_ = 1;
};

} // namespace xinfer
