#pragma once
#include <ATen/core/Tensor.h>

namespace xinfer {
at::Tensor flash_attention(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                           const at::Tensor &extent, double scale);
} // namespace xinfer
