#pragma once
#include <ATen/core/Tensor.h>

namespace xinfer {
at::Tensor batched_flash_attention(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                                   const at::Tensor &offsets, const at::Tensor &cuda_offsets,
                                   const at::Tensor &blocks, double scale);
at::Tensor flash_attention(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                           const at::Tensor &extent, double scale);
} // namespace xinfer
