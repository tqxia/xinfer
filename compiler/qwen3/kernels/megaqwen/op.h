#pragma once

#include <ATen/ATen.h>

namespace xinfer {

// Shared implementation used by dispatcher registration and the AOTI C shim.
// Weights are [embedding, final norm, LM head, then 28 groups in LayerPointers
// field order]. KV is [K0,V0,...]; extent.size(0) is populated KV length plus one.
at::Tensor megaqwen_decode(const at::Tensor &ids, at::TensorList weights, at::TensorList kv,
                           const at::Tensor &cos, const at::Tensor &sin, at::TensorList workspace,
                           const at::Tensor &extent);

} // namespace xinfer
