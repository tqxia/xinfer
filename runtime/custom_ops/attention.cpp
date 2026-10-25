#include "attention.h"

#include <ATen/ATen.h>
#include <ATen/ops/_scaled_dot_product_flash_attention.h>
#include <torch/library.h>

#include <cmath>
#include <tuple>

namespace xinfer {

int attention_ops_abi_version() { return 1; }
const char* flashinfer_revision() {
#ifdef XINFER_FLASHINFER_REVISION
  return XINFER_FLASHINFER_REVISION;
#else
  return "";
#endif
}

// Inference-only, batch-one Qwen3-0.6B contract. No input is mutated or returned.
at::Tensor flash_attention(const at::Tensor& q, const at::Tensor& k,
                          const at::Tensor& v, const at::Tensor& extent,
                          double scale) {
  TORCH_CHECK(q.is_cuda() && k.device() == q.device() && v.device() == q.device(),
              "xinfer attention requires Q/K/V on the same CUDA device");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
              v.scalar_type() == at::kBFloat16, "xinfer attention requires BF16");
  TORCH_CHECK(q.dim() == 4 && q.size(0) == 1 && q.size(1) == 16 && q.size(3) == 128 &&
              q.size(2) >= 1 && q.size(2) <= 512, "Unsupported Q shape");
  TORCH_CHECK(k.sizes() == at::IntArrayRef({1, 8, 1024, 128}) && v.sizes() == k.sizes(),
              "Expected full-capacity K/V inputs [1,8,1024,128]");
  TORCH_CHECK(extent.device().is_cpu() && extent.scalar_type() == at::kByte &&
              extent.dim() == 1, "Expected a CPU uint8 extent vector");
  const int64_t length = extent.size(0) - 1;  // Metadata only; never read tensor data.
  TORCH_CHECK(length >= q.size(2) && length <= k.size(2), "Invalid valid-prefix length");
  TORCH_CHECK(std::isfinite(scale) && scale > 0, "Attention scale must be finite and positive");
  TORCH_CHECK(!q.requires_grad() && !k.requires_grad() && !v.requires_grad(),
              "xinfer attention supports inference only");

  // Accept arbitrary input layouts as promised by the flexible_layout tag.
  // The normal Qwen path has unit feature strides, so these are views only.
  auto query = q.stride(3) == 1 ? q : q.contiguous();
  auto key = k.slice(2, 0, length);
  auto value = v.slice(2, 0, length);
  if (key.stride(3) != 1) key = key.contiguous();
  if (value.stride(3) != 1) value = value.contiguous();
  // Native ATen Flash uses lower-right causality, including appended chunks.
  auto out = std::get<0>(at::_scaled_dot_product_flash_attention(
      query, key, value, 0.0, true, false, scale));
  // Normalize singleton strides to the fake kernel's contiguous contract.
  // The pinned native kernel already returns logically contiguous output for
  // these shapes; contiguous() is a no-op in the validated decode/prefill cases.
  const auto tokens = q.size(2);
  return out.contiguous().as_strided({1, 16, tokens, 128},
                                    {16 * tokens * 128, tokens * 128, 128, 1});
}

}  // namespace xinfer

TORCH_LIBRARY(xinfer, m) {
  m.def("flashinfer_revision() -> str", []() { return std::string(xinfer::flashinfer_revision()); });
  // Without this tag Inductor repacks the padded full caches at every call.
  m.def("flash_attention(Tensor q, Tensor k, Tensor v, Tensor kv_extent, float scale) -> Tensor",
        {at::Tag::flexible_layout});
}

TORCH_LIBRARY_IMPL(xinfer, CUDA, m) {
  m.impl("flash_attention", TORCH_FN(xinfer::flash_attention));
}
