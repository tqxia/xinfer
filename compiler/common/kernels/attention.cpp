#include "attention.h"

#include <ATen/ATen.h>
#include <ATen/ops/_scaled_dot_product_flash_attention.h>
#include <ATen/ops/_flash_attention_forward.h>
#include <torch/library.h>

#include <cmath>
#include <tuple>

namespace xinfer {

// Pack only real queries, then launch one variable-length FlashAttention for all rows.
at::Tensor batched_flash_attention(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                                   const at::Tensor &offsets, const at::Tensor &cuda_offsets,
                                   const at::Tensor &blocks, double scale) {
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.sizes() == k.sizes() && k.size(0) == q.size(0) &&
                  k.size(3) == q.size(3),
              "Invalid Q/K/V shapes");
  TORCH_CHECK(q.is_cuda() && k.device() == q.device() && v.device() == q.device() &&
                  q.scalar_type() == at::kBFloat16 && k.scalar_type() == q.scalar_type() &&
                  v.scalar_type() == q.scalar_type(),
              "Expected colocated CUDA BF16 Q/K/V");
  const auto batch = q.size(0);
  TORCH_CHECK(offsets.device().is_cpu() && offsets.scalar_type() == at::kInt &&
                  offsets.is_contiguous() && offsets.sizes() == at::IntArrayRef({2, batch + 1}),
              "Expected CPU int32 cumulative query/KV lengths [2,B+1]");
  TORCH_CHECK(cuda_offsets.device() == q.device() && cuda_offsets.scalar_type() == at::kInt &&
                  cuda_offsets.sizes() == offsets.sizes() && cuda_offsets.is_contiguous() &&
                  blocks.device() == q.device() && blocks.scalar_type() == at::kInt &&
                  blocks.sizes() == at::IntArrayRef({batch, 1}),
              "Invalid CUDA attention metadata");
  const auto *cuq = offsets.const_data_ptr<int32_t>();
  const auto *cuk = cuq + batch + 1;
  TORCH_CHECK(cuq[0] == 0 && cuk[0] == 0, "Cumulative lengths must start at zero");
  int64_t max_q = 0, max_k = 0;
  std::vector<at::Tensor> queries, keys, values;
  // Paged Flash requires a multiple-of-256 block size. Other capacities keep
  // the same contract by packing valid KV; the usual aligned path uses views.
  const bool paged = k.size(2) % 256 == 0;
  for (int64_t row = 0; row < batch; ++row) {
    const int64_t count = cuq[row + 1] - cuq[row], length = cuk[row + 1] - cuk[row];
    TORCH_CHECK(count >= 0 && count <= q.size(2) && length >= count && length >= 1 &&
                    length <= k.size(2),
                "Invalid query/KV lengths");
    max_q = std::max(max_q, count);
    max_k = std::max(max_k, length);
    queries.push_back(q[row].narrow(1, 0, count).transpose(0, 1));
    if (!paged) {
      keys.push_back(k[row].narrow(1, 0, length).transpose(0, 1));
      values.push_back(v[row].narrow(1, 0, length).transpose(0, 1));
    }
  }
  auto output = at::zeros(q.sizes(), q.options());
  if (max_q == 0)
    return output;
  auto key = paged ? k.transpose(1, 2) : at::cat(keys);
  auto value = paged ? v.transpose(1, 2) : at::cat(values);
  if (key.stride(-1) != 1)
    key = key.contiguous();
  if (value.stride(-1) != 1)
    value = value.contiguous();
  // ATen's max_q==1 GQA shortcut reshapes Q as if every row were active.
  // A bound of two selects the general varlen path for mixed empty/decode rows.
  const auto kernel_max_q = max_q == 1 && cuq[batch] != batch ? 2 : max_q;
  auto packed = std::get<0>(at::_flash_attention_forward(
      at::cat(queries), key, value, cuda_offsets[0], cuda_offsets[1], kernel_max_q, max_k, 0.0,
      true, false, scale, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      // Inductor may assign any stride to the singleton page dimension.
      paged ? std::optional<at::Tensor>(blocks.as_strided({batch, 1}, {blocks.stride(0), 1}))
            : std::nullopt,
      1));
  for (int64_t row = 0; row < batch; ++row) {
    const int64_t count = cuq[row + 1] - cuq[row];
    if (count)
      output[row].narrow(1, 0, count).copy_(packed.narrow(0, cuq[row], count).transpose(0, 1));
  }
  return output;
}

// Inference-only, aligned-prefix Qwen3/Qwen3.5 contract. No input is mutated or returned.
at::Tensor flash_attention(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                           const at::Tensor &extent, double scale) {
  TORCH_CHECK(q.is_cuda() && k.device() == q.device() && v.device() == q.device(),
              "xinfer attention requires Q/K/V on the same CUDA device");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                  v.scalar_type() == at::kBFloat16,
              "xinfer attention requires BF16");
  TORCH_CHECK(q.dim() == 4 && q.size(0) > 0 && q.size(2) >= 1 && q.size(2) <= 512,
              "Unsupported Q shape");
  TORCH_CHECK(k.dim() == 4 && v.dim() == 4 && k.size(0) == q.size(0) && k.size(2) > 0 &&
                  k.size(3) == q.size(3) && v.sizes() == k.sizes(),
              "Expected matching full-capacity K/V inputs [batch,kv_heads,capacity,head_dim]");
  TORCH_CHECK(q.size(1) > 0 && k.size(1) > 0 && q.size(1) % k.size(1) == 0,
              "Query heads must be a positive multiple of KV heads");
  TORCH_CHECK(q.size(3) >= 8 && q.size(3) <= 256 && q.size(3) % 8 == 0,
              "Flash head dimension must be a multiple of 8 in 8..256");
  TORCH_CHECK(extent.device().is_cpu() && extent.scalar_type() == at::kByte && extent.dim() == 1,
              "Expected a CPU uint8 extent vector");
  const int64_t length = extent.size(0) - 1; // Metadata only; never read tensor data.
  TORCH_CHECK(length >= q.size(2) && length <= k.size(2), "Invalid valid-prefix length");
  TORCH_CHECK(std::isfinite(scale) && scale > 0, "Attention scale must be finite and positive");
  TORCH_CHECK(!q.requires_grad() && !k.requires_grad() && !v.requires_grad(),
              "xinfer attention supports inference only");

  // Accept arbitrary input layouts as promised by the flexible_layout tag.
  // The normal Qwen path has unit feature strides, so these are views only.
  auto query = q.stride(3) == 1 ? q : q.contiguous();
  auto key = k.slice(2, 0, length);
  auto value = v.slice(2, 0, length);
  if (key.stride(3) != 1)
    key = key.contiguous();
  if (value.stride(3) != 1)
    value = value.contiguous();
  // Native ATen Flash uses lower-right causality, including appended chunks.
  auto out = std::get<0>(
      at::_scaled_dot_product_flash_attention(query, key, value, 0.0, true, false, scale));
  // Normalize singleton strides to the fake kernel's contiguous contract.
  // The pinned native kernel already returns logically contiguous output for
  // these shapes; contiguous() is a no-op in the validated decode/prefill cases.
  const auto heads = q.size(1), tokens = q.size(2), dim = q.size(3);
  return out.contiguous().as_strided({q.size(0), heads, tokens, dim},
                                     {heads * tokens * dim, tokens * dim, dim, 1});
}

} // namespace xinfer

TORCH_LIBRARY(xinfer, m) {
  m.def("batched_flash_attention(Tensor q, Tensor k, Tensor v, Tensor offsets, "
        "Tensor cuda_offsets, Tensor blocks, float scale) -> Tensor",
        {at::Tag::flexible_layout});
  // Preserve cache views instead of forcing layout copies at each call.
  m.def("flash_attention(Tensor q, Tensor k, Tensor v, Tensor kv_extent, float scale) -> Tensor",
        {at::Tag::flexible_layout});
}

TORCH_LIBRARY_IMPL(xinfer, CUDA, m) {
  m.impl("batched_flash_attention", TORCH_FN(xinfer::batched_flash_attention));
  m.impl("flash_attention", TORCH_FN(xinfer::flash_attention));
}
