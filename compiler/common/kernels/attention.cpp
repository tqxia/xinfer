#include "attention.h"

#include <ATen/ATen.h>
#include <ATen/ops/_flash_attention_forward.h>
#include <torch/library.h>

#include <cmath>
#include <tuple>

namespace xinfer {

// Run varlen FlashAttention over padded [B,H,T,D] queries and a [B,H_kv,C,D] KV cache.
// CPU offsets[2,B+1] carry cumulative query/KV lengths to avoid device syncs while packing.
at::Tensor batched_flash_attention(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                                   const at::Tensor &offsets, const at::Tensor &cuda_offsets,
                                   const at::Tensor &blocks, double scale) {
  TORCH_CHECK(q.is_cuda() && k.device() == q.device() && v.device() == q.device(),
              "Expected colocated CUDA BF16 Q/K/V");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                  v.scalar_type() == at::kBFloat16,
              "Expected colocated CUDA BF16 Q/K/V");
  TORCH_CHECK(q.dim() == 4 && q.size(0) > 0 && q.size(2) >= 1 && q.size(2) <= 512,
              "Unsupported Q shape");
  TORCH_CHECK(k.dim() == 4 && v.dim() == 4 && v.sizes() == k.sizes() && k.size(0) == q.size(0) &&
                  k.size(2) > 0 && k.size(3) == q.size(3),
              "Invalid Q/K/V shapes");
  TORCH_CHECK(q.size(1) > 0 && k.size(1) > 0 && q.size(1) % k.size(1) == 0,
              "Query heads must be a positive multiple of KV heads");
  TORCH_CHECK(q.size(3) >= 8 && q.size(3) <= 256 && q.size(3) % 8 == 0,
              "Flash head dimension must be a multiple of 8 in 8..256");
  TORCH_CHECK(std::isfinite(scale) && scale > 0, "Attention scale must be finite and positive");
  TORCH_CHECK(!q.requires_grad() && !k.requires_grad() && !v.requires_grad(),
              "xinfer attention supports inference only");
  const auto batch = q.size(0), heads = q.size(1), width = q.size(2), dim = q.size(3);
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
  // True when every row has `width` real queries (no padded/inactive slots): allows
  // view-based pack/unpack instead of per-row slicing, at::cat, and zero-filled scatter.
  bool all_full = true;
  // Paged Flash requires a multiple-of-256 block size. Other capacities keep
  // the same contract by packing valid KV; the usual aligned path uses views.
  const bool paged = k.size(2) % 256 == 0;
  for (int64_t row = 0; row < batch; ++row) {
    const int64_t count = cuq[row + 1] - cuq[row], length = cuk[row + 1] - cuk[row];
    TORCH_CHECK(count >= 0 && count <= width && length >= count && length >= 1 &&
                    length <= k.size(2),
                "Invalid query/KV lengths");
    max_q = std::max(max_q, count);
    max_k = std::max(max_k, length);
    all_full = all_full && (count == width);
  }
  if (max_q == 0)
    return at::zeros(q.sizes(), q.options());

  // Pack [B,H,T,D] queries into FlashAttention's [total_q,H,D] varlen layout.
  // Full uniform steps (decode, B=1 prefill, uniform batches) avoid per-row slicing/cat.
  at::Tensor packed_q;
  if (all_full) {
    if (width == 1) {
      packed_q = q.squeeze(2);
    } else if (batch == 1) {
      packed_q = q.squeeze(0).transpose(0, 1);
    } else {
      packed_q = q.transpose(1, 2).reshape({batch * width, heads, dim});
    }
    if (packed_q.stride(-1) != 1)
      packed_q = packed_q.contiguous();
  } else {
    std::vector<at::Tensor> queries;
    queries.reserve(static_cast<size_t>(batch));
    for (int64_t row = 0; row < batch; ++row) {
      const int64_t count = cuq[row + 1] - cuq[row];
      if (count)
        queries.push_back(q[row].narrow(1, 0, count).transpose(0, 1));
    }
    packed_q = at::cat(queries);
  }

  // Paged mode views [B,C,H_kv,D] directly with one page per row; B=1 unpaged views
  // row 0's active prefix; multi-row unpaged packs each row's valid KV via at::cat.
  at::Tensor key, value;
  if (paged) {
    key = k.transpose(1, 2);
    value = v.transpose(1, 2);
  } else if (batch == 1) {
    key = k[0].narrow(1, 0, max_k).transpose(0, 1);
    value = v[0].narrow(1, 0, max_k).transpose(0, 1);
  } else {
    std::vector<at::Tensor> keys, values;
    keys.reserve(static_cast<size_t>(batch));
    values.reserve(static_cast<size_t>(batch));
    for (int64_t row = 0; row < batch; ++row) {
      const int64_t length = cuk[row + 1] - cuk[row];
      keys.push_back(k[row].narrow(1, 0, length).transpose(0, 1));
      values.push_back(v[row].narrow(1, 0, length).transpose(0, 1));
    }
    key = at::cat(keys);
    value = at::cat(values);
  }
  if (key.stride(-1) != 1)
    key = key.contiguous();
  if (value.stride(-1) != 1)
    value = value.contiguous();
  // ATen's max_q==1 GQA shortcut reshapes Q as if every row were active.
  // A bound of two selects the general varlen path for mixed empty/decode rows.
  const auto kernel_max_q = max_q == 1 && cuq[batch] != batch ? 2 : max_q;
  auto packed = std::get<0>(at::_flash_attention_forward(
      packed_q, key, value, cuda_offsets[0], cuda_offsets[1], kernel_max_q, max_k, 0.0, true, false,
      scale, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      // Inductor may assign any stride to the singleton page dimension.
      paged ? std::optional<at::Tensor>(blocks.as_strided({batch, 1}, {blocks.stride(0), 1}))
            : std::nullopt,
      1));
  // Unpack [total_q,H,D] back to [B,H,T,D], normalizing singleton strides for AOTI.
  // Ragged or partially active batches zero-fill padded slots before scattering active rows.
  if (all_full) {
    if (width == 1) {
      if (!packed.is_contiguous())
        packed = packed.contiguous();
      return packed.as_strided({batch, heads, 1, dim}, {heads * dim, dim, dim, 1});
    }
    auto out = packed.reshape({batch, width, heads, dim}).transpose(1, 2).contiguous();
    return out.as_strided({batch, heads, width, dim}, {heads * width * dim, width * dim, dim, 1});
  }
  auto output = at::zeros(q.sizes(), q.options());
  for (int64_t row = 0; row < batch; ++row) {
    const int64_t count = cuq[row + 1] - cuq[row];
    if (count)
      output[row].narrow(1, 0, count).copy_(packed.narrow(0, cuq[row], count).transpose(0, 1));
  }
  return output;
}

} // namespace xinfer

TORCH_LIBRARY(xinfer, m) {
  // Preserve cache views instead of forcing layout copies at each call.
  m.def("batched_flash_attention(Tensor q, Tensor k, Tensor v, Tensor offsets, "
        "Tensor cuda_offsets, Tensor blocks, float scale) -> Tensor",
        {at::Tag::flexible_layout});
}

TORCH_LIBRARY_IMPL(xinfer, CUDA, m) {
  m.impl("batched_flash_attention", TORCH_FN(xinfer::batched_flash_attention));
}
