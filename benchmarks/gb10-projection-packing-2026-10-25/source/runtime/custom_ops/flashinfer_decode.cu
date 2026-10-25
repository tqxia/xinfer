// FlashInfer v0.7.2.post1, single-request CUDA-core decode (not XQA).
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/decode.cuh>

#include <cmath>
#include <limits>

namespace xinfer {

at::Tensor flashinfer_decode(const at::Tensor& q, const at::Tensor& k,
                            const at::Tensor& v, const at::Tensor& extent,
                            double scale) {
  TORCH_CHECK(q.is_cuda() && k.device() == q.device() && v.device() == q.device(),
              "xinfer decode requires Q/K/V on the same CUDA device");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
              v.scalar_type() == at::kBFloat16, "xinfer decode requires BF16");
  TORCH_CHECK(q.sizes() == at::IntArrayRef({1, 16, 1, 128}), "Expected decode Q [1,16,1,128]");
  TORCH_CHECK(k.sizes() == at::IntArrayRef({1, 8, 1024, 128}) && v.sizes() == k.sizes(),
              "Expected full-capacity K/V inputs [1,8,1024,128]");
  TORCH_CHECK(extent.device().is_cpu() && extent.scalar_type() == at::kByte && extent.dim() == 1,
              "Expected a CPU uint8 extent vector");
  const int64_t length = extent.size(0) - 1;
  TORCH_CHECK(length >= 1 && length <= 1024, "Invalid valid-prefix length");
  TORCH_CHECK(std::isfinite(scale) && scale > 0 && scale <= std::numeric_limits<float>::max(),
              "Attention scale must be finite and positive");
  TORCH_CHECK(!q.requires_grad() && !k.requires_grad() && !v.requires_grad(),
              "xinfer decode supports inference only");
  const c10::cuda::CUDAGuard guard(q.device());
  // The pinned kernel vector-loads 16 bytes. Normalize unsupported layouts;
  // the normal Qwen full cache and strided prefix need no repacking.
  auto aligned = [](const at::Tensor& t) {
    return t.stride(3) == 1 && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0 &&
           t.stride(1) % 8 == 0 && t.stride(2) % 8 == 0 &&
           t.stride(1) <= std::numeric_limits<uint32_t>::max() &&
           t.stride(2) <= std::numeric_limits<uint32_t>::max();
  };
  // clone(contiguous) also fixes an unaligned but already contiguous view.
  auto query = aligned(q) ? q : q.clone(at::MemoryFormat::Contiguous);
  auto key = k, value = v;
  if (!aligned(k) || !aligned(v) || k.stride(1) != v.stride(1) || k.stride(2) != v.stride(2)) {
    key = k.clone(at::MemoryFormat::Contiguous);
    value = v.clone(at::MemoryFormat::Contiguous);
  }
  auto out = at::empty({1, 16, 1, 128}, q.options());
  using Params = flashinfer::SingleDecodeParams<nv_bfloat16, nv_bfloat16, nv_bfloat16>;
  Params params;
  params.q = reinterpret_cast<nv_bfloat16*>(query.data_ptr());
  params.k = reinterpret_cast<nv_bfloat16*>(key.data_ptr());
  params.v = reinterpret_cast<nv_bfloat16*>(value.data_ptr());
  params.o = reinterpret_cast<nv_bfloat16*>(out.data_ptr());
  params.kv_len = length;
  params.num_qo_heads = 16;
  params.num_kv_heads = 8;
  params.q_stride_n = 16 * 128;
  params.q_stride_h = query.stride(1);
  params.kv_stride_n = key.stride(2);
  params.kv_stride_h = key.stride(1);
  params.window_left = -1;
  params.sm_scale = static_cast<float>(scale);
  // The upstream dispatcher uses >=256 tokens per partition. Each partition
  // needs Hq*D BF16 output plus Hq float LSE; ceil(L/256) is a safe upper bound.
  at::Tensor scratch;
  nv_bfloat16* tmp = nullptr;
  if (length > 256) {
    const int64_t chunks = (length + 255) / 256;
    scratch = at::empty({chunks * 16 * (128 * 2 + 4)}, q.options().dtype(at::kByte));
    tmp = reinterpret_cast<nv_bfloat16*>(scratch.data_ptr());
  }
  using Attention = flashinfer::DefaultAttention<false, false, false, false>;
  C10_CUDA_CHECK((flashinfer::SingleDecodeWithKVCacheDispatched<
      128, flashinfer::PosEncodingMode::kNone, Attention>(
          params, tmp, c10::cuda::getCurrentCUDAStream(q.get_device()).stream())));
  return out;
}

}  // namespace xinfer

TORCH_LIBRARY_FRAGMENT(xinfer, m) {
  m.def("flashinfer_decode(Tensor q, Tensor k, Tensor v, Tensor kv_extent, float scale) -> Tensor",
        {at::Tag::flexible_layout});
}
TORCH_LIBRARY_IMPL(xinfer, CUDA, m) {
  m.impl("flashinfer_decode", TORCH_FN(xinfer::flashinfer_decode));
}
