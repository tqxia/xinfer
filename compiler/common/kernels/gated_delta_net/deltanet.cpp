#include "deltanet.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <algorithm>

namespace xinfer {

namespace {

constexpr int64_t kMaxBatchSize = 8;
constexpr int64_t kMaxTokens = 512;
constexpr int64_t kKeyHeads = 16;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kConvWidth = 4;

// Required opt-in shared memory for the BF16x3 Tensor Core chunked kernel (~54 KB).
constexpr size_t kMinChunkSmemBytes = 60 * 1024;

bool supports_tensorcore_chunked(int device_index) {
  const auto *props = at::cuda::getDeviceProperties(device_index);
  return props->major >= 8 && props->sharedMemPerBlockOptin >= kMinChunkSmemBytes;
}

} // namespace

at::Tensor batched_deltanet(const at::Tensor &qkv, const at::Tensor &b, const at::Tensor &a,
                            at::Tensor conv, at::Tensor state, const at::Tensor &weight,
                            const at::Tensor &A_log, const at::Tensor &dt_bias,
                            const at::Tensor &cu_query, const at::Tensor &cuda_cu_query) {
  TORCH_CHECK(qkv.dim() == 3 && qkv.size(0) >= 1 && qkv.size(0) <= kMaxBatchSize,
              "Expected QKV [B,T,channels] with B=1..8");
  const int64_t batch = qkv.size(0);
  const int64_t tokens = qkv.size(1);
  TORCH_CHECK(tokens >= 1 && tokens <= kMaxTokens, "Token length must be 1..512");

  TORCH_CHECK(b.dim() == 3 && b.size(0) == batch && b.size(1) == tokens && a.sizes() == b.sizes(),
              "Expected matching A/B [B,T,value_heads]");
  const int64_t value_heads = b.size(2);
  const int64_t channels = (2 * kKeyHeads + value_heads) * kHeadDim;
  TORCH_CHECK((value_heads == 16 || value_heads == 32 || value_heads == 48) &&
                  qkv.size(2) == channels,
              "Expected Hk=16, Hv=16, 32 or 48 and K=V=128");

  TORCH_CHECK(conv.sizes() == at::IntArrayRef({batch, channels, kConvWidth}) &&
                  weight.sizes() == at::IntArrayRef({channels, kConvWidth}),
              "Expected matching width-4 convolution state and weights");
  TORCH_CHECK(state.sizes() == at::IntArrayRef({batch, value_heads, kHeadDim, kHeadDim}) &&
                  A_log.sizes() == at::IntArrayRef({value_heads}) &&
                  dt_bias.sizes() == at::IntArrayRef({value_heads}),
              "Expected per-value-head recurrent state and decay parameters");

  const auto dtype = qkv.scalar_type();
  TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kFloat,
              "Expected matching BF16 or FP32 inputs and parameters");
  TORCH_CHECK(b.scalar_type() == dtype && a.scalar_type() == dtype && conv.scalar_type() == dtype &&
                  weight.scalar_type() == dtype &&
                  (A_log.scalar_type() == dtype || A_log.scalar_type() == at::kFloat) &&
                  (dt_bias.scalar_type() == dtype || dt_bias.scalar_type() == at::kFloat),
              "Expected matching BF16 or FP32 inputs and parameters");
  TORCH_CHECK(state.scalar_type() == at::kFloat, "Recurrent state must be FP32");

  TORCH_CHECK(qkv.is_cuda() && b.device() == qkv.device() && a.device() == qkv.device() &&
                  conv.device() == qkv.device() && state.device() == qkv.device() &&
                  weight.device() == qkv.device() && A_log.device() == qkv.device() &&
                  dt_bias.device() == qkv.device(),
              "Expected CUDA tensors on one device");
  TORCH_CHECK(!qkv.requires_grad() && !b.requires_grad() && !a.requires_grad() &&
                  !conv.requires_grad() && !state.requires_grad() && !weight.requires_grad() &&
                  !A_log.requires_grad() && !dt_bias.requires_grad(),
              "xinfer DeltaNet supports inference only");
  TORCH_CHECK(conv.is_contiguous() && state.is_contiguous(),
              "In-place DeltaNet state buffers must be contiguous");

  TORCH_CHECK(cu_query.device().is_cpu() && cu_query.scalar_type() == at::kInt &&
                  cu_query.is_contiguous() && cu_query.sizes() == at::IntArrayRef({batch + 1}),
              "Expected CPU int32 cumulative query lengths [B+1]");
  TORCH_CHECK(cuda_cu_query.device() == qkv.device() && cuda_cu_query.scalar_type() == at::kInt &&
                  cuda_cu_query.is_contiguous() &&
                  cuda_cu_query.sizes() == at::IntArrayRef({batch + 1}),
              "Expected CUDA int32 cumulative query lengths [B+1]");

  const auto *cuq = cu_query.const_data_ptr<int32_t>();
  TORCH_CHECK(cuq[0] == 0, "Cumulative query lengths must start at zero");
  int64_t max_len = 0;
  bool has_short = false;
  bool has_long = false;
  for (int64_t row = 0; row < batch; ++row) {
    const int64_t count = cuq[row + 1] - cuq[row];
    TORCH_CHECK(count >= 0 && count <= tokens, "Invalid query lengths");
    max_len = std::max(max_len, count);
    if (count > 0 && count < detail::kChunkThreshold)
      has_short = true;
    else if (count >= detail::kChunkThreshold)
      has_long = true;
  }
  if (max_len == 0)
    return at::zeros({batch, tokens, value_heads, kHeadDim}, qkv.options());

  const auto qkv_c = qkv.contiguous();
  const auto b_c = b.contiguous();
  const auto a_c = a.contiguous();
  const auto weight_c = weight.contiguous();
  const auto a_log_c = A_log.contiguous();
  const auto dt_bias_c = dt_bias.contiguous();
  auto output = at::empty({batch, tokens, value_heads, kHeadDim}, qkv.options());

  at::cuda::CUDAGuard device_guard(qkv.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(qkv.device().index());

  if (dtype == at::kBFloat16) {
    const bool use_chunked = has_long && supports_tensorcore_chunked(qkv.device().index());
    detail::launch_deltanet_bf16(
        qkv_c.const_data_ptr(), b_c.const_data_ptr(), a_c.const_data_ptr(), conv.data_ptr(),
        state.data_ptr<float>(), weight_c.const_data_ptr(), a_log_c.const_data_ptr(),
        dt_bias_c.const_data_ptr(), output.data_ptr(), cuda_cu_query.const_data_ptr<int32_t>(),
        batch, tokens, kKeyHeads, value_heads, a_log_c.scalar_type() == at::kFloat,
        dt_bias_c.scalar_type() == at::kFloat, has_short, has_long, use_chunked, stream);
  } else {
    // Keep FP32 on the exact IEEE FMA recurrent kernel (Tensor Core accumulators truncate).
    detail::launch_deltanet_fp32(qkv_c.const_data_ptr<float>(), b_c.const_data_ptr<float>(),
                                 a_c.const_data_ptr<float>(), conv.data_ptr<float>(),
                                 state.data_ptr<float>(), weight_c.const_data_ptr<float>(),
                                 a_log_c.const_data_ptr<float>(), dt_bias_c.const_data_ptr<float>(),
                                 output.data_ptr<float>(), cuda_cu_query.const_data_ptr<int32_t>(),
                                 batch, tokens, kKeyHeads, value_heads, stream);
  }
  return output;
}

} // namespace xinfer

TORCH_LIBRARY_FRAGMENT(xinfer, m) {
  m.def("batched_deltanet(Tensor qkv, Tensor b, Tensor a, Tensor(a!) conv, Tensor(b!) state, "
        "Tensor weight, Tensor A_log, Tensor dt_bias, Tensor cu_query, Tensor cuda_cu_query) "
        "-> Tensor",
        {at::Tag::flexible_layout});
}

TORCH_LIBRARY_IMPL(xinfer, CUDA, m) {
  m.impl("batched_deltanet", TORCH_FN(xinfer::batched_deltanet));
}
