#pragma once

#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

namespace xinfer {

namespace detail {

// Per-row active length threshold for dispatching BF16 rows to the fused BF16x3 Tensor Core
// chunked kernel. Keeping dispatch strictly per-row guarantees bitwise scalar-vs-batched parity.
constexpr int64_t kChunkThreshold = 16;

void launch_deltanet_bf16(const void *qkv, const void *b, const void *a, void *conv, float *state,
                          const void *weight, const void *A_log, const void *dt_bias, void *output,
                          const int32_t *cuda_cu_query, int64_t batch, int64_t tokens,
                          int64_t key_heads, int64_t value_heads, bool a_log_fp32,
                          bool dt_bias_fp32, bool has_short, bool has_long, bool use_chunked,
                          c10::cuda::CUDAStream stream);

void launch_deltanet_fp32(const float *qkv, const float *b, const float *a, float *conv,
                          float *state, const float *weight, const float *A_log,
                          const float *dt_bias, float *output, const int32_t *cuda_cu_query,
                          int64_t batch, int64_t tokens, int64_t key_heads, int64_t value_heads,
                          c10::cuda::CUDAStream stream);

} // namespace detail

at::Tensor batched_deltanet(const at::Tensor &qkv, const at::Tensor &b, const at::Tensor &a,
                            at::Tensor conv, at::Tensor state, const at::Tensor &weight,
                            const at::Tensor &A_log, const at::Tensor &dt_bias,
                            const at::Tensor &cu_query, const at::Tensor &cuda_cu_query);

} // namespace xinfer
