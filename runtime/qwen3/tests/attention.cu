#include "attention.h"
#include "qwen3/kernels/megaqwen/attention.cuh"

#include <c10/cuda/CUDAException.h>

namespace xinfer::megaqwen {

__global__ void __launch_bounds__(LDG_BLOCK_SIZE, 1)
    attention_test_kernel(const float *q, const __nv_bfloat16 *k_cache,
                          const __nv_bfloat16 *v_cache, float *output, AttentionPartials partials,
                          int cache_len, int capacity, int partitions) {
  cg::grid_group grid = cg::this_grid();
  ldg_attention(grid, q, k_cache, v_cache, output, partials, cache_len, capacity,
                1.0f / sqrtf(float(HEAD_DIM)), partitions, nullptr, nullptr, nullptr);
}

LaunchInfo query_attention_launch_info() {
  int device = 0, cooperative = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device));
  TORCH_CHECK(cooperative, "Attention validation requires cooperative CUDA launch support");
  int active = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, attention_test_kernel,
                                                               LDG_BLOCK_SIZE, 0));
  cudaFuncAttributes attributes{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, attention_test_kernel));
  TORCH_CHECK(properties.multiProcessorCount >= NUM_Q_HEADS && active > 0,
              "Attention validation cooperative grid cannot reside concurrently");
  return {properties.multiProcessorCount,
          active,
          properties.multiProcessorCount,
          LDG_BLOCK_SIZE,
          attributes.numRegs,
          int(attributes.sharedSizeBytes)};
}

void launch_attention_for_test(const float *q, const void *k_cache, const void *v_cache,
                               float *output, const AttentionPartials &partials, int cache_len,
                               int capacity, int partitions, int grid_blocks, cudaStream_t stream) {
  TORCH_CHECK(cache_len > 0 && cache_len <= capacity,
              "Attention validation requires a nonempty valid KV prefix");
  TORCH_CHECK((partitions == 1 || partitions == 2 || partitions == 4) &&
                  grid_blocks >= NUM_Q_HEADS * partitions,
              "Attention validation partitions do not fit the cooperative grid");
  const auto launch = query_attention_launch_info();
  TORCH_CHECK(grid_blocks <= launch.sm_count * launch.active_blocks_per_sm,
              "Attention validation cooperative grid exceeds device residency");
  const auto *keys = static_cast<const __nv_bfloat16 *>(k_cache);
  const auto *values = static_cast<const __nv_bfloat16 *>(v_cache);
  AttentionPartials scratch = partials;
  void *arguments[] = {&q, &keys, &values, &output, &scratch, &cache_len, &capacity, &partitions};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void *>(attention_test_kernel),
                                             dim3(grid_blocks), dim3(LDG_BLOCK_SIZE), arguments, 0,
                                             stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace xinfer::megaqwen
