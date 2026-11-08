#pragma once

#include "qwen3/kernels/megaqwen/kernel.h"

namespace xinfer::megaqwen {

// Exercise the production device attention without model weights.
LaunchInfo query_attention_launch_info();
void launch_attention_for_test(const float *q, const void *k_cache, const void *v_cache,
                               float *output, const AttentionPartials &partials, int cache_len,
                               int capacity, int partitions, int grid_blocks, cudaStream_t stream);

} // namespace xinfer::megaqwen
