// Derived from Infatoshi/MegaQwen; provenance and license declaration: PROVENANCE.md.
// Shared device arithmetic used by decode and the standalone attention test.
#pragma once

#include "config.cuh"
#include "kernel.h"

namespace xinfer::megaqwen {
namespace cg = cooperative_groups;

constexpr int LDG_BLOCK_SIZE = 256;
constexpr int LDG_NUM_WARPS = LDG_BLOCK_SIZE / WARP_SIZE;

// Preserve eager BF16 tensor boundaries while using FP32 accumulators.
__device__ __forceinline__ float ldg_round(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

// These buffers are modified during this kernel invocation. Explicit coherent
// global loads avoid the read-only (__ldg / ld.global.nc) cache contract.
__device__ __forceinline__ float ldg_mutable(const __nv_bfloat16 *value) {
  return __bfloat162float(__ldcg(value));
}

__device__ __forceinline__ float ldg_mutable(const float *value) { return __ldcg(value); }

__device__ __forceinline__ float ldg_warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// =============================================================================
// Streaming attention with optional KV partitions across resident blocks
// =============================================================================

// Prefetch immutable weights while other blocks compute attention.
__device__ inline void ldg_prefetch_weights_l2(const __nv_bfloat16 *__restrict__ weights,
                                               int num_elements) {
  // One explicit L2 prefetch per 128-byte line. An unused __ldg result
  // can be optimized away and is not a reliable prefetch primitive.
  for (int i = threadIdx.x * 64; i < num_elements; i += LDG_BLOCK_SIZE * 64) {
    asm volatile("prefetch.global.L2 [%0];" ::"l"(weights + i));
  }
}

// One block computes one head's contiguous KV partition. Its eight warps retain
// the original strided online-softmax loop and merge through shared memory.
__device__ inline void ldg_attention_partition(const float *q,
                                               const __nv_bfloat16 *__restrict__ k_cache,
                                               const __nv_bfloat16 *__restrict__ v_cache,
                                               float *attn_out, const AttentionPartials &partials,
                                               int cache_len, int max_seq_len, float attn_scale,
                                               int partitions) {
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;
  int qh = blockIdx.x / partitions;
  int partition = blockIdx.x % partitions;
  int partition_size = (cache_len + partitions - 1) / partitions;
  int position_begin = partition * partition_size;
  int position_end = min(position_begin + partition_size, cache_len);

  // Shared memory for cross-warp reduction of online softmax
  __shared__ float s_max_score[LDG_NUM_WARPS];
  __shared__ float s_sum_exp[LDG_NUM_WARPS];
  __shared__ float s_out_acc[LDG_NUM_WARPS][HEAD_DIM];

  int kv_head = qh / (NUM_Q_HEADS / NUM_KV_HEADS);
  const float *q_head = q + qh * HEAD_DIM;
  float *out_head = attn_out + qh * HEAD_DIM;

  float max_score = -INFINITY;
  float sum_exp = 0.0f;
  float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  // Each warp processes a subset of cache positions
  for (int pos = position_begin + warp_id; pos < position_end; pos += LDG_NUM_WARPS) {
    const __nv_bfloat16 *k_pos = k_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;
    const __nv_bfloat16 *v_pos = v_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;

    // KV was written earlier in this cooperative kernel; use coherent loads.
    float score = 0.0f;
    for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
      score += q_head[d] * ldg_mutable(k_pos + d);
    }
    score = ldg_warp_reduce_sum(score) * attn_scale;
    score = __shfl_sync(0xffffffff, score, 0);

    float old_max = max_score;
    max_score = fmaxf(max_score, score);
    float exp_diff = expf(old_max - max_score);
    sum_exp = sum_exp * exp_diff + expf(score - max_score);

    float weight = expf(score - max_score);
#pragma unroll
    for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++) {
      out_acc[j] = out_acc[j] * exp_diff + weight * ldg_mutable(v_pos + d);
    }
  }

  // Store each warp's partial results to shared memory
  if (lane_id == 0) {
    s_max_score[warp_id] = max_score;
    s_sum_exp[warp_id] = sum_exp;
  }
#pragma unroll
  for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++) {
    s_out_acc[warp_id][d] = out_acc[j];
  }
  __syncthreads();

  // Warp 0 combines results from all warps
  if (warp_id == 0) {
    // Find global max across all warps
    float global_max = s_max_score[0];
    for (int w = 1; w < LDG_NUM_WARPS; w++) {
      if (s_max_score[w] > -INFINITY) { // Only consider warps that processed positions
        global_max = fmaxf(global_max, s_max_score[w]);
      }
    }

    // Rescale and sum the partial results
    float total_sum_exp = 0.0f;
    float final_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int w = 0; w < LDG_NUM_WARPS; w++) {
      if (s_max_score[w] > -INFINITY) { // Only consider warps that processed positions
        float scale = expf(s_max_score[w] - global_max);
        total_sum_exp += s_sum_exp[w] * scale;

#pragma unroll
        for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++) {
          final_out[j] += s_out_acc[w][d] * scale;
        }
      }
    }

    if (partitions == 1) {
      // Preserve the original short-context path and its BF16 boundary.
#pragma unroll
      for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++)
        out_head[d] = ldg_round(final_out[j] / total_sum_exp);
    } else {
      int index = qh * kMaxAttentionPartitions + partition;
      if (lane_id == 0) {
        partials.max_scores[index] = global_max;
        partials.sum_exp[index] = total_sum_exp;
      }
      // Empty partitions write (-inf, 0, zero-vector), so no earlier layer or
      // request can leak into the merge even when scratch starts poisoned.
#pragma unroll
      for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++)
        partials.weighted_values[index * HEAD_DIM + d] = final_out[j];
    }
  }
  __syncthreads();
}

// Merge independent partitions with the same max-rescaling used by the
// intra-block warp reduction. Only warp zero of each head's block is needed.
__device__ inline void ldg_attention_merge(const AttentionPartials &partials, float *attn_out,
                                           int partitions) {
  if (blockIdx.x >= NUM_Q_HEADS || threadIdx.x >= WARP_SIZE)
    return;
  int qh = blockIdx.x;
  int lane_id = threadIdx.x;
  int base = qh * kMaxAttentionPartitions;
  float max_score = -INFINITY;
  for (int partition = 0; partition < partitions; ++partition)
    max_score = fmaxf(max_score, ldg_mutable(partials.max_scores + base + partition));

  float sum_exp = 0.0f;
  float output[HEAD_DIM / WARP_SIZE] = {};
  for (int partition = 0; partition < partitions; ++partition) {
    int index = base + partition;
    float partition_max = ldg_mutable(partials.max_scores + index);
    if (partition_max > -INFINITY) {
      float scale = expf(partition_max - max_score);
      sum_exp += ldg_mutable(partials.sum_exp + index) * scale;
#pragma unroll
      for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, ++j)
        output[j] += ldg_mutable(partials.weighted_values + index * HEAD_DIM + d) * scale;
    }
  }
#pragma unroll
  for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, ++j)
    attn_out[qh * HEAD_DIM + d] = ldg_round(output[j] / sum_exp);
}

__device__ inline void ldg_attention_prefetch(int attention_blocks,
                                              const __nv_bfloat16 *__restrict__ o_weight,
                                              const __nv_bfloat16 *__restrict__ gate_weight,
                                              const __nv_bfloat16 *__restrict__ up_weight) {
  int prefetch_block_id = blockIdx.x - attention_blocks;
  int num_prefetch_blocks = gridDim.x - attention_blocks;
  // Partition all cache lines without assuming an RTX 3090 grid size.
  int group = min(2, prefetch_block_id * 3 / num_prefetch_blocks);
  int group_begin = (group * num_prefetch_blocks + 2) / 3;
  int group_end = ((group + 1) * num_prefetch_blocks + 2) / 3;
  int count = group_end - group_begin;
  int elements = group == 0 ? Q_SIZE * HIDDEN_SIZE : HIDDEN_SIZE * INTERMEDIATE_SIZE;
  const __nv_bfloat16 *weight = group == 0 ? o_weight : (group == 1 ? gate_weight : up_weight);
  int chunk = (elements + count - 1) / count;
  int start = (prefetch_block_id - group_begin) * chunk;
  if (start < elements)
    ldg_prefetch_weights_l2(weight + start, min(chunk, elements - start));
}

__device__ inline void ldg_attention(cg::grid_group &grid, const float *q,
                                     const __nv_bfloat16 *__restrict__ k_cache,
                                     const __nv_bfloat16 *__restrict__ v_cache, float *attn_out,
                                     const AttentionPartials &partials, int cache_len,
                                     int max_seq_len, float attn_scale, int partitions,
                                     const __nv_bfloat16 *__restrict__ o_weight,
                                     const __nv_bfloat16 *__restrict__ gate_weight,
                                     const __nv_bfloat16 *__restrict__ up_weight) {
  int attention_blocks = NUM_Q_HEADS * partitions;
  if (blockIdx.x < attention_blocks) {
    ldg_attention_partition(q, k_cache, v_cache, attn_out, partials, cache_len, max_seq_len,
                            attn_scale, partitions);
  } else if (o_weight != nullptr) {
    ldg_attention_prefetch(attention_blocks, o_weight, gate_weight, up_weight);
  }

  // All blocks follow the same barrier sequence. The first barrier publishes
  // partials; the second prevents O projection or the next layer reusing them
  // before every head's merge has completed. One partition needs no merge.
  grid.sync();
  if (partitions > 1) {
    ldg_attention_merge(partials, attn_out, partitions);
    grid.sync();
  }
}

} // namespace xinfer::megaqwen
