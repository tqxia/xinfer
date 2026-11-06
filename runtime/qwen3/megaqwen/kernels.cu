// Derived from Infatoshi/MegaQwen; provenance and license declaration: PROVENANCE.md.
/**
 * Fused Decode with __ldg() cached reads
 *
 * Strategy: Use __ldg() for read-only weight data to leverage texture cache
 * Based on minimal_sync but with cached loads
 */

#include "config.cuh"
#include "kernel.h"
#include <cooperative_groups.h>
#include <c10/cuda/CUDAException.h>
#include <cstddef>

namespace xinfer::megaqwen {
namespace cg = cooperative_groups;

// Configuration
// Grid size is selected from the actual device and compiled kernel occupancy.
constexpr int LDG_BLOCK_SIZE = 256;
constexpr int LDG_NUM_WARPS = LDG_BLOCK_SIZE / WARP_SIZE;
constexpr float LDG_RMS_EPS = 1e-6f;

// LM head
constexpr int LDG_LM_NUM_BLOCKS = 1184;
constexpr int LDG_LM_BLOCK_SIZE = 256;
constexpr int LDG_VOCAB_SIZE = 151936;

struct LDGLayerWeights {
  const __nv_bfloat16 *input_layernorm_weight;
  const __nv_bfloat16 *q_proj_weight;
  const __nv_bfloat16 *k_proj_weight;
  const __nv_bfloat16 *v_proj_weight;
  const __nv_bfloat16 *q_norm_weight;
  const __nv_bfloat16 *k_norm_weight;
  const __nv_bfloat16 *o_proj_weight;
  const __nv_bfloat16 *post_attn_layernorm_weight;
  const __nv_bfloat16 *gate_proj_weight;
  const __nv_bfloat16 *up_proj_weight;
  const __nv_bfloat16 *down_proj_weight;
  __nv_bfloat16 *k_cache;
  __nv_bfloat16 *v_cache;
};
static_assert(sizeof(LDGLayerWeights) == sizeof(LayerPointers));
static_assert(alignof(LDGLayerWeights) == alignof(LayerPointers));
static_assert(offsetof(LDGLayerWeights, k_cache) == offsetof(LayerPointers, k_cache));
static_assert(offsetof(LDGLayerWeights, v_cache) == offsetof(LayerPointers, v_cache));

// =============================================================================
// Helpers
// =============================================================================

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

__device__ __forceinline__ float ldg_silu(float x) { return x / (1.0f + expf(-x)); }

// =============================================================================
// Optimized matvec with __ldg and aggressive unrolling
// =============================================================================

__device__ void ldg_matvec_qkv(cg::grid_group &grid, const __nv_bfloat16 *__restrict__ input,
                               const __nv_bfloat16 *__restrict__ norm_weight,
                               const __nv_bfloat16 *__restrict__ q_weight,
                               const __nv_bfloat16 *__restrict__ k_weight,
                               const __nv_bfloat16 *__restrict__ v_weight,
                               float *__restrict__ g_normalized, float *__restrict__ g_residual,
                               float *__restrict__ q_out, float *__restrict__ k_out,
                               float *__restrict__ v_out) {
  int block_id = blockIdx.x;
  int num_blocks = gridDim.x;
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  // Block 0 does RMSNorm
  if (block_id == 0) {
    __shared__ float smem[HIDDEN_SIZE];
    __shared__ float smem_reduce[LDG_NUM_WARPS];

    float local_sum_sq = 0.0f;

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
      float v = ldg_mutable(input + i);
      smem[i] = v;
      g_residual[i] = v;
      local_sum_sq += v * v;
    }

    local_sum_sq = ldg_warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) {
      smem_reduce[warp_id] = local_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
      float sum = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
      sum = ldg_warp_reduce_sum(sum);
      if (lane_id == 0) {
        smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
      }
    }
    __syncthreads();

    float rstd = smem_reduce[0];

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
      float w = __bfloat162float(__ldg(norm_weight + i));
      g_normalized[i] = ldg_round(ldg_round(smem[i] * rstd) * w);
    }
  }

  grid.sync();

  // QKV projection with vec4 and __ldg
  constexpr int TOTAL_ROWS = Q_SIZE + KV_SIZE + KV_SIZE;
  int rows_per_block = (TOTAL_ROWS + num_blocks - 1) / num_blocks;
  int row_start = block_id * rows_per_block;
  int row_end = min(row_start + rows_per_block, TOTAL_ROWS);

  for (int m_base = row_start; m_base < row_end; m_base += LDG_NUM_WARPS) {
    int m = m_base + warp_id;

    if (m < row_end) {
      const __nv_bfloat16 *weight_row;
      float *output_ptr;

      if (m < Q_SIZE) {
        weight_row = q_weight + m * HIDDEN_SIZE;
        output_ptr = q_out + m;
      } else if (m < Q_SIZE + KV_SIZE) {
        weight_row = k_weight + (m - Q_SIZE) * HIDDEN_SIZE;
        output_ptr = k_out + (m - Q_SIZE);
      } else {
        weight_row = v_weight + (m - Q_SIZE - KV_SIZE) * HIDDEN_SIZE;
        output_ptr = v_out + (m - Q_SIZE - KV_SIZE);
      }

      // Use vec4 loads with __ldg through uint2
      float sum = 0.0f;
#pragma unroll 8
      for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
        uint2 w_u2 = __ldg(reinterpret_cast<const uint2 *>(weight_row + k));
        __nv_bfloat16 *w_ptr = reinterpret_cast<__nv_bfloat16 *>(&w_u2);

        sum += __bfloat162float(w_ptr[0]) * g_normalized[k] +
               __bfloat162float(w_ptr[1]) * g_normalized[k + 1] +
               __bfloat162float(w_ptr[2]) * g_normalized[k + 2] +
               __bfloat162float(w_ptr[3]) * g_normalized[k + 3];
      }

      sum = ldg_warp_reduce_sum(sum);
      if (lane_id == 0) {
        *output_ptr = ldg_round(sum);
      }
    }
  }

  grid.sync();
}

// =============================================================================
// QK Norm + RoPE + KV Cache
// =============================================================================

__device__ void ldg_qk_norm_rope_cache(
    cg::grid_group &grid, float *__restrict__ q, float *__restrict__ k, const float *v,
    const __nv_bfloat16 *__restrict__ q_norm_weight,
    const __nv_bfloat16 *__restrict__ k_norm_weight, const __nv_bfloat16 *__restrict__ cos_table,
    const __nv_bfloat16 *__restrict__ sin_table, __nv_bfloat16 *__restrict__ k_cache,
    __nv_bfloat16 *__restrict__ v_cache, int position, int max_seq_len) {
  int block_id = blockIdx.x;
  int num_blocks = gridDim.x;
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  const __nv_bfloat16 *cos_pos = cos_table + position * HEAD_DIM;
  const __nv_bfloat16 *sin_pos = sin_table + position * HEAD_DIM;

  // Process Q heads
  int q_heads_per_block = (NUM_Q_HEADS + num_blocks - 1) / num_blocks;
  int q_head_start = block_id * q_heads_per_block;
  int q_head_end = min(q_head_start + q_heads_per_block, NUM_Q_HEADS);

  for (int h = q_head_start + warp_id; h < q_head_end; h += LDG_NUM_WARPS) {
    float *q_head = q + h * HEAD_DIM;

    float sum_sq = 0.0f;
    for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
      sum_sq += q_head[i] * q_head[i];
    }
    sum_sq = ldg_warp_reduce_sum(sum_sq);
    float scale = rsqrtf(sum_sq / float(HEAD_DIM) + LDG_RMS_EPS);
    scale = __shfl_sync(0xffffffff, scale, 0);

    float q_local[HEAD_DIM / WARP_SIZE];
#pragma unroll
    for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
      q_local[j] =
          ldg_round(ldg_round(q_head[i] * scale) * __bfloat162float(__ldg(q_norm_weight + i)));
    }

#pragma unroll
    for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
      float cos_v = __bfloat162float(__ldg(cos_pos + i));
      float sin_v = __bfloat162float(__ldg(sin_pos + i));

      int pair_offset = (i < HEAD_DIM / 2) ? HEAD_DIM / 2 : -HEAD_DIM / 2;
      int pair_idx = i + pair_offset;
      int pair_j = pair_idx / WARP_SIZE;
      float pair_v = __shfl_sync(0xffffffff, q_local[pair_j], pair_idx % WARP_SIZE);

      if (i < HEAD_DIM / 2) {
        q_head[i] = ldg_round(ldg_round(q_local[j] * cos_v) - ldg_round(pair_v * sin_v));
      } else {
        q_head[i] = ldg_round(ldg_round(pair_v * sin_v) + ldg_round(q_local[j] * cos_v));
      }
    }
  }

  // Process K heads + cache
  int k_heads_per_block = (NUM_KV_HEADS + num_blocks - 1) / num_blocks;
  int k_head_start = block_id * k_heads_per_block;
  int k_head_end = min(k_head_start + k_heads_per_block, NUM_KV_HEADS);

  for (int h = k_head_start + warp_id; h < k_head_end; h += LDG_NUM_WARPS) {
    float *k_head = k + h * HEAD_DIM;
    const float *v_head = v + h * HEAD_DIM;
    __nv_bfloat16 *k_cache_head = k_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;
    __nv_bfloat16 *v_cache_head = v_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;

    float sum_sq = 0.0f;
    for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
      sum_sq += k_head[i] * k_head[i];
    }
    sum_sq = ldg_warp_reduce_sum(sum_sq);
    float scale = rsqrtf(sum_sq / float(HEAD_DIM) + LDG_RMS_EPS);
    scale = __shfl_sync(0xffffffff, scale, 0);

    float k_local[HEAD_DIM / WARP_SIZE];
#pragma unroll
    for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
      k_local[j] =
          ldg_round(ldg_round(k_head[i] * scale) * __bfloat162float(__ldg(k_norm_weight + i)));
    }

#pragma unroll
    for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
      float cos_v = __bfloat162float(__ldg(cos_pos + i));
      float sin_v = __bfloat162float(__ldg(sin_pos + i));

      int pair_offset = (i < HEAD_DIM / 2) ? HEAD_DIM / 2 : -HEAD_DIM / 2;
      int pair_idx = i + pair_offset;
      int pair_j = pair_idx / WARP_SIZE;
      float pair_v = __shfl_sync(0xffffffff, k_local[pair_j], pair_idx % WARP_SIZE);

      float k_final;
      if (i < HEAD_DIM / 2) {
        k_final = ldg_round(ldg_round(k_local[j] * cos_v) - ldg_round(pair_v * sin_v));
      } else {
        k_final = ldg_round(ldg_round(pair_v * sin_v) + ldg_round(k_local[j] * cos_v));
      }
      k_head[i] = k_final;
      k_cache_head[i] = __float2bfloat16(k_final);
      v_cache_head[i] = __float2bfloat16(v_head[i]);
    }
  }

  grid.sync();
}

// =============================================================================
// Streaming attention with optional KV partitions across resident blocks
// =============================================================================

// Prefetch immutable weights while other blocks compute attention.
__device__ void ldg_prefetch_weights_l2(const __nv_bfloat16 *__restrict__ weights,
                                        int num_elements) {
  // One explicit L2 prefetch per 128-byte line. An unused __ldg result
  // can be optimized away and is not a reliable prefetch primitive.
  for (int i = threadIdx.x * 64; i < num_elements; i += LDG_BLOCK_SIZE * 64) {
    asm volatile("prefetch.global.L2 [%0];" ::"l"(weights + i));
  }
}

// One block computes one head's contiguous KV partition. Its eight warps retain
// the original strided online-softmax loop and merge through shared memory.
__device__ void ldg_attention_partition(const float *q, const __nv_bfloat16 *__restrict__ k_cache,
                                        const __nv_bfloat16 *__restrict__ v_cache, float *attn_out,
                                        const AttentionPartials &partials, int cache_len,
                                        int max_seq_len, float attn_scale, int partitions) {
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
__device__ void ldg_attention_merge(const AttentionPartials &partials, float *attn_out,
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

__device__ void ldg_attention_prefetch(int attention_blocks,
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

__device__ void ldg_attention(cg::grid_group &grid, const float *q,
                              const __nv_bfloat16 *__restrict__ k_cache,
                              const __nv_bfloat16 *__restrict__ v_cache, float *attn_out,
                              const AttentionPartials &partials, int cache_len, int max_seq_len,
                              float attn_scale, int partitions,
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

// =============================================================================
// O Projection + Residual + PostNorm + MLP (all with __ldg)
// =============================================================================

__device__ void ldg_o_proj_postnorm_mlp(
    cg::grid_group &grid, const __nv_bfloat16 *__restrict__ o_weight,
    const __nv_bfloat16 *__restrict__ post_norm_weight,
    const __nv_bfloat16 *__restrict__ gate_weight, const __nv_bfloat16 *__restrict__ up_weight,
    const __nv_bfloat16 *__restrict__ down_weight, const float *attn_out,
    float *__restrict__ g_residual, float *__restrict__ g_activations,
    float *__restrict__ g_mlp_intermediate, __nv_bfloat16 *__restrict__ hidden_out) {
  int block_id = blockIdx.x;
  int num_blocks = gridDim.x;
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  // O Projection + Residual
  int hid_per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
  int hid_start = block_id * hid_per_block;
  int hid_end = min(hid_start + hid_per_block, HIDDEN_SIZE);

  for (int m_base = hid_start; m_base < hid_end; m_base += LDG_NUM_WARPS) {
    int m = m_base + warp_id;

    if (m < hid_end) {
      const __nv_bfloat16 *o_row = o_weight + m * Q_SIZE;

      float sum = 0.0f;
#pragma unroll 8
      for (int k = lane_id * 4; k < Q_SIZE; k += WARP_SIZE * 4) {
        uint2 w_u2 = __ldg(reinterpret_cast<const uint2 *>(o_row + k));
        __nv_bfloat16 *w_ptr = reinterpret_cast<__nv_bfloat16 *>(&w_u2);

        sum += __bfloat162float(w_ptr[0]) * attn_out[k] +
               __bfloat162float(w_ptr[1]) * attn_out[k + 1] +
               __bfloat162float(w_ptr[2]) * attn_out[k + 2] +
               __bfloat162float(w_ptr[3]) * attn_out[k + 3];
      }

      sum = ldg_warp_reduce_sum(sum);
      if (lane_id == 0) {
        g_activations[m] = ldg_round(ldg_round(sum) + g_residual[m]);
      }
    }
  }

  grid.sync();

  // Post-attention RMSNorm
  if (block_id == 0) {
    __shared__ float smem_reduce[LDG_NUM_WARPS];

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
      float v = g_activations[i];
      g_residual[i] = v;
      local_sum_sq += v * v;
    }

    local_sum_sq = ldg_warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) {
      smem_reduce[warp_id] = local_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
      float sum = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
      sum = ldg_warp_reduce_sum(sum);
      if (lane_id == 0) {
        smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
      }
    }
    __syncthreads();

    float rstd = smem_reduce[0];

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
      float w = __bfloat162float(__ldg(post_norm_weight + i));
      g_activations[i] = ldg_round(ldg_round(g_residual[i] * rstd) * w);
    }
  }

  grid.sync();

  // Gate + Up + SiLU
  int int_per_block = (INTERMEDIATE_SIZE + num_blocks - 1) / num_blocks;
  int int_start = block_id * int_per_block;
  int int_end = min(int_start + int_per_block, INTERMEDIATE_SIZE);

  for (int m_base = int_start; m_base < int_end; m_base += LDG_NUM_WARPS) {
    int m = m_base + warp_id;

    if (m < int_end) {
      const __nv_bfloat16 *gate_row = gate_weight + m * HIDDEN_SIZE;
      const __nv_bfloat16 *up_row = up_weight + m * HIDDEN_SIZE;

      float gate_sum = 0.0f, up_sum = 0.0f;

#pragma unroll 8
      for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
        uint2 g_u2 = __ldg(reinterpret_cast<const uint2 *>(gate_row + k));
        uint2 u_u2 = __ldg(reinterpret_cast<const uint2 *>(up_row + k));
        __nv_bfloat16 *g_ptr = reinterpret_cast<__nv_bfloat16 *>(&g_u2);
        __nv_bfloat16 *u_ptr = reinterpret_cast<__nv_bfloat16 *>(&u_u2);

        gate_sum += __bfloat162float(g_ptr[0]) * g_activations[k] +
                    __bfloat162float(g_ptr[1]) * g_activations[k + 1] +
                    __bfloat162float(g_ptr[2]) * g_activations[k + 2] +
                    __bfloat162float(g_ptr[3]) * g_activations[k + 3];

        up_sum += __bfloat162float(u_ptr[0]) * g_activations[k] +
                  __bfloat162float(u_ptr[1]) * g_activations[k + 1] +
                  __bfloat162float(u_ptr[2]) * g_activations[k + 2] +
                  __bfloat162float(u_ptr[3]) * g_activations[k + 3];
      }

      gate_sum = ldg_warp_reduce_sum(gate_sum);
      up_sum = ldg_warp_reduce_sum(up_sum);

      if (lane_id == 0) {
        g_mlp_intermediate[m] =
            ldg_round(ldg_round(ldg_silu(ldg_round(gate_sum))) * ldg_round(up_sum));
      }
    }
  }

  grid.sync();

  // Down projection + residual
  for (int m_base = hid_start; m_base < hid_end; m_base += LDG_NUM_WARPS) {
    int m = m_base + warp_id;

    if (m < hid_end) {
      const __nv_bfloat16 *down_row = down_weight + m * INTERMEDIATE_SIZE;

      float sum = 0.0f;
#pragma unroll 8
      for (int k = lane_id * 4; k < INTERMEDIATE_SIZE; k += WARP_SIZE * 4) {
        uint2 d_u2 = __ldg(reinterpret_cast<const uint2 *>(down_row + k));
        __nv_bfloat16 *d_ptr = reinterpret_cast<__nv_bfloat16 *>(&d_u2);

        sum += __bfloat162float(d_ptr[0]) * g_mlp_intermediate[k] +
               __bfloat162float(d_ptr[1]) * g_mlp_intermediate[k + 1] +
               __bfloat162float(d_ptr[2]) * g_mlp_intermediate[k + 2] +
               __bfloat162float(d_ptr[3]) * g_mlp_intermediate[k + 3];
      }

      sum = ldg_warp_reduce_sum(sum);
      if (lane_id == 0) {
        hidden_out[m] = __float2bfloat16(ldg_round(sum) + g_residual[m]);
      }
    }
  }

  grid.sync();
}

// =============================================================================
// Main Kernel
// =============================================================================

__global__ void __launch_bounds__(LDG_BLOCK_SIZE, 1)
    ldg_decode_kernel(int input_token_id, const __nv_bfloat16 *__restrict__ embed_weight,
                      const LDGLayerWeights *__restrict__ layer_weights,
                      const __nv_bfloat16 *__restrict__ final_norm_weight,
                      const __nv_bfloat16 *__restrict__ cos_table,
                      const __nv_bfloat16 *__restrict__ sin_table,
                      __nv_bfloat16 *__restrict__ hidden_buffer, float *__restrict__ g_activations,
                      float *__restrict__ g_residual, float *__restrict__ g_q,
                      float *__restrict__ g_k, float *__restrict__ g_v,
                      float *__restrict__ g_attn_out, float *__restrict__ g_mlp_intermediate,
                      float *__restrict__ g_normalized, AttentionPartials attention_partials,
                      int attention_partitions, int num_layers, int position, int cache_len,
                      int max_seq_len, float attn_scale) {
  cg::grid_group grid = cg::this_grid();
  int block_id = blockIdx.x;
  int num_blocks = gridDim.x;

  // Embedding lookup with __ldg
  const __nv_bfloat16 *embed_row = embed_weight + input_token_id * HIDDEN_SIZE;
  for (int i = block_id * LDG_BLOCK_SIZE + threadIdx.x; i < HIDDEN_SIZE;
       i += num_blocks * LDG_BLOCK_SIZE) {
    hidden_buffer[i] = __ldg(embed_row + i);
  }
  grid.sync();

  for (int layer = 0; layer < num_layers; layer++) {
    const LDGLayerWeights &w = layer_weights[layer];
    __nv_bfloat16 *layer_k_cache = w.k_cache;
    __nv_bfloat16 *layer_v_cache = w.v_cache;

    ldg_matvec_qkv(grid, hidden_buffer, w.input_layernorm_weight, w.q_proj_weight, w.k_proj_weight,
                   w.v_proj_weight, g_activations, g_residual, g_q, g_k, g_v);

    ldg_qk_norm_rope_cache(grid, g_q, g_k, g_v, w.q_norm_weight, w.k_norm_weight, cos_table,
                           sin_table, layer_k_cache, layer_v_cache, position, max_seq_len);

    ldg_attention(grid, g_q, layer_k_cache, layer_v_cache, g_attn_out, attention_partials,
                  cache_len, max_seq_len, attn_scale, attention_partitions, w.o_proj_weight,
                  w.gate_proj_weight, w.up_proj_weight);

    ldg_o_proj_postnorm_mlp(grid, w.o_proj_weight, w.post_attn_layernorm_weight, w.gate_proj_weight,
                            w.up_proj_weight, w.down_proj_weight, g_attn_out, g_residual,
                            g_activations, g_mlp_intermediate, hidden_buffer);
  }

  // Final RMSNorm
  if (block_id == 0) {
    __shared__ float smem_reduce[LDG_NUM_WARPS];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
      float v = __bfloat162float(hidden_buffer[i]);
      g_activations[i] = v;
      local_sum_sq += v * v;
    }

    local_sum_sq = ldg_warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) {
      smem_reduce[warp_id] = local_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
      float sum = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
      sum = ldg_warp_reduce_sum(sum);
      if (lane_id == 0) {
        smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
      }
    }
    __syncthreads();

    float rstd = smem_reduce[0];

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
      float wt = __bfloat162float(__ldg(final_norm_weight + i));
      g_normalized[i] = ldg_round(ldg_round(g_activations[i] * rstd) * wt);
    }
  }
}

// =============================================================================
// LM Head (same structure)
// =============================================================================

// Kernel to compute full logits (for KL divergence measurement)
__global__ void ldg_lm_head_logits(const float *hidden, const __nv_bfloat16 *__restrict__ weight,
                                   __nv_bfloat16 *__restrict__ logits) {
  __shared__ float s_hidden[HIDDEN_SIZE];

  for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_LM_BLOCK_SIZE) {
    s_hidden[i] = hidden[i];
  }
  __syncthreads();

  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  int rows_per_block = (LDG_VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
  int row_start = blockIdx.x * rows_per_block;
  int row_end = min(row_start + rows_per_block, LDG_VOCAB_SIZE);

  for (int m = row_start + warp_id; m < row_end; m += LDG_LM_BLOCK_SIZE / WARP_SIZE) {
    const __nv_bfloat16 *w_row = weight + m * HIDDEN_SIZE;

    float sum = 0.0f;
#pragma unroll 8
    for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
      uint2 w_u2 = __ldg(reinterpret_cast<const uint2 *>(w_row + k));
      __nv_bfloat16 *w_ptr = reinterpret_cast<__nv_bfloat16 *>(&w_u2);

      sum += __bfloat162float(w_ptr[0]) * s_hidden[k] +
             __bfloat162float(w_ptr[1]) * s_hidden[k + 1] +
             __bfloat162float(w_ptr[2]) * s_hidden[k + 2] +
             __bfloat162float(w_ptr[3]) * s_hidden[k + 3];
    }
    sum = ldg_warp_reduce_sum(sum);

    if (lane_id == 0) {
      logits[m] = __float2bfloat16_rn(sum);
    }
  }
}

// Query once at backend construction; a grid barrier requires co-residency.
LaunchInfo query_launch_info() {
  int device = 0, cooperative = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device));
  TORCH_CHECK(cooperative, "MegaQwen requires cooperative CUDA launch support");
  int active = 0;
  C10_CUDA_CHECK(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, ldg_decode_kernel, LDG_BLOCK_SIZE, 0));
  cudaFuncAttributes attributes{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, ldg_decode_kernel));
  const int blocks = properties.multiProcessorCount;
  TORCH_CHECK(blocks >= NUM_Q_HEADS && active > 0 && blocks <= blocks * active,
              "MegaQwen cooperative grid cannot reside concurrently");
  return {properties.multiProcessorCount, active, blocks, LDG_BLOCK_SIZE, attributes.numRegs,
          int(attributes.sharedSizeBytes)};
}

int attention_partition_count(int cache_len, int grid_blocks) {
  int partitions = 1;
  while (partitions < kMaxAttentionPartitions &&
         cache_len > partitions * kAttentionPartitionTokens &&
         grid_blocks >= NUM_Q_HEADS * partitions * 2)
    partitions *= 2;
  return partitions;
}

void launch_logits(int token, int position, int capacity, const LayerPointers *layers,
                   const void *embedding, const void *final_norm, const void *lm_head,
                   const void *cos_table, const void *sin_table, const Workspace &workspace,
                   int grid_blocks, cudaStream_t stream) {
  const auto *device_layers = reinterpret_cast<const LDGLayerWeights *>(layers);
  int num_layers = 28;
  int cache_length = position + 1;
  int partitions = attention_partition_count(cache_length, grid_blocks);
  float scale = 1.0f / sqrtf(128.0f);
  Workspace buffers = workspace;
  void *arguments[] = {&token,
                       &embedding,
                       &device_layers,
                       &final_norm,
                       &cos_table,
                       &sin_table,
                       &buffers.hidden,
                       &buffers.activations,
                       &buffers.residual,
                       &buffers.q,
                       &buffers.key,
                       &buffers.value,
                       &buffers.attention,
                       &buffers.mlp,
                       &buffers.normalized,
                       &buffers.attention_partials,
                       &partitions,
                       &num_layers,
                       &position,
                       &cache_length,
                       &capacity,
                       &scale};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void *>(ldg_decode_kernel),
                                             dim3(grid_blocks), dim3(LDG_BLOCK_SIZE), arguments, 0,
                                             stream));
  // Compute every vocabulary row once. The caller handles sampling.
  ldg_lm_head_logits<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
      workspace.normalized, static_cast<const __nv_bfloat16 *>(lm_head),
      static_cast<__nv_bfloat16 *>(workspace.logits));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#ifdef XINFER_MEGAQWEN_TESTING
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
#endif

} // namespace xinfer::megaqwen
