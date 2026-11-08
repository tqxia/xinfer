// Derived from Infatoshi/MegaQwen; provenance and license declaration: PROVENANCE.md.
/**
 * Fused Decode with __ldg() cached reads
 *
 * Strategy: Use __ldg() for read-only weight data to leverage texture cache
 * Based on minimal_sync but with cached loads
 */

#include "attention.cuh"
#include <c10/cuda/CUDAException.h>
#include <c10/macros/Macros.h>
#include <cstddef>

namespace xinfer::megaqwen {

// Configuration
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

__device__ __forceinline__ void
ldg_decode_body(int input_token_id, const __nv_bfloat16 *__restrict__ embed_weight,
                const LDGLayerWeights *__restrict__ layer_weights,
                const __nv_bfloat16 *__restrict__ final_norm_weight,
                const __nv_bfloat16 *__restrict__ cos_table,
                const __nv_bfloat16 *__restrict__ sin_table,
                __nv_bfloat16 *__restrict__ hidden_buffer, float *__restrict__ g_activations,
                float *__restrict__ g_residual, float *__restrict__ g_q, float *__restrict__ g_k,
                float *__restrict__ g_v, float *__restrict__ g_attn_out,
                float *__restrict__ g_mlp_intermediate, float *__restrict__ g_normalized,
                AttentionPartials attention_partials, int attention_partitions, int num_layers,
                int position, int cache_len, int max_seq_len, float attn_scale) {
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

// The stateless operator supplies live tensor addresses in kernel parameters.
// Grid-constant arguments avoid materializing a 2,912-byte table per thread.
static_assert(sizeof(LayerPointerTable) + sizeof(Workspace) + 256 <= 4096,
              "MegaQwen operator arguments must fit the CUDA kernel parameter budget");

__global__ void __launch_bounds__(LDG_BLOCK_SIZE, 1)
    ldg_decode_token_kernel(const int64_t *input_ids,
                            const __grid_constant__ LayerPointerTable layer_table,
                            const __grid_constant__ Workspace workspace,
                            const __nv_bfloat16 *embedding, const __nv_bfloat16 *final_norm,
                            const __nv_bfloat16 *cos_table, const __nv_bfloat16 *sin_table,
                            int position, int capacity, int partitions, float attn_scale) {
  // IDs remain on the device: the CPU only supplies prefix length metadata.
  int64_t token = input_ids[0];
  CUDA_KERNEL_ASSERT(token >= 0 && token < LDG_VOCAB_SIZE);
  const auto *layers = reinterpret_cast<const LDGLayerWeights *>(layer_table.layers);
  ldg_decode_body(int(token), embedding, layers, final_norm, cos_table, sin_table,
                  static_cast<__nv_bfloat16 *>(workspace.hidden), workspace.activations,
                  workspace.residual, workspace.q, workspace.key, workspace.value,
                  workspace.attention, workspace.mlp, workspace.normalized,
                  workspace.attention_partials, partitions, kDecoderLayers, position, position + 1,
                  capacity, attn_scale);
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

// A grid barrier requires co-residency for the actual compiled entry point.
LaunchInfo query_kernel_launch_info(const void *kernel) {
  int device = 0, cooperative = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device));
  TORCH_CHECK(cooperative, "MegaQwen requires cooperative CUDA launch support");
  int active = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, LDG_BLOCK_SIZE, 0));
  cudaFuncAttributes attributes{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
  const int blocks = properties.multiProcessorCount;
  TORCH_CHECK(blocks >= NUM_Q_HEADS && active > 0 && blocks <= blocks * active,
              "MegaQwen cooperative grid cannot reside concurrently");
  return {properties.multiProcessorCount, active, blocks, LDG_BLOCK_SIZE, attributes.numRegs,
          int(attributes.sharedSizeBytes)};
}

LaunchInfo query_token_launch_info() {
  return query_kernel_launch_info(reinterpret_cast<const void *>(ldg_decode_token_kernel));
}

int attention_partition_count(int cache_len, int grid_blocks) {
  int partitions = 1;
  while (partitions < kMaxAttentionPartitions &&
         cache_len > partitions * kAttentionPartitionTokens &&
         grid_blocks >= NUM_Q_HEADS * partitions * 2)
    partitions *= 2;
  return partitions;
}

void launch_token_logits(const int64_t *ids, int position, int capacity,
                         const LayerPointerTable &layers, const void *embedding,
                         const void *final_norm, const void *lm_head, const void *cos_table,
                         const void *sin_table, const Workspace &workspace, int grid_blocks,
                         cudaStream_t stream) {
  LayerPointerTable table = layers;
  Workspace buffers = workspace;
  int partitions = attention_partition_count(position + 1, grid_blocks);
  float scale = 1.0f / sqrtf(128.0f);
  void *arguments[] = {&ids,       &table,    &buffers,  &embedding,  &final_norm, &cos_table,
                       &sin_table, &position, &capacity, &partitions, &scale};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void *>(ldg_decode_token_kernel),
                                             dim3(grid_blocks), dim3(LDG_BLOCK_SIZE), arguments, 0,
                                             stream));
  ldg_lm_head_logits<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
      workspace.normalized, static_cast<const __nv_bfloat16 *>(lm_head),
      static_cast<__nv_bfloat16 *>(workspace.logits));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace xinfer::megaqwen
