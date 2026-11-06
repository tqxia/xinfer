#pragma once

#include <cuda_runtime_api.h>

namespace xinfer::megaqwen {

// The untuned policy keeps at most 128 KV positions per partition until it
// reaches the four-partition limit or runs out of resident attention blocks.
inline constexpr int kMaxAttentionPartitions = 4;
inline constexpr int kAttentionPartitionTokens = 128;

struct AttentionPartials {
  float *max_scores;      // [16, kMaxAttentionPartitions]
  float *sum_exp;         // [16, kMaxAttentionPartitions]
  float *weighted_values; // [16, kMaxAttentionPartitions, 128], unnormalized
};

// This table contains borrowed addresses, never copies of weights or KV storage.
struct LayerPointers {
  const void *input_layernorm_weight;
  const void *q_proj_weight;
  const void *k_proj_weight;
  const void *v_proj_weight;
  const void *q_norm_weight;
  const void *k_norm_weight;
  const void *o_proj_weight;
  const void *post_attn_layernorm_weight;
  const void *gate_proj_weight;
  const void *up_proj_weight;
  const void *down_proj_weight;
  void *k_cache;
  void *v_cache;
};

struct Workspace {
  void *hidden;
  float *activations;
  float *residual;
  float *q;
  float *key;
  float *value;
  float *attention;
  float *mlp;
  float *normalized;
  void *logits;
  AttentionPartials attention_partials;
};

struct LaunchInfo {
  int sm_count;
  int active_blocks_per_sm;
  int grid_blocks;
  int threads_per_block;
  int registers_per_thread;
  int static_shared_bytes;
};

LaunchInfo query_launch_info();
int attention_partition_count(int cache_len, int grid_blocks);
void launch_logits(int token, int position, int capacity, const LayerPointers *layers,
                   const void *embedding, const void *final_norm, const void *lm_head,
                   const void *cos_table, const void *sin_table, const Workspace &workspace,
                   int grid_blocks, cudaStream_t stream);

#ifdef XINFER_MEGAQWEN_TESTING
// Exercise the same device attention implementation without model weights.
LaunchInfo query_attention_launch_info();
void launch_attention_for_test(const float *q, const void *k_cache, const void *v_cache,
                               float *output, const AttentionPartials &partials, int cache_len,
                               int capacity, int partitions, int grid_blocks, cudaStream_t stream);
#endif

} // namespace xinfer::megaqwen
