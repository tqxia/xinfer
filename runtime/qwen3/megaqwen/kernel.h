#pragma once

#include <cuda_runtime_api.h>

namespace xinfer::megaqwen {

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
void launch_logits(int token, int position, int capacity, const LayerPointers *layers,
                   const void *embedding, const void *final_norm, const void *lm_head,
                   const void *cos_table, const void *sin_table, const Workspace &workspace,
                   int grid_blocks, cudaStream_t stream);

} // namespace xinfer::megaqwen
