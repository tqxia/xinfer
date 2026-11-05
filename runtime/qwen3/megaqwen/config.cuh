#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>

// Required dimensional subset of the upstream config.cuh; see PROVENANCE.md.
constexpr int WARP_SIZE = 32;
constexpr int HIDDEN_SIZE = 1024;
constexpr int INTERMEDIATE_SIZE = 3072;
constexpr int NUM_Q_HEADS = 16;
constexpr int NUM_KV_HEADS = 8;
constexpr int HEAD_DIM = 128;
constexpr int Q_SIZE = NUM_Q_HEADS * HEAD_DIM;
constexpr int KV_SIZE = NUM_KV_HEADS * HEAD_DIM;
