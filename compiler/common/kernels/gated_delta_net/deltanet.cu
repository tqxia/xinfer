#include "deltanet.h"

#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace xinfer::detail {
namespace {

constexpr int kHeadDim = 128;
constexpr int kValueTile = 32;
constexpr int kChunk = 32;
constexpr int kConvWidth = 4;
constexpr int kRawRows = kChunk + kConvWidth - 1;
constexpr int kChunkWarps = 8;
constexpr int kChunkThreads = kChunkWarps * 32;
constexpr float kInvSqrtHeadDim = 0.08838834764831845f; // 1 / sqrt(128)
constexpr float kNormEps = 1e-6f;
constexpr float kLog2E = 1.4426950408889634f;

template <typename T> __device__ __forceinline__ float to_float(T x);
template <> __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}
template <> __device__ __forceinline__ float to_float<float>(float x) { return x; }

template <typename T> __device__ __forceinline__ T from_float(float x);
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}
template <> __device__ __forceinline__ float from_float<float>(float x) { return x; }

template <typename T> __device__ __forceinline__ float round_as(float x);
template <> __device__ __forceinline__ float round_as<__nv_bfloat16>(float x) {
  return __bfloat162float(__float2bfloat16(x));
}
template <> __device__ __forceinline__ float round_as<float>(float x) { return x; }

template <typename T>
__device__ __forceinline__ float load_decay_param(const void *ptr, int64_t idx, bool is_fp32) {
  if (is_fp32)
    return static_cast<const float *>(ptr)[idx];
  return to_float<T>(static_cast<const T *>(ptr)[idx]);
}

template <typename T>
__device__ __forceinline__ float causal_conv_silu(const T *__restrict__ qkv_b,
                                                  const T *__restrict__ conv_b, const float w[4],
                                                  int64_t ch, int64_t t, int64_t channels) {
  float total = 0.0f;
#pragma unroll
  for (int tap = 0; tap < kConvWidth; ++tap) {
    const int64_t sample = t - (kConvWidth - 1) + tap;
    const T history = sample < 0 ? conv_b[ch * kConvWidth + (kConvWidth + sample)]
                                 : qkv_b[sample * channels + ch];
    total = fmaf(to_float<T>(history), w[tap], total);
  }
  // HF conv1d accumulates in FP32, then rounds to input dtype before and after SiLU.
  const float val = round_as<T>(total);
  const float silu = val / (1.0f + expf(-val));
  return round_as<T>(silu);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  return value;
}

// ============================================================================
// Phase 1 fused recurrent kernel (optimal for decode and short sequences < 16)
// ============================================================================

// Reads conv (READ-ONLY) and updates CTA-disjoint recurrent state in place.
// Trailing update_conv_inplace_kernel commits conv history without cross-block races.
// Block layout: (32 cols, 4 warps) = 128 threads; each warp owns 32 rows of the [128, 32] tile.
template <typename T>
__global__ void deltanet_fused_kernel(const T *__restrict__ qkv, const T *__restrict__ b,
                                      const T *__restrict__ a, const T *__restrict__ conv,
                                      float *__restrict__ state, const T *__restrict__ weight,
                                      const void *__restrict__ A_log,
                                      const void *__restrict__ dt_bias, T *__restrict__ output,
                                      const int32_t *__restrict__ cu_query, int64_t tokens,
                                      int64_t key_heads, int64_t value_heads, bool a_log_fp32,
                                      bool dt_bias_fp32, int64_t max_row_len) {
  const int64_t head = blockIdx.x;
  const int64_t tile = blockIdx.y;
  const int64_t batch = blockIdx.z;

  const int col = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + col;

  const int32_t raw_len = cu_query[batch + 1] - cu_query[batch];
  const int64_t length = raw_len < 0 ? 0 : (raw_len > tokens ? tokens : raw_len);
  if (length > max_row_len)
    return;

  T *out_b = output + batch * tokens * value_heads * kHeadDim;
  const int64_t v_col = tile * kValueTile + col;

  // Inactive rows preserve both conv and recurrent state untouched and write zero output.
  if (length == 0) {
    if (warp == 0) {
      const T zero_val = from_float<T>(0.0f);
      for (int64_t t = 0; t < tokens; ++t)
        out_b[(t * value_heads + head) * kHeadDim + v_col] = zero_val;
    }
    return;
  }

  const int64_t group = value_heads / key_heads;
  const int64_t key_head = head / group;
  const int64_t channels = (2 * key_heads + value_heads) * kHeadDim;

  const T *qkv_b = qkv + batch * tokens * channels;
  const T *b_b = b + batch * tokens * value_heads;
  const T *a_b = a + batch * tokens * value_heads;
  const T *conv_b = conv + batch * channels * kConvWidth;
  float *state_head = state + (batch * value_heads + head) * kHeadDim * kHeadDim;

  const int row_base = warp * 32;
  float s[32];
#pragma unroll
  for (int i = 0; i < 32; ++i)
    s[i] = state_head[(row_base + i) * kHeadDim + v_col];

  const int64_t q_ch = key_head * kHeadDim + tid;
  const int64_t k_ch = (key_heads + key_head) * kHeadDim + tid;
  const int64_t v_ch = (2 * key_heads + head) * kHeadDim + v_col;

  float wq[kConvWidth], wk[kConvWidth], wv[kConvWidth];
#pragma unroll
  for (int tap = 0; tap < kConvWidth; ++tap) {
    wq[tap] = to_float<T>(weight[q_ch * kConvWidth + tap]);
    wk[tap] = to_float<T>(weight[k_ch * kConvWidth + tap]);
  }
  if (warp == 0) {
#pragma unroll
    for (int tap = 0; tap < kConvWidth; ++tap)
      wv[tap] = to_float<T>(weight[v_ch * kConvWidth + tap]);
  }

  float scale = 0.0f;
  float dt = 0.0f;
  if (tid == 0) {
    scale = -expf(load_decay_param<T>(A_log, head, a_log_fp32));
    dt = load_decay_param<T>(dt_bias, head, dt_bias_fp32);
  }

  __shared__ float smem_q[kHeadDim];
  __shared__ float smem_k[kHeadDim];
  __shared__ float smem_v[kValueTile];
  __shared__ float smem_q_sq[4];
  __shared__ float smem_k_sq[4];
  __shared__ float smem_pred[4][kValueTile];
  __shared__ float smem_out[4][kValueTile];
  __shared__ float smem_beta;
  __shared__ float smem_decay;

  for (int64_t t = 0; t < length; ++t) {
    const float q_val = causal_conv_silu<T>(qkv_b, conv_b, wq, q_ch, t, channels);
    const float k_val = causal_conv_silu<T>(qkv_b, conv_b, wk, k_ch, t, channels);

    const float q_wsum = warp_sum(q_val * q_val);
    const float k_wsum = warp_sum(k_val * k_val);
    if (col == 0) {
      smem_q_sq[warp] = q_wsum;
      smem_k_sq[warp] = k_wsum;
    }

    if (warp == 0)
      smem_v[col] = causal_conv_silu<T>(qkv_b, conv_b, wv, v_ch, t, channels);

    if (tid == 0) {
      const float b_raw = to_float<T>(b_b[t * value_heads + head]);
      smem_beta = round_as<T>(1.0f / (1.0f + expf(-b_raw)));
      const float x = to_float<T>(a_b[t * value_heads + head]) + dt;
      const float softplus = x > 20.0f ? x : log1pf(expf(x));
      smem_decay = expf(scale * softplus);
    }
    __syncthreads();

    const float q_inv =
        rsqrtf(smem_q_sq[0] + smem_q_sq[1] + smem_q_sq[2] + smem_q_sq[3] + kNormEps) *
        kInvSqrtHeadDim;
    const float k_inv =
        rsqrtf(smem_k_sq[0] + smem_k_sq[1] + smem_k_sq[2] + smem_k_sq[3] + kNormEps);
    smem_q[tid] = q_val * q_inv;
    smem_k[tid] = k_val * k_inv;
    __syncthreads();

    const float decay = smem_decay;
    const float beta = smem_beta;
    const float v_val = smem_v[col];

    float pred_part = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      s[i] *= decay;
      pred_part = fmaf(s[i], smem_k[row_base + i], pred_part);
    }
    smem_pred[warp][col] = pred_part;
    __syncthreads();

    const float pred =
        smem_pred[0][col] + smem_pred[1][col] + smem_pred[2][col] + smem_pred[3][col];
    const float delta = (v_val - pred) * beta;

    float out_part = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      s[i] = fmaf(smem_k[row_base + i], delta, s[i]);
      out_part = fmaf(s[i], smem_q[row_base + i], out_part);
    }
    smem_out[warp][col] = out_part;
    __syncthreads();

    if (warp == 0) {
      const float out_val =
          smem_out[0][col] + smem_out[1][col] + smem_out[2][col] + smem_out[3][col];
      out_b[(t * value_heads + head) * kHeadDim + v_col] = from_float<T>(out_val);
    }
  }

  if (warp == 0 && length < tokens) {
    const T zero_val = from_float<T>(0.0f);
    for (int64_t t = length; t < tokens; ++t)
      out_b[(t * value_heads + head) * kHeadDim + v_col] = zero_val;
  }

#pragma unroll
  for (int i = 0; i < 32; ++i)
    state_head[(row_base + i) * kHeadDim + v_col] = s[i];
}

// ============================================================================
// Phase 2 fused zero-workspace Tensor Core chunked kernels (SM >= 8.0)
// ============================================================================

// Shared exact FP32 block-recursive inversion of 32x32 unit-lower-triangular (I - A).
// On entry A_fp32[r][c] holds A[r][c] (c < r), 1.0f (r == c), 0.0f (c > r).
// tmp_inv must point to at least 256 floats (1 KB) of temporary shared memory.
__device__ __forceinline__ void invert_unit_lower_32x32(float A_fp32[kChunk][kChunk + 1],
                                                        float *tmp_inv, int tid, int warp,
                                                        int lane) {
  // Stage 1: Invert four 8x8 diagonal blocks in registers across warps 0..3.
  if (warp < 4) {
    const int base = warp * 8;
    float col[8];
    if (lane < 8) {
#pragma unroll
      for (int r = 0; r < 8; ++r)
        col[r] = A_fp32[base + r][base + lane];
    }
#pragma unroll
    for (int i = 1; i < 8; ++i) {
      float acc = 0.0f;
#pragma unroll
      for (int m = 0; m < i; ++m) {
        const float w = __shfl_sync(0xffffffffu, col[i], m);
        if (lane <= m)
          acc = fmaf(w, col[m], acc);
      }
      if (lane < i)
        col[i] = acc;
    }
    if (lane < 8) {
#pragma unroll
      for (int r = 1; r < 8; ++r)
        if (lane < r)
          A_fp32[base + r][base + lane] = col[r];
    }
  }
  __syncthreads();

  // Stage 2: Combine into two 16x16 inverses: off-diag 8x8 blocks (1,0) and (3,2).
  if (tid < 128) {
    const int pair = tid >> 6;
    const int r = (tid >> 3) & 7;
    const int c = tid & 7;
    const int rb = pair * 16 + 8;
    const int cb = pair * 16;
    float acc = 0.0f;
#pragma unroll
    for (int m = 0; m <= r; ++m)
      acc = fmaf(A_fp32[rb + r][rb + m], A_fp32[rb + m][cb + c], acc);
    tmp_inv[tid] = acc;
  }
  __syncthreads();
  if (tid < 128) {
    const int pair = tid >> 6;
    const int r = (tid >> 3) & 7;
    const int c = tid & 7;
    const int rb = pair * 16 + 8;
    const int cb = pair * 16;
    float acc = 0.0f;
#pragma unroll
    for (int m = c; m < 8; ++m)
      acc = fmaf(tmp_inv[(pair << 6) + (r << 3) + m], A_fp32[cb + m][cb + c], acc);
    A_fp32[rb + r][cb + c] = acc;
  }
  __syncthreads();

  // Stage 3: Combine into full 32x32 inverse: off-diag 16x16 block (16..31, 0..15).
  {
    const int r = tid >> 4;
    const int c = tid & 15;
    float acc = 0.0f;
    for (int m = 0; m <= r; ++m)
      acc = fmaf(A_fp32[16 + r][16 + m], A_fp32[16 + m][c], acc);
    tmp_inv[tid] = acc;
  }
  __syncthreads();
  {
    const int r = tid >> 4;
    const int c = tid & 15;
    float acc = 0.0f;
    for (int m = c; m < 16; ++m)
      acc = fmaf(tmp_inv[(r << 4) + m], A_fp32[m][c], acc);
    A_fp32[16 + r][c] = acc;
  }
  __syncthreads();
}

// --- BF16x3 Tensor Core helpers (m16n8k16, 16-bit split precision, 30.8 TFLOPS on GB10) ---

__device__ __forceinline__ float fast_rcp(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float fast_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// Branchless 2-SFU-instruction SiLU rounded to BF16 (7-bit mantissa << ~22-bit SFU accuracy).
__device__ __forceinline__ float silu_bf16(float total) {
  const float val = round_as<__nv_bfloat16>(total);
  return round_as<__nv_bfloat16>(val * fast_rcp(1.0f + fast_exp2(val * -kLog2E)));
}

__device__ __forceinline__ uint32_t pack_bf16x2_f(float a, float b) {
  __nv_bfloat162 p = __float22bfloat162_rn(make_float2(a, b));
  return *reinterpret_cast<uint32_t *>(&p);
}

__device__ __forceinline__ void split_pack_bf16x2(float a, float b, uint32_t &hi, uint32_t &lo) {
  __nv_bfloat162 h = __float22bfloat162_rn(make_float2(a, b));
  float2 hf = __bfloat1622float2(h);
  __nv_bfloat162 l = __float22bfloat162_rn(make_float2(a - hf.x, b - hf.y));
  hi = *reinterpret_cast<uint32_t *>(&h);
  lo = *reinterpret_cast<uint32_t *>(&l);
}

__device__ __forceinline__ uint32_t smem_u32addr(const void *ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

template <int kCols> __device__ __forceinline__ int swz_bf16(int r, int c);
template <> __device__ __forceinline__ int swz_bf16<128>(int r, int c) {
  return c ^ ((r & 7) << 3);
}
template <> __device__ __forceinline__ int swz_bf16<32>(int r, int c) {
  return c ^ (((r >> 1) & 3) << 3);
}

template <int kCols>
__device__ __forceinline__ uint32_t load_bf16x2_swz(const __nv_bfloat16 *base, int r, int c) {
  return *reinterpret_cast<const uint32_t *>(base + r * kCols + swz_bf16<kCols>(r, c));
}

template <int kCols>
__device__ __forceinline__ void store_bf16x2_swz(__nv_bfloat16 *base, int r, int c, uint32_t val) {
  *reinterpret_cast<uint32_t *>(base + r * kCols + swz_bf16<kCols>(r, c)) = val;
}

__device__ __forceinline__ void mma_m16n8k16_bf16(float &d0, float &d1, float &d2, float &d3,
                                                  uint32_t a0, uint32_t a1, uint32_t a2,
                                                  uint32_t a3, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
               : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
  (void)d0;
  (void)d1;
  (void)d2;
  (void)d3;
  (void)a0;
  (void)a1;
  (void)a2;
  (void)a3;
  (void)b0;
  (void)b1;
#endif
}

template <int kCols>
__device__ __forceinline__ void ldmatrix_A_row(const __nv_bfloat16 *mat, int r0, int c0, int lane,
                                               uint32_t &a0, uint32_t &a1, uint32_t &a2,
                                               uint32_t &a3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const int r = r0 + (lane & 15);
  const int c = c0 + ((lane >> 4) << 3);
  const uint32_t addr = smem_u32addr(mat + r * kCols + swz_bf16<kCols>(r, c));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
               : "r"(addr));
#else
  a0 = a1 = a2 = a3 = 0u;
#endif
}

template <int kCols>
__device__ __forceinline__ void ldmatrix_B_rowBT(const __nv_bfloat16 *mat, int n0, int k0, int lane,
                                                 uint32_t &b0, uint32_t &b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const int r = n0 + (lane & 7);
  const int c = k0 + (((lane >> 3) & 1) << 3);
  const uint32_t addr = smem_u32addr(mat + r * kCols + swz_bf16<kCols>(r, c));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(b0), "=r"(b1)
               : "r"(addr));
#else
  b0 = b1 = 0u;
#endif
}

template <int kCols>
__device__ __forceinline__ void ldmatrix_B_rowB(const __nv_bfloat16 *mat, int k0, int n0, int lane,
                                                uint32_t &b0, uint32_t &b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const int r = k0 + (lane & 15);
  const uint32_t addr = smem_u32addr(mat + r * kCols + swz_bf16<kCols>(r, n0));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(b0), "=r"(b1)
               : "r"(addr));
#else
  b0 = b1 = 0u;
#endif
}

template <int kCols>
__device__ __forceinline__ void ldmatrix_A_col(const __nv_bfloat16 *mat, int r0, int k0, int lane,
                                               uint32_t &a0, uint32_t &a1, uint32_t &a2,
                                               uint32_t &a3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const int r = k0 + ((lane >> 4) << 3) + (lane & 7);
  const int c = r0 + (((lane >> 3) & 1) << 3);
  const uint32_t addr = smem_u32addr(mat + r * kCols + swz_bf16<kCols>(r, c));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
               : "r"(addr));
#else
  a0 = a1 = a2 = a3 = 0u;
#endif
}

struct alignas(16) ChunkSmemBf16 {
  __nv_bfloat16 K_hi[kChunk * kHeadDim];  // 8 KB
  __nv_bfloat16 K_lo[kChunk * kHeadDim];  // 8 KB
  __nv_bfloat16 WQ_hi[kChunk * kHeadDim]; // 8 KB (W then Q; raw_K+raw_V stage across WQ+M32)
  __nv_bfloat16 WQ_lo[kChunk * kHeadDim]; // 8 KB (raw_Q stages starting here into M32_hi)
  __nv_bfloat16 M32_hi[kChunk * kChunk];  // 2 KB
  __nv_bfloat16 M32_lo[kChunk * kChunk];  // 2 KB
  union {
    struct {
      float A_fp32[kChunk][kChunk + 1]; // 4,224 B
      __nv_bfloat16 V_raw[kChunk * kHeadDim];
    } stage1;
    struct {
      __nv_bfloat16 hi[kChunk * kHeadDim];
      __nv_bfloat16 lo[kChunk * kHeadDim];
    } vnew;
  } aux;
  float g_cum[kChunk];
  float beta[kChunk];
  float exp_g[kChunk];
  float exp_glast_g[kChunk];
  float sq_part[kChunk][2];
  float inv_norm[kChunk];
};

__device__ __forceinline__ void stage_raw_bf16_parallel(const __nv_bfloat16 *__restrict__ qkv_b,
                                                        const __nv_bfloat16 *__restrict__ conv_b,
                                                        int64_t ch_base, int64_t start, int L,
                                                        int64_t channels, __nv_bfloat16 *raw_buf,
                                                        int tid) {
  constexpr int kVecPerRow = kHeadDim / 8; // 16 uint4s per row
  if (start >= 3) {
    const int total_vecs = (L + 3) * kVecPerRow;
    for (int idx = tid; idx < total_vecs; idx += kChunkThreads) {
      const int r = idx / kVecPerRow;
      const int c = (idx % kVecPerRow) * 8;
      const int64_t tok = start - 3 + r;
      *reinterpret_cast<uint4 *>(raw_buf + r * kHeadDim + c) =
          *reinterpret_cast<const uint4 *>(qkv_b + tok * channels + ch_base + c);
    }
  } else {
    if (tid < kHeadDim) {
      const __nv_bfloat16 *ch_conv = conv_b + (ch_base + tid) * kConvWidth;
      raw_buf[0 * kHeadDim + tid] = ch_conv[1];
      raw_buf[1 * kHeadDim + tid] = ch_conv[2];
      raw_buf[2 * kHeadDim + tid] = ch_conv[3];
    }
    const int total_vecs = L * kVecPerRow;
    for (int idx = tid; idx < total_vecs; idx += kChunkThreads) {
      const int r = idx / kVecPerRow;
      const int c = (idx % kVecPerRow) * 8;
      *reinterpret_cast<uint4 *>(raw_buf + (r + 3) * kHeadDim + c) =
          *reinterpret_cast<const uint4 *>(qkv_b + r * channels + ch_base + c);
    }
  }
}

__device__ __forceinline__ void
conv_norm_qk_bf16_simd(const __nv_bfloat16 *raw_buf, const float w0[4], const float w1[4], int L,
                       float extra_scale, __nv_bfloat16 *out_hi, __nv_bfloat16 *out_lo,
                       float sq_part[kChunk][2], float inv_norm[kChunk], int tid, int warp,
                       int lane) {
  // 256 threads = 4 row-groups of 8 rows x 64 bf16x2 column-pairs (2 warps per row).
  const int r_grp = tid >> 6;
  const int c_pair = (tid & 63) * 2;
  const int w_in_grp = warp & 1;
  const int r_beg = r_grp * 8;

  uint32_t silu_cache[8];
  auto load_f2 = [&](int row) -> float2 {
    const __nv_bfloat162 p =
        *reinterpret_cast<const __nv_bfloat162 *>(raw_buf + row * kHeadDim + c_pair);
    return __bfloat1622float2(p);
  };

  if (r_beg < L) {
    float2 h0 = load_f2(r_beg + 0);
    float2 h1 = load_f2(r_beg + 1);
    float2 h2 = load_f2(r_beg + 2);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int i = r_beg + k;
      if (i < L) {
        const float2 h3 = load_f2(i + 3);
        const float tot0 = fmaf(h0.x, w0[0], fmaf(h1.x, w0[1], fmaf(h2.x, w0[2], h3.x * w0[3])));
        const float tot1 = fmaf(h0.y, w1[0], fmaf(h1.y, w1[1], fmaf(h2.y, w1[2], h3.y * w1[3])));
        h0 = h1;
        h1 = h2;
        h2 = h3;
        const float s0 = silu_bf16(tot0);
        const float s1 = silu_bf16(tot1);
        silu_cache[k] = pack_bf16x2_f(s0, s1);
        const float ws = warp_sum(fmaf(s0, s0, s1 * s1));
        if (lane == 0)
          sq_part[i][w_in_grp] = ws;
      } else {
        silu_cache[k] = 0u;
      }
    }
  } else {
#pragma unroll
    for (int k = 0; k < 8; ++k)
      silu_cache[k] = 0u;
  }
  __syncthreads();

  if (tid < L)
    inv_norm[tid] = rsqrtf(sq_part[tid][0] + sq_part[tid][1] + kNormEps) * extra_scale;
  __syncthreads();

#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int i = r_beg + k;
    if (i < L) {
      const float inv = inv_norm[i];
      const __nv_bfloat162 p = *reinterpret_cast<const __nv_bfloat162 *>(&silu_cache[k]);
      const float2 sf = __bfloat1622float2(p);
      uint32_t hi, lo;
      split_pack_bf16x2(sf.x * inv, sf.y * inv, hi, lo);
      store_bf16x2_swz<kHeadDim>(out_hi, i, c_pair, hi);
      store_bf16x2_swz<kHeadDim>(out_lo, i, c_pair, lo);
    } else {
      store_bf16x2_swz<kHeadDim>(out_hi, i, c_pair, 0u);
      store_bf16x2_swz<kHeadDim>(out_lo, i, c_pair, 0u);
    }
  }
}

// Zero-workspace fused BF16x3 Tensor Core chunked kernel (1 CTA per (batch, head), 8 warps).
// Keeps S_T[128, 128] resident in registers across chunks (each warp owns 16 V-rows x 128 K-cols).
__global__ void deltanet_chunked_bf16_kernel(
    const __nv_bfloat16 *__restrict__ qkv, const __nv_bfloat16 *__restrict__ b,
    const __nv_bfloat16 *__restrict__ a, const __nv_bfloat16 *__restrict__ conv,
    float *__restrict__ state, const __nv_bfloat16 *__restrict__ weight,
    const void *__restrict__ A_log, const void *__restrict__ dt_bias,
    __nv_bfloat16 *__restrict__ output, const int32_t *__restrict__ cu_query, int64_t tokens,
    int64_t key_heads, int64_t value_heads, bool a_log_fp32, bool dt_bias_fp32, int64_t min_row_len,
    bool zero_inactive) {
  const int64_t head = blockIdx.x;
  const int64_t batch = blockIdx.y;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;

  const int32_t raw_len = cu_query[batch + 1] - cu_query[batch];
  const int64_t length = raw_len < 0 ? 0 : (raw_len > tokens ? tokens : raw_len);

  __nv_bfloat16 *out_b = output + batch * tokens * value_heads * kHeadDim;
  if (length == 0) {
    if (zero_inactive && tid < kHeadDim / 2) {
      for (int64_t tok = 0; tok < tokens; ++tok)
        *reinterpret_cast<uint32_t *>(out_b + (tok * value_heads + head) * kHeadDim + tid * 2) = 0u;
    }
    return;
  }
  if (length < min_row_len)
    return;

  extern __shared__ unsigned char smem_raw[];
  auto &sm = *reinterpret_cast<ChunkSmemBf16 *>(smem_raw);

  const int64_t group = value_heads / key_heads;
  const int64_t key_head = head / group;
  const int64_t channels = (2 * key_heads + value_heads) * kHeadDim;

  const __nv_bfloat16 *qkv_b = qkv + batch * tokens * channels;
  const __nv_bfloat16 *b_b = b + batch * tokens * value_heads;
  const __nv_bfloat16 *a_b = a + batch * tokens * value_heads;
  const __nv_bfloat16 *conv_b = conv + batch * channels * kConvWidth;
  float *state_head = state + (batch * value_heads + head) * kHeadDim * kHeadDim;

  const int64_t q_ch_base = key_head * kHeadDim;
  const int64_t k_ch_base = (key_heads + key_head) * kHeadDim;
  const int64_t v_ch_base = (2 * key_heads + head) * kHeadDim;

  // Hoist conv weights once per CTA for bf16x2 pairs (c_pair, c_pair+1).
  const int c_pair = (tid & 63) * 2;
  float wq0[4], wq1[4], wk0[4], wk1[4], wv0[4], wv1[4];
#pragma unroll
  for (int tap = 0; tap < kConvWidth; ++tap) {
    wq0[tap] = to_float(weight[(q_ch_base + c_pair) * kConvWidth + tap]);
    wq1[tap] = to_float(weight[(q_ch_base + c_pair + 1) * kConvWidth + tap]);
    wk0[tap] = to_float(weight[(k_ch_base + c_pair) * kConvWidth + tap]);
    wk1[tap] = to_float(weight[(k_ch_base + c_pair + 1) * kConvWidth + tap]);
    wv0[tap] = to_float(weight[(v_ch_base + c_pair) * kConvWidth + tap]);
    wv1[tap] = to_float(weight[(v_ch_base + c_pair + 1) * kConvWidth + tap]);
  }

  // Load warp's 16 V-rows of S_T[128, 128] in exact m16n8k16 D/B register layout.
  const int v_warp = warp * 16;
  const int v_g0 = v_warp + g;
  const int v_g1 = v_g0 + 8;
  float s_reg[16][4];
#pragma unroll
  for (int kt = 0; kt < 16; ++kt) {
    const int k0 = kt * 8 + 2 * t;
    s_reg[kt][0] = state_head[k0 * kHeadDim + v_g0];
    s_reg[kt][1] = state_head[(k0 + 1) * kHeadDim + v_g0];
    s_reg[kt][2] = state_head[k0 * kHeadDim + v_g1];
    s_reg[kt][3] = state_head[(k0 + 1) * kHeadDim + v_g1];
  }

  const float scale = -expf(load_decay_param<__nv_bfloat16>(A_log, head, a_log_fp32));
  const float dt = load_decay_param<__nv_bfloat16>(dt_bias, head, dt_bias_fp32);

  for (int64_t start = 0; start < length; start += kChunk) {
    const int L = static_cast<int>((length - start) < kChunk ? (length - start) : kChunk);

    // 1. Parallel 128-bit staging of raw_K and raw_V across dead WQ+M32 (20 KB) + warp 0 g/beta.
    __nv_bfloat16 *raw_K = sm.WQ_hi;
    __nv_bfloat16 *raw_V = raw_K + kRawRows * kHeadDim;
    stage_raw_bf16_parallel(qkv_b, conv_b, k_ch_base, start, L, channels, raw_K, tid);
    stage_raw_bf16_parallel(qkv_b, conv_b, v_ch_base, start, L, channels, raw_V, tid);

    if (warp == 0) {
      float g_step = 0.0f;
      float beta_val = 0.0f;
      if (lane < L) {
        const int64_t tok = start + lane;
        const float b_raw = to_float(b_b[tok * value_heads + head]);
        beta_val = round_as<__nv_bfloat16>(fast_rcp(1.0f + fast_exp2(b_raw * -kLog2E)));
        const float x = to_float(a_b[tok * value_heads + head]) + dt;
        const float sp = x > 20.0f ? x : log1pf(expf(x));
        g_step = scale * sp;
      }
      float gc = g_step;
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        const float prev = __shfl_up_sync(0xffffffffu, gc, off);
        if (lane >= off)
          gc += prev;
      }
      const float g_last = __shfl_sync(0xffffffffu, gc, L - 1);
      sm.g_cum[lane] = lane < L ? gc : g_last;
      sm.beta[lane] = beta_val;
      sm.exp_g[lane] = lane < L ? expf(gc) : 0.0f;
      sm.exp_glast_g[lane] = lane < L ? expf(g_last - gc) : 0.0f;
    }
    __syncthreads();

    // 2. Conv + SiLU + L2 norm for K, and 4-way row-split Conv + SiLU for V.
    conv_norm_qk_bf16_simd(raw_K, wk0, wk1, L, 1.0f, sm.K_hi, sm.K_lo, sm.sq_part, sm.inv_norm, tid,
                           warp, lane);

    {
      const int r_beg = (tid >> 6) * 8;
      auto load_vf2 = [&](int row) -> float2 {
        const __nv_bfloat162 p =
            *reinterpret_cast<const __nv_bfloat162 *>(raw_V + row * kHeadDim + c_pair);
        return __bfloat1622float2(p);
      };
      if (r_beg < L) {
        float2 h0 = load_vf2(r_beg + 0);
        float2 h1 = load_vf2(r_beg + 1);
        float2 h2 = load_vf2(r_beg + 2);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int i = r_beg + k;
          if (i < L) {
            const float2 h3 = load_vf2(i + 3);
            const float t0 =
                fmaf(h0.x, wv0[0], fmaf(h1.x, wv0[1], fmaf(h2.x, wv0[2], h3.x * wv0[3])));
            const float t1 =
                fmaf(h0.y, wv1[0], fmaf(h1.y, wv1[1], fmaf(h2.y, wv1[2], h3.y * wv1[3])));
            h0 = h1;
            h1 = h2;
            h2 = h3;
            store_bf16x2_swz<kHeadDim>(sm.aux.stage1.V_raw, i, c_pair,
                                       pack_bf16x2_f(silu_bf16(t0), silu_bf16(t1)));
          } else {
            store_bf16x2_swz<kHeadDim>(sm.aux.stage1.V_raw, i, c_pair, 0u);
          }
        }
      } else {
#pragma unroll
        for (int k = 0; k < 8; ++k)
          store_bf16x2_swz<kHeadDim>(sm.aux.stage1.V_raw, r_beg + k, c_pair, 0u);
      }
    }
    __syncthreads();

    // 3. GEMM 1: KK[32, 32] = K @ K^T -> strictly lower A in A_fp32 (1 tile per warp).
    {
      const int mt = warp >> 2;
      const int nt = warp & 3;
      const int r0 = mt * 16;
      const int c0 = nt * 8;
      float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
      if (!(mt == 0 && nt >= 2)) {
        for (int k = 0; k < kHeadDim; k += 16) {
          uint32_t a0_h, a1_h, a2_h, a3_h, a0_l, a1_l, a2_l, a3_l;
          uint32_t b0_h, b1_h, b0_l, b1_l;
          ldmatrix_A_row<kHeadDim>(sm.K_hi, r0, k, lane, a0_h, a1_h, a2_h, a3_h);
          ldmatrix_A_row<kHeadDim>(sm.K_lo, r0, k, lane, a0_l, a1_l, a2_l, a3_l);
          ldmatrix_B_rowBT<kHeadDim>(sm.K_hi, c0, k, lane, b0_h, b1_h);
          ldmatrix_B_rowBT<kHeadDim>(sm.K_lo, c0, k, lane, b0_l, b1_l);
          mma_m16n8k16_bf16(d0, d1, d2, d3, a0_h, a1_h, a2_h, a3_h, b0_h, b1_h);
          mma_m16n8k16_bf16(d0, d1, d2, d3, a0_h, a1_h, a2_h, a3_h, b0_l, b1_l);
          mma_m16n8k16_bf16(d0, d1, d2, d3, a0_l, a1_l, a2_l, a3_l, b0_h, b1_h);
        }
      }
      auto store_A = [&](int r, int c, float val) {
        if (c < r && r < L)
          sm.aux.stage1.A_fp32[r][c] =
              -sm.beta[r] * fast_exp2((sm.g_cum[r] - sm.g_cum[c]) * kLog2E) * val;
        else
          sm.aux.stage1.A_fp32[r][c] = (r == c) ? 1.0f : 0.0f;
      };
      store_A(r0 + g, c0 + 2 * t, d0);
      store_A(r0 + g, c0 + 2 * t + 1, d1);
      store_A(r0 + g + 8, c0 + 2 * t, d2);
      store_A(r0 + g + 8, c0 + 2 * t + 1, d3);
    }
    __syncthreads();

    // 4. Exact FP32 block-recursive inversion: A_inv = (I - A)^{-1}.
    invert_unit_lower_32x32(sm.aux.stage1.A_fp32, reinterpret_cast<float *>(sm.M32_hi), tid, warp,
                            lane);

    // 5. Build A_inv_bg[32, 32] = A_inv * diag_col(beta * exp(g)) into M32_hi/lo.
    for (int idx = tid; idx < 512; idx += kChunkThreads) {
      const int r = idx >> 4;
      const int c = (idx & 15) * 2;
      const float v0 =
          (c <= r && r < L) ? (sm.aux.stage1.A_fp32[r][c] * sm.beta[c] * sm.exp_g[c]) : 0.0f;
      const float v1 = (c + 1 <= r && r < L)
                           ? (sm.aux.stage1.A_fp32[r][c + 1] * sm.beta[c + 1] * sm.exp_g[c + 1])
                           : 0.0f;
      uint32_t hi, lo;
      split_pack_bf16x2(v0, v1, hi, lo);
      store_bf16x2_swz<kChunk>(sm.M32_hi, r, c, hi);
      store_bf16x2_swz<kChunk>(sm.M32_lo, r, c, lo);
    }
    __syncthreads();

    // 6. GEMM 2: W[32, 128] = A_inv_bg @ K -> WQ_hi/lo.
    for (int tile_idx = warp; tile_idx < 32; tile_idx += kChunkWarps) {
      const int mt = tile_idx >> 4;
      const int nt = tile_idx & 15;
      const int r0 = mt * 16;
      const int c0 = nt * 8;
      float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
      const int k_limit = (mt == 0) ? 16 : 32;
      for (int k = 0; k < k_limit; k += 16) {
        uint32_t a0_h, a1_h, a2_h, a3_h, a0_l, a1_l, a2_l, a3_l;
        uint32_t b0_h, b1_h, b0_l, b1_l;
        ldmatrix_A_row<kChunk>(sm.M32_hi, r0, k, lane, a0_h, a1_h, a2_h, a3_h);
        ldmatrix_A_row<kChunk>(sm.M32_lo, r0, k, lane, a0_l, a1_l, a2_l, a3_l);
        ldmatrix_B_rowB<kHeadDim>(sm.K_hi, k, c0, lane, b0_h, b1_h);
        ldmatrix_B_rowB<kHeadDim>(sm.K_lo, k, c0, lane, b0_l, b1_l);
        mma_m16n8k16_bf16(d0, d1, d2, d3, a0_h, a1_h, a2_h, a3_h, b0_h, b1_h);
        mma_m16n8k16_bf16(d0, d1, d2, d3, a0_h, a1_h, a2_h, a3_h, b0_l, b1_l);
        mma_m16n8k16_bf16(d0, d1, d2, d3, a0_l, a1_l, a2_l, a3_l, b0_h, b1_h);
      }
      uint32_t h0, l0, h1, l1;
      split_pack_bf16x2(d0, d1, h0, l0);
      split_pack_bf16x2(d2, d3, h1, l1);
      store_bf16x2_swz<kHeadDim>(sm.WQ_hi, r0 + g, c0 + 2 * t, h0);
      store_bf16x2_swz<kHeadDim>(sm.WQ_lo, r0 + g, c0 + 2 * t, l0);
      store_bf16x2_swz<kHeadDim>(sm.WQ_hi, r0 + g + 8, c0 + 2 * t, h1);
      store_bf16x2_swz<kHeadDim>(sm.WQ_lo, r0 + g + 8, c0 + 2 * t, l1);
    }
    __syncthreads();

    // 7. Build A_inv_b[32, 32] = A_inv * diag_col(beta) into M32_hi/lo.
    for (int idx = tid; idx < 512; idx += kChunkThreads) {
      const int r = idx >> 4;
      const int c = (idx & 15) * 2;
      const float v0 = (c <= r && r < L) ? (sm.aux.stage1.A_fp32[r][c] * sm.beta[c]) : 0.0f;
      const float v1 =
          (c + 1 <= r && r < L) ? (sm.aux.stage1.A_fp32[r][c + 1] * sm.beta[c + 1]) : 0.0f;
      uint32_t hi, lo;
      split_pack_bf16x2(v0, v1, hi, lo);
      store_bf16x2_swz<kChunk>(sm.M32_hi, r, c, hi);
      store_bf16x2_swz<kChunk>(sm.M32_lo, r, c, lo);
    }
    __syncthreads();

    // 8. Fused U - W @ S -> v_new in registers (negated accumulator: -U + W@S, then negated).
    float vn[2][2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      const int r0 = mt * 16;
#pragma unroll
      for (int ns = 0; ns < 2; ++ns) {
        const int c0 = v_warp + ns * 8;
        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        const int k_limit = (mt == 0) ? 16 : 32;
        for (int k = 0; k < k_limit; k += 16) {
          uint32_t a0_h, a1_h, a2_h, a3_h, a0_l, a1_l, a2_l, a3_l;
          uint32_t b0, b1;
          ldmatrix_A_row<kChunk>(sm.M32_hi, r0, k, lane, a0_h, a1_h, a2_h, a3_h);
          ldmatrix_A_row<kChunk>(sm.M32_lo, r0, k, lane, a0_l, a1_l, a2_l, a3_l);
          ldmatrix_B_rowB<kHeadDim>(sm.aux.stage1.V_raw, k, c0, lane, b0, b1);
          mma_m16n8k16_bf16(d0, d1, d2, d3, a0_h, a1_h, a2_h, a3_h, b0, b1);
          mma_m16n8k16_bf16(d0, d1, d2, d3, a0_l, a1_l, a2_l, a3_l, b0, b1);
        }
        vn[mt][ns][0] = -d0;
        vn[mt][ns][1] = -d1;
        vn[mt][ns][2] = -d2;
        vn[mt][ns][3] = -d3;
      }
    }
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) {
      const int k0 = kb * 16;
      uint32_t b0_h0, b0_l0, b1_h0, b1_l0;
      uint32_t b0_h1, b0_l1, b1_h1, b1_l1;
      split_pack_bf16x2(s_reg[2 * kb][0], s_reg[2 * kb][1], b0_h0, b0_l0);
      split_pack_bf16x2(s_reg[2 * kb + 1][0], s_reg[2 * kb + 1][1], b1_h0, b1_l0);
      split_pack_bf16x2(s_reg[2 * kb][2], s_reg[2 * kb][3], b0_h1, b0_l1);
      split_pack_bf16x2(s_reg[2 * kb + 1][2], s_reg[2 * kb + 1][3], b1_h1, b1_l1);
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        uint32_t a0_h, a1_h, a2_h, a3_h, a0_l, a1_l, a2_l, a3_l;
        ldmatrix_A_row<kHeadDim>(sm.WQ_hi, mt * 16, k0, lane, a0_h, a1_h, a2_h, a3_h);
        ldmatrix_A_row<kHeadDim>(sm.WQ_lo, mt * 16, k0, lane, a0_l, a1_l, a2_l, a3_l);
        mma_m16n8k16_bf16(vn[mt][0][0], vn[mt][0][1], vn[mt][0][2], vn[mt][0][3], a0_h, a1_h, a2_h,
                          a3_h, b0_h0, b1_h0);
        mma_m16n8k16_bf16(vn[mt][0][0], vn[mt][0][1], vn[mt][0][2], vn[mt][0][3], a0_h, a1_h, a2_h,
                          a3_h, b0_l0, b1_l0);
        mma_m16n8k16_bf16(vn[mt][0][0], vn[mt][0][1], vn[mt][0][2], vn[mt][0][3], a0_l, a1_l, a2_l,
                          a3_l, b0_h0, b1_h0);
        mma_m16n8k16_bf16(vn[mt][1][0], vn[mt][1][1], vn[mt][1][2], vn[mt][1][3], a0_h, a1_h, a2_h,
                          a3_h, b0_h1, b1_h1);
        mma_m16n8k16_bf16(vn[mt][1][0], vn[mt][1][1], vn[mt][1][2], vn[mt][1][3], a0_h, a1_h, a2_h,
                          a3_h, b0_l1, b1_l1);
        mma_m16n8k16_bf16(vn[mt][1][0], vn[mt][1][1], vn[mt][1][2], vn[mt][1][3], a0_l, a1_l, a2_l,
                          a3_l, b0_h1, b1_h1);
      }
    }
    __syncthreads();

    // Store v_new (aliases dead stage1) and stage raw_Q in parallel into dead WQ_lo+M32_hi.
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      const int r0 = mt * 16;
      const int r_hi = r0 + g;
      const int r_lo = r0 + g + 8;
#pragma unroll
      for (int ns = 0; ns < 2; ++ns) {
        const int c = v_warp + ns * 8 + 2 * t;
        uint32_t h0 = 0u, l0 = 0u, h1 = 0u, l1 = 0u;
        if (r_hi < L)
          split_pack_bf16x2(-vn[mt][ns][0], -vn[mt][ns][1], h0, l0);
        if (r_lo < L)
          split_pack_bf16x2(-vn[mt][ns][2], -vn[mt][ns][3], h1, l1);
        store_bf16x2_swz<kHeadDim>(sm.aux.vnew.hi, r_hi, c, h0);
        store_bf16x2_swz<kHeadDim>(sm.aux.vnew.lo, r_hi, c, l0);
        store_bf16x2_swz<kHeadDim>(sm.aux.vnew.hi, r_lo, c, h1);
        store_bf16x2_swz<kHeadDim>(sm.aux.vnew.lo, r_lo, c, l1);
      }
    }
    __nv_bfloat16 *raw_Q = sm.WQ_lo;
    stage_raw_bf16_parallel(qkv_b, conv_b, q_ch_base, start, L, channels, raw_Q, tid);
    __syncthreads();

    // 9. Conv + SiLU + L2 norm for Q -> WQ_hi/WQ_lo.
    conv_norm_qk_bf16_simd(raw_Q, wq0, wq1, L, kInvSqrtHeadDim, sm.WQ_hi, sm.WQ_lo, sm.sq_part,
                           sm.inv_norm, tid, warp, lane);
    __syncthreads();

    // 10. GEMM 5: G_qk[32, 32] = tril(Q @ K^T) * exp(g_i - g_j) -> M32_hi/lo.
    {
      const int mt = warp >> 2;
      const int nt = warp & 3;
      const int r0 = mt * 16;
      const int c0 = nt * 8;
      float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
      if (!(mt == 0 && nt >= 2)) {
        for (int k = 0; k < kHeadDim; k += 16) {
          uint32_t a0_h, a1_h, a2_h, a3_h, a0_l, a1_l, a2_l, a3_l;
          uint32_t b0_h, b1_h, b0_l, b1_l;
          ldmatrix_A_row<kHeadDim>(sm.WQ_hi, r0, k, lane, a0_h, a1_h, a2_h, a3_h);
          ldmatrix_A_row<kHeadDim>(sm.WQ_lo, r0, k, lane, a0_l, a1_l, a2_l, a3_l);
          ldmatrix_B_rowBT<kHeadDim>(sm.K_hi, c0, k, lane, b0_h, b1_h);
          ldmatrix_B_rowBT<kHeadDim>(sm.K_lo, c0, k, lane, b0_l, b1_l);
          mma_m16n8k16_bf16(d0, d1, d2, d3, a0_h, a1_h, a2_h, a3_h, b0_h, b1_h);
          mma_m16n8k16_bf16(d0, d1, d2, d3, a0_h, a1_h, a2_h, a3_h, b0_l, b1_l);
          mma_m16n8k16_bf16(d0, d1, d2, d3, a0_l, a1_l, a2_l, a3_l, b0_h, b1_h);
        }
      }
      const int r_hi = r0 + g;
      const int r_lo = r0 + g + 8;
      const int c = c0 + 2 * t;
      const float g0 = (c <= r_hi && r_hi < L)
                           ? (d0 * fast_exp2((sm.g_cum[r_hi] - sm.g_cum[c]) * kLog2E))
                           : 0.0f;
      const float g1 = (c + 1 <= r_hi && r_hi < L)
                           ? (d1 * fast_exp2((sm.g_cum[r_hi] - sm.g_cum[c + 1]) * kLog2E))
                           : 0.0f;
      const float g2 = (c <= r_lo && r_lo < L)
                           ? (d2 * fast_exp2((sm.g_cum[r_lo] - sm.g_cum[c]) * kLog2E))
                           : 0.0f;
      const float g3 = (c + 1 <= r_lo && r_lo < L)
                           ? (d3 * fast_exp2((sm.g_cum[r_lo] - sm.g_cum[c + 1]) * kLog2E))
                           : 0.0f;
      uint32_t h0, l0, h1, l1;
      split_pack_bf16x2(g0, g1, h0, l0);
      split_pack_bf16x2(g2, g3, h1, l1);
      store_bf16x2_swz<kChunk>(sm.M32_hi, r_hi, c, h0);
      store_bf16x2_swz<kChunk>(sm.M32_lo, r_hi, c, l0);
      store_bf16x2_swz<kChunk>(sm.M32_hi, r_lo, c, h1);
      store_bf16x2_swz<kChunk>(sm.M32_lo, r_lo, c, l1);
    }
    __syncthreads();

    // 11. Fused O = diag(exp(g)) @ (Q @ S) + G_qk @ v_new -> output.
    float out_reg[2][2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int ns = 0; ns < 2; ++ns)
#pragma unroll
        for (int i = 0; i < 4; ++i)
          out_reg[mt][ns][i] = 0.0f;

#pragma unroll
    for (int kb = 0; kb < 8; ++kb) {
      const int k0 = kb * 16;
      uint32_t b0_h0, b0_l0, b1_h0, b1_l0;
      uint32_t b0_h1, b0_l1, b1_h1, b1_l1;
      split_pack_bf16x2(s_reg[2 * kb][0], s_reg[2 * kb][1], b0_h0, b0_l0);
      split_pack_bf16x2(s_reg[2 * kb + 1][0], s_reg[2 * kb + 1][1], b1_h0, b1_l0);
      split_pack_bf16x2(s_reg[2 * kb][2], s_reg[2 * kb][3], b0_h1, b0_l1);
      split_pack_bf16x2(s_reg[2 * kb + 1][2], s_reg[2 * kb + 1][3], b1_h1, b1_l1);
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        uint32_t a0_h, a1_h, a2_h, a3_h, a0_l, a1_l, a2_l, a3_l;
        ldmatrix_A_row<kHeadDim>(sm.WQ_hi, mt * 16, k0, lane, a0_h, a1_h, a2_h, a3_h);
        ldmatrix_A_row<kHeadDim>(sm.WQ_lo, mt * 16, k0, lane, a0_l, a1_l, a2_l, a3_l);
        mma_m16n8k16_bf16(out_reg[mt][0][0], out_reg[mt][0][1], out_reg[mt][0][2],
                          out_reg[mt][0][3], a0_h, a1_h, a2_h, a3_h, b0_h0, b1_h0);
        mma_m16n8k16_bf16(out_reg[mt][0][0], out_reg[mt][0][1], out_reg[mt][0][2],
                          out_reg[mt][0][3], a0_h, a1_h, a2_h, a3_h, b0_l0, b1_l0);
        mma_m16n8k16_bf16(out_reg[mt][0][0], out_reg[mt][0][1], out_reg[mt][0][2],
                          out_reg[mt][0][3], a0_l, a1_l, a2_l, a3_l, b0_h0, b1_h0);
        mma_m16n8k16_bf16(out_reg[mt][1][0], out_reg[mt][1][1], out_reg[mt][1][2],
                          out_reg[mt][1][3], a0_h, a1_h, a2_h, a3_h, b0_h1, b1_h1);
        mma_m16n8k16_bf16(out_reg[mt][1][0], out_reg[mt][1][1], out_reg[mt][1][2],
                          out_reg[mt][1][3], a0_h, a1_h, a2_h, a3_h, b0_l1, b1_l1);
        mma_m16n8k16_bf16(out_reg[mt][1][0], out_reg[mt][1][1], out_reg[mt][1][2],
                          out_reg[mt][1][3], a0_l, a1_l, a2_l, a3_l, b0_h1, b1_h1);
      }
    }

#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      const int r0 = mt * 16;
      const float eg0 = sm.exp_g[r0 + g];
      const float eg1 = sm.exp_g[r0 + g + 8];
#pragma unroll
      for (int ns = 0; ns < 2; ++ns) {
        out_reg[mt][ns][0] *= eg0;
        out_reg[mt][ns][1] *= eg0;
        out_reg[mt][ns][2] *= eg1;
        out_reg[mt][ns][3] *= eg1;
        const int c0 = v_warp + ns * 8;
        const int k_limit = (mt == 0) ? 16 : 32;
        for (int k = 0; k < k_limit; k += 16) {
          uint32_t a0_h, a1_h, a2_h, a3_h, a0_l, a1_l, a2_l, a3_l;
          uint32_t b0_h, b1_h, b0_l, b1_l;
          ldmatrix_A_row<kChunk>(sm.M32_hi, r0, k, lane, a0_h, a1_h, a2_h, a3_h);
          ldmatrix_A_row<kChunk>(sm.M32_lo, r0, k, lane, a0_l, a1_l, a2_l, a3_l);
          ldmatrix_B_rowB<kHeadDim>(sm.aux.vnew.hi, k, c0, lane, b0_h, b1_h);
          ldmatrix_B_rowB<kHeadDim>(sm.aux.vnew.lo, k, c0, lane, b0_l, b1_l);
          mma_m16n8k16_bf16(out_reg[mt][ns][0], out_reg[mt][ns][1], out_reg[mt][ns][2],
                            out_reg[mt][ns][3], a0_h, a1_h, a2_h, a3_h, b0_h, b1_h);
          mma_m16n8k16_bf16(out_reg[mt][ns][0], out_reg[mt][ns][1], out_reg[mt][ns][2],
                            out_reg[mt][ns][3], a0_h, a1_h, a2_h, a3_h, b0_l, b1_l);
          mma_m16n8k16_bf16(out_reg[mt][ns][0], out_reg[mt][ns][1], out_reg[mt][ns][2],
                            out_reg[mt][ns][3], a0_l, a1_l, a2_l, a3_l, b0_h, b1_h);
        }
        const int r_hi = r0 + g;
        const int r_lo = r0 + g + 8;
        const int vc0 = c0 + 2 * t;
        if (r_hi < L) {
          __nv_bfloat16 *dst = out_b + ((start + r_hi) * value_heads + head) * kHeadDim + vc0;
          *reinterpret_cast<uint32_t *>(dst) =
              pack_bf16x2_f(out_reg[mt][ns][0], out_reg[mt][ns][1]);
        }
        if (r_lo < L) {
          __nv_bfloat16 *dst = out_b + ((start + r_lo) * value_heads + head) * kHeadDim + vc0;
          *reinterpret_cast<uint32_t *>(dst) =
              pack_bf16x2_f(out_reg[mt][ns][2], out_reg[mt][ns][3]);
        }
      }
    }

    // 12. Update S_T_warp = exp(g_last) * S_T_warp + v_new_decay^T @ K directly in registers.
    for (int idx = lane; idx < kChunk * 8; idx += 32) {
      const int r = idx >> 3;
      const int c = v_warp + ((idx & 7) * 2);
      if (r < L) {
        const uint32_t uh = load_bf16x2_swz<kHeadDim>(sm.aux.vnew.hi, r, c);
        const uint32_t ul = load_bf16x2_swz<kHeadDim>(sm.aux.vnew.lo, r, c);
        const float2 fh = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&uh));
        const float2 fl = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&ul));
        const float sc = sm.exp_glast_g[r];
        uint32_t nh, nl;
        split_pack_bf16x2((fh.x + fl.x) * sc, (fh.y + fl.y) * sc, nh, nl);
        store_bf16x2_swz<kHeadDim>(sm.aux.vnew.hi, r, c, nh);
        store_bf16x2_swz<kHeadDim>(sm.aux.vnew.lo, r, c, nl);
      }
    }
    __syncwarp();

    const float exp_last = sm.exp_g[L - 1];
#pragma unroll
    for (int kt = 0; kt < 16; ++kt) {
      s_reg[kt][0] *= exp_last;
      s_reg[kt][1] *= exp_last;
      s_reg[kt][2] *= exp_last;
      s_reg[kt][3] *= exp_last;
    }

    for (int kc = 0; kc < kChunk; kc += 16) {
      if (kc >= L)
        break;
      uint32_t a0_h, a1_h, a2_h, a3_h, a0_l, a1_l, a2_l, a3_l;
      ldmatrix_A_col<kHeadDim>(sm.aux.vnew.hi, v_warp, kc, lane, a0_h, a1_h, a2_h, a3_h);
      ldmatrix_A_col<kHeadDim>(sm.aux.vnew.lo, v_warp, kc, lane, a0_l, a1_l, a2_l, a3_l);
#pragma unroll
      for (int kt = 0; kt < 16; ++kt) {
        uint32_t b0_h, b1_h, b0_l, b1_l;
        ldmatrix_B_rowB<kHeadDim>(sm.K_hi, kc, kt * 8, lane, b0_h, b1_h);
        ldmatrix_B_rowB<kHeadDim>(sm.K_lo, kc, kt * 8, lane, b0_l, b1_l);
        mma_m16n8k16_bf16(s_reg[kt][0], s_reg[kt][1], s_reg[kt][2], s_reg[kt][3], a0_h, a1_h, a2_h,
                          a3_h, b0_h, b1_h);
        mma_m16n8k16_bf16(s_reg[kt][0], s_reg[kt][1], s_reg[kt][2], s_reg[kt][3], a0_h, a1_h, a2_h,
                          a3_h, b0_l, b1_l);
        mma_m16n8k16_bf16(s_reg[kt][0], s_reg[kt][1], s_reg[kt][2], s_reg[kt][3], a0_l, a1_l, a2_l,
                          a3_l, b0_h, b1_h);
      }
    }
    __syncthreads();
  }

  if (length < tokens && tid < kHeadDim / 2) {
    for (int64_t tok = length; tok < tokens; ++tok)
      *reinterpret_cast<uint32_t *>(out_b + (tok * value_heads + head) * kHeadDim + tid * 2) = 0u;
  }

#pragma unroll
  for (int kt = 0; kt < 16; ++kt) {
    const int k0 = kt * 8 + 2 * t;
    state_head[k0 * kHeadDim + v_g0] = s_reg[kt][0];
    state_head[(k0 + 1) * kHeadDim + v_g0] = s_reg[kt][1];
    state_head[k0 * kHeadDim + v_g1] = s_reg[kt][2];
    state_head[(k0 + 1) * kHeadDim + v_g1] = s_reg[kt][3];
  }
}

// ============================================================================
// Trailing race-free in-place conv state update
// ============================================================================

template <typename T>
__global__ void update_conv_inplace_kernel(const T *__restrict__ qkv, T *__restrict__ conv,
                                           const int32_t *__restrict__ cu_query, int64_t tokens,
                                           int64_t channels) {
  const int64_t batch = blockIdx.y;
  const int64_t ch = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (ch >= channels)
    return;

  const int32_t raw_len = cu_query[batch + 1] - cu_query[batch];
  const int64_t length = raw_len < 0 ? 0 : (raw_len > tokens ? tokens : raw_len);
  if (length == 0)
    return;

  const T *qkv_b = qkv + batch * tokens * channels;
  T *conv_ch = conv + (batch * channels + ch) * kConvWidth;

  T old_taps[kConvWidth];
  if (length < kConvWidth) {
#pragma unroll
    for (int tap = 0; tap < kConvWidth; ++tap)
      old_taps[tap] = conv_ch[tap];
  }

#pragma unroll
  for (int tap = 0; tap < kConvWidth; ++tap) {
    const int64_t sample = length - kConvWidth + tap;
    conv_ch[tap] = sample < 0 ? old_taps[kConvWidth + sample] : qkv_b[sample * channels + ch];
  }
}

template <typename T>
void launch_update_conv(const T *qkv, T *conv, const int32_t *cuda_cu_query, int64_t batch,
                        int64_t tokens, int64_t channels, cudaStream_t raw_stream) {
  constexpr int kConvThreads = 256;
  const dim3 conv_grid(static_cast<unsigned int>((channels + kConvThreads - 1) / kConvThreads),
                       static_cast<unsigned int>(batch));
  update_conv_inplace_kernel<T>
      <<<conv_grid, kConvThreads, 0, raw_stream>>>(qkv, conv, cuda_cu_query, tokens, channels);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename KernelFn> void ensure_optin_smem(KernelFn kernel, size_t smem_bytes) {
  if (smem_bytes > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(smem_bytes)));
  }
}

} // namespace

void launch_deltanet_bf16(const void *qkv, const void *b, const void *a, void *conv, float *state,
                          const void *weight, const void *A_log, const void *dt_bias, void *output,
                          const int32_t *cuda_cu_query, int64_t batch, int64_t tokens,
                          int64_t key_heads, int64_t value_heads, bool a_log_fp32,
                          bool dt_bias_fp32, bool has_short, bool has_long, bool use_chunked,
                          c10::cuda::CUDAStream stream) {
  const cudaStream_t raw_stream = stream.stream();
  const int64_t channels = (2 * key_heads + value_heads) * kHeadDim;
  const auto *qkv_t = static_cast<const __nv_bfloat16 *>(qkv);
  const auto *b_t = static_cast<const __nv_bfloat16 *>(b);
  const auto *a_t = static_cast<const __nv_bfloat16 *>(a);
  auto *conv_t = static_cast<__nv_bfloat16 *>(conv);
  const auto *weight_t = static_cast<const __nv_bfloat16 *>(weight);
  auto *output_t = static_cast<__nv_bfloat16 *>(output);

  const bool run_fused = !use_chunked || has_short;
  const bool run_chunk = use_chunked && has_long;

  if (run_fused) {
    const int64_t max_row_len = run_chunk ? (kChunkThreshold - 1) : tokens;
    const dim3 block(kValueTile, 4);
    const dim3 grid(static_cast<unsigned int>(value_heads),
                    static_cast<unsigned int>(kHeadDim / kValueTile),
                    static_cast<unsigned int>(batch));
    deltanet_fused_kernel<__nv_bfloat16><<<grid, block, 0, raw_stream>>>(
        qkv_t, b_t, a_t, conv_t, state, weight_t, A_log, dt_bias, output_t, cuda_cu_query, tokens,
        key_heads, value_heads, a_log_fp32, dt_bias_fp32, max_row_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  if (run_chunk) {
    const size_t smem_bytes = sizeof(ChunkSmemBf16);
    ensure_optin_smem(deltanet_chunked_bf16_kernel, smem_bytes);
    const dim3 grid(static_cast<unsigned int>(value_heads), static_cast<unsigned int>(batch));
    const dim3 block(kChunkThreads);
    deltanet_chunked_bf16_kernel<<<grid, block, smem_bytes, raw_stream>>>(
        qkv_t, b_t, a_t, conv_t, state, weight_t, A_log, dt_bias, output_t, cuda_cu_query, tokens,
        key_heads, value_heads, a_log_fp32, dt_bias_fp32, kChunkThreshold, !run_fused);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  launch_update_conv<__nv_bfloat16>(qkv_t, conv_t, cuda_cu_query, batch, tokens, channels,
                                    raw_stream);
}

void launch_deltanet_fp32(const float *qkv, const float *b, const float *a, float *conv,
                          float *state, const float *weight, const float *A_log,
                          const float *dt_bias, float *output, const int32_t *cuda_cu_query,
                          int64_t batch, int64_t tokens, int64_t key_heads, int64_t value_heads,
                          c10::cuda::CUDAStream stream) {
  const cudaStream_t raw_stream = stream.stream();
  const int64_t channels = (2 * key_heads + value_heads) * kHeadDim;

  const dim3 block(kValueTile, 4);
  const dim3 grid(static_cast<unsigned int>(value_heads),
                  static_cast<unsigned int>(kHeadDim / kValueTile),
                  static_cast<unsigned int>(batch));
  deltanet_fused_kernel<float><<<grid, block, 0, raw_stream>>>(
      qkv, b, a, conv, state, weight, A_log, dt_bias, output, cuda_cu_query, tokens, key_heads,
      value_heads, true, true, tokens);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  launch_update_conv<float>(qkv, conv, cuda_cu_query, batch, tokens, channels, raw_stream);
}

} // namespace xinfer::detail
