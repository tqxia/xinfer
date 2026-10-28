// Correctness-first DeltaNet chunk rule using ordinary ATen CUDA operations.
// Keep shape-dependent padding and the chunk scan behind one dispatcher call.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include <tuple>

namespace {
using at::Tensor;

std::tuple<Tensor, Tensor> delta_prefill(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
    const Tensor& beta, const Tensor& initial_state) {
  TORCH_CHECK(q.dim() == 4 && k.sizes() == q.sizes() && v.dim() == 4,
              "Expected Q/K [B,T,H,K] and V [B,T,H,V]");
  const auto b = q.size(0), t = q.size(1), h = q.size(2), dk = q.size(3), dv = v.size(3);
  TORCH_CHECK(b > 0 && t > 0 && h > 0 && dk > 0 && dv > 0 &&
              v.size(0) == b && v.size(1) == t && v.size(2) == h,
              "Invalid Q/K/V dimensions");
  TORCH_CHECK(g.sizes() == at::IntArrayRef({b, t, h}) && beta.sizes() == g.sizes() &&
              initial_state.sizes() == at::IntArrayRef({b, h, dk, dv}), "Invalid state/decay/beta shape");
  TORCH_CHECK(q.is_cuda() && (q.scalar_type() == at::kBFloat16 || q.scalar_type() == at::kFloat),
              "Expected CUDA BF16 or FP32 inputs");
  for (const auto* x : {&k, &v, &beta})
    TORCH_CHECK(x->scalar_type() == q.scalar_type(), "Q/K/V/beta dtypes must match");
  for (const auto* x : {&k, &v, &g, &beta, &initial_state})
    TORCH_CHECK(x->device() == q.device(), "All inputs must use the same device");
  TORCH_CHECK(g.scalar_type() == at::kFloat && initial_state.scalar_type() == at::kFloat,
              "Decay and recurrent state must be FP32");

  auto fp32_heads = [](const Tensor& x) { return x.transpose(1, 2).to(at::kFloat).contiguous(); };
  auto normalize = [](const Tensor& x) { return x * at::rsqrt((x * x).sum(-1, true) + 1e-6); };
  auto query = normalize(fp32_heads(q)) * std::pow(double(dk), -0.5);
  auto key = normalize(fp32_heads(k));
  auto value = fp32_heads(v), decay = fp32_heads(g), rate = fp32_heads(beta);
  constexpr int64_t chunk = 64;
  const auto pad = (chunk - t % chunk) % chunk, chunks = (t + pad) / chunk;
  query = at::constant_pad_nd(query, {0, 0, 0, pad}, 0);
  key = at::constant_pad_nd(key, {0, 0, 0, pad}, 0);
  value = at::constant_pad_nd(value, {0, 0, 0, pad}, 0);
  // Internal padding has beta=0 and log-decay=0: no recurrent state update.
  rate = at::constant_pad_nd(rate, {0, pad}, 0);
  decay = at::constant_pad_nd(decay, {0, pad}, 0);
  auto v_beta = (value * rate.unsqueeze(-1)).reshape({b, h, chunks, chunk, dv});
  auto k_beta = (key * rate.unsqueeze(-1)).reshape({b, h, chunks, chunk, dk});
  query = query.reshape({b, h, chunks, chunk, dk});
  key = key.reshape({b, h, chunks, chunk, dk});
  auto cumulative = decay.reshape({b, h, chunks, chunk}).cumsum(3);
  auto upper = at::ones({chunk, chunk}, q.options().dtype(at::kBool)).triu(1);
  auto pairwise = (cumulative.unsqueeze(4) - cumulative.unsqueeze(3))
                      .masked_fill(upper, -INFINITY).exp();
  auto system = at::matmul(k_beta, key.transpose(-1, -2)) * pairwise;
  auto intra = at::matmul(query, key.transpose(-1, -2)) * pairwise;
  auto decayed_k = k_beta * cumulative.exp().unsqueeze(-1);
  // Match HF eager's unit-lower-triangular solve, without tracing its inverse fallback.
  auto values = at::linalg_solve_triangular(system, v_beta, false, true, true);
  auto keys = at::linalg_solve_triangular(system, decayed_k, false, true, true);
  auto state = initial_state;
  auto output = at::zeros_like(values);
  query = query * cumulative.exp().unsqueeze(-1);
  auto last = cumulative.slice(-1, chunk - 1, chunk);
  key = key * (last - cumulative).exp().unsqueeze(-1);
  auto chunk_decay = last.squeeze(-1).exp().unsqueeze(-1).unsqueeze(-1);
  for (int64_t i = 0; i < chunks; ++i) {
    auto correction = values.select(2, i) - at::matmul(keys.select(2, i), state);
    output.select(2, i).copy_(at::matmul(query.select(2, i), state) +
                             at::matmul(intra.select(2, i), correction));
    state = state * chunk_decay.select(2, i) +
            at::matmul(key.select(2, i).transpose(-1, -2), correction);
  }
  output = output.reshape({b, h, t + pad, dv}).slice(2, 0, t).transpose(1, 2);
  return {output.to(q.scalar_type()).contiguous(), state.contiguous()};
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(xinfer, m) {
  m.def("delta_prefill(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor state) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(xinfer, CUDA, m) { m.impl("delta_prefill", delta_prefill); }

// Referenced by the native probe so the linker retains the registration library.
extern "C" void xinfer_delta_ops_link() {}
