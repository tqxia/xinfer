// Minimal production-only C++ bridge for LibTorch APIs that lack a C ABI.
#include "torch_shim.h"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/version.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct XinferPkg {
  explicit XinferPkg(const char *pt2_path) : loader(pt2_path) {}

  torch::inductor::AOTIModelPackageLoader loader;
};

namespace {

// Translate C++ exceptions into C status codes under InferenceMode so autograd
// state is never allocated across the FFI boundary.
template <class Function> int xinfer_shim_call(XinferError *error, Function &&function) noexcept {
  try {
    c10::InferenceMode inference;
    function();
    return 0;
  } catch (const std::exception &exception) {
    if (error)
      std::snprintf(error->message, sizeof(error->message), "%s", exception.what());
  } catch (...) {
    if (error)
      std::snprintf(error->message, sizeof(error->message), "Unknown native exception");
  }
  return 1;
}

// Borrow the underlying at::Tensor without taking ownership of `handle`.
const at::Tensor &require_tensor(AtenTensorHandle handle, const char *message) {
  const auto *tensor = torch::aot_inductor::tensor_handle_to_tensor_pointer(handle);
  if (!tensor || !tensor->defined())
    throw std::runtime_error(message);
  return *tensor;
}

std::pair<at::Tensor, at::Tensor> sample_cdf_device(const at::Tensor &probs, float uniform) {
  auto cdf = probs.cumsum(-1);
  auto total = cdf.narrow(0, probs.size(0) - 1, 1);
  auto valid = at::isfinite(total).logical_and(total.gt(0.0));

  // Scale `u` strictly below `cdf[-1]` on device so `searchsorted(..., right=true)`
  // never walks past the last positive-probability entry into masked-out zero tails.
  constexpr float kMaxUniform = 1.0f - 2.0f * std::numeric_limits<float>::epsilon();
  const float u = std::clamp(uniform, 0.0f, kMaxUniform);
  auto max_target = at::nextafter(total, at::zeros_like(total));
  auto target = at::minimum(total * static_cast<double>(u), max_target);

  auto pos = at::searchsorted(cdf, target, false, true).clamp(0, probs.size(0) - 1);
  return {pos, valid};
}

at::Tensor mask_top_p(const at::Tensor &sorted_probs, float top_p) {
  auto cdf = sorted_probs.cumsum(-1);
  // Keep every token whose preceding cumulative mass is strictly below top_p,
  // always retaining at least the highest-probability token.
  auto keep = (cdf - sorted_probs) < static_cast<double>(top_p);
  keep.narrow(0, 0, 1).fill_(true);
  return sorted_probs.masked_fill(keep.logical_not(), 0.0);
}

// Returns a 1-element int64 tensor on `row.device()`; writes -1 if logits are non-finite.
at::Tensor sample_row_device(const at::Tensor &row, const XinferSampleRow &param) {
  if (!std::isfinite(param.temperature))
    throw std::runtime_error("Sampling temperature must be finite");
  if (param.temperature <= 0.0f)
    return row.argmax(-1).reshape({1}).to(at::kLong);
  if (!std::isfinite(param.top_p) || param.top_p <= 0.0f || param.top_p > 1.0f)
    throw std::runtime_error("Sampling top_p must be finite and in (0, 1]");
  if (!std::isfinite(param.uniform))
    throw std::runtime_error("Sampling uniform variate must be finite");

  const int64_t vocab = row.size(0);
  auto scores = row.to(at::kFloat) / static_cast<double>(param.temperature);
  const int64_t top_k = param.top_k > 0 ? std::min<int64_t>(param.top_k, vocab) : 0;
  const bool use_top_p = param.top_p < 1.0f;

  at::Tensor token;
  at::Tensor valid;
  if (top_k > 0) {
    auto [values, indices] = at::topk(scores, top_k, -1, true, true);
    auto probs = at::softmax(values, -1);
    if (use_top_p)
      probs = mask_top_p(probs, param.top_p);
    auto [pos, ok] = sample_cdf_device(probs, param.uniform);
    token = indices.index_select(0, pos);
    valid = ok;
  } else if (use_top_p) {
    auto [values, indices] = at::sort(scores, -1, true);
    auto probs = mask_top_p(at::softmax(values, -1), param.top_p);
    auto [pos, ok] = sample_cdf_device(probs, param.uniform);
    token = indices.index_select(0, pos);
    valid = ok;
  } else {
    auto probs = at::softmax(scores, -1);
    std::tie(token, valid) = sample_cdf_device(probs, param.uniform);
  }

  auto invalid_sentinel = at::full({1}, -1, token.options());
  return at::where(valid, token, invalid_sentinel);
}

} // namespace

void xinfer_shim_torch_info(XinferTorchInfo *out) noexcept {
  *out = {TORCH_VERSION, at::Context::versionCUDART(), bool(_GLIBCXX_USE_CXX11_ABI)};
}

const char *xinfer_shim_last_torch_error(void) noexcept {
  const char *message = torch::aot_inductor::get_last_error();
  return message ? message : "";
}

int xinfer_shim_configure_runtime(XinferError *error) noexcept {
  return xinfer_shim_call(error, [] {
    at::set_num_threads(1);
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
  });
}

int xinfer_shim_pkg_open(const char *pt2_path, XinferPkg **out, XinferError *error) noexcept {
  *out = nullptr;
  return xinfer_shim_call(error, [&] { *out = new XinferPkg(pt2_path); });
}

void xinfer_shim_pkg_close(XinferPkg *pkg) noexcept { delete pkg; }

int xinfer_shim_pkg_constant_fqns(XinferPkg *pkg, void (*callback)(const char *fqn, void *ctx),
                                  void *ctx, XinferError *error) noexcept {
  return xinfer_shim_call(error, [&] {
    for (const auto &name : pkg->loader.get_constant_fqns())
      callback(name.c_str(), ctx);
  });
}

int xinfer_shim_pkg_load_constants(XinferPkg *pkg, const char *const *names,
                                   const AtenTensorHandle *handles, size_t count,
                                   XinferError *error) noexcept {
  return xinfer_shim_call(error, [&] {
    std::unordered_map<std::string, at::Tensor> constants;
    constants.reserve(count);
    for (size_t i = 0; i < count; ++i)
      constants.emplace(names[i], require_tensor(handles[i], "Null constant tensor handle"));

    // Bind directly into the active constant map (`use_inactive=false`), require every
    // constant (`check_full_update=true`), and borrow caller storage (`user_managed=true`).
    pkg->loader.load_constants(constants, false, true, true);

    // Verify AOTI aliased the caller-owned weight pool verbatim rather than cloning.
    const auto actual = pkg->loader.get_runner()->extract_constants_map(false);
    if (actual.size() != constants.size())
      throw std::runtime_error("Shared weights native constant map is incomplete");

    for (const auto &[name, value] : actual) {
      const auto &expected = constants.at(name);
      if (value.data_ptr() != expected.data_ptr() || value.sizes() != expected.sizes() ||
          value.strides() != expected.strides() || value.scalar_type() != expected.scalar_type() ||
          value.device() != expected.device())
        throw std::runtime_error("Shared weights native constant does not alias the retained pool");
    }
  });
}

int xinfer_shim_pkg_run(XinferPkg *pkg, const AtenTensorHandle *inputs, size_t num_inputs,
                        AtenTensorHandle *out_logits, XinferError *error) noexcept {
  *out_logits = nullptr;
  return xinfer_shim_call(error, [&] {
    std::vector<at::Tensor> tensors;
    tensors.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i)
      tensors.push_back(require_tensor(inputs[i], "Null input tensor handle"));

    // All xinfer packages use the logits-only in-place state ABI (exactly 1 output).
    auto outputs = pkg->loader.run(tensors);
    if (outputs.size() != 1)
      throw std::runtime_error("Incorrect package output count");

    // Transfer heap-allocated at::Tensor ownership to Rust (`Tensor::from_raw`).
    *out_logits = torch::aot_inductor::new_tensor_handle(std::move(outputs[0]));
  });
}

int xinfer_shim_argmax(AtenTensorHandle logits, int64_t *token, XinferError *error) noexcept {
  return xinfer_shim_call(error, [&] {
    const auto &tensor = require_tensor(logits, "No logits; execute a step first");
    *token = tensor.argmax(-1).item<int64_t>();
  });
}

int xinfer_shim_argmax_batch(AtenTensorHandle logits, int64_t *tokens, size_t count,
                             XinferError *error) noexcept {
  return xinfer_shim_call(error, [&] {
    const auto &tensor = require_tensor(logits, "No logits; execute a step first");
    if (tensor.dim() != 2 || static_cast<size_t>(tensor.size(0)) != count)
      throw std::runtime_error("Argmax batch size mismatch");
    if (count == 0)
      return;
    auto ids = tensor.argmax(-1).to(at::kCPU, at::kLong, false, false).contiguous();
    std::memcpy(tokens, ids.const_data_ptr<int64_t>(), count * sizeof(int64_t));
  });
}

int xinfer_shim_sample_batch(AtenTensorHandle logits, const XinferSampleRow *params,
                             const uint8_t *active, int64_t *out_tokens, size_t count,
                             XinferError *error) noexcept {
  return xinfer_shim_call(error, [&] {
    const auto &tensor = require_tensor(logits, "Null logits tensor handle");
    if (tensor.dim() != 2 || static_cast<size_t>(tensor.size(0)) != count)
      throw std::runtime_error("Sampler batch size mismatch");
    if (count == 0)
      return;
    if (tensor.size(1) <= 0)
      throw std::runtime_error("Sampler logits vocabulary must be non-empty");

    bool all_greedy = true;
    for (size_t i = 0; i < count; ++i) {
      if (!active[i])
        continue;
      if (!std::isfinite(params[i].temperature))
        throw std::runtime_error("Sampling temperature must be finite");
      if (params[i].temperature > 0.0f) {
        all_greedy = false;
        break;
      }
    }

    if (all_greedy) {
      auto ids = tensor.argmax(-1).to(at::kCPU, at::kLong, false, false).contiguous();
      const auto *src = ids.const_data_ptr<int64_t>();
      for (size_t i = 0; i < count; ++i)
        out_tokens[i] = active[i] ? src[i] : 0;
      return;
    }

    std::vector<size_t> active_rows;
    std::vector<at::Tensor> sampled_device;
    active_rows.reserve(count);
    sampled_device.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      if (!active[i]) {
        out_tokens[i] = 0;
        continue;
      }
      active_rows.push_back(i);
      sampled_device.push_back(sample_row_device(tensor[static_cast<int64_t>(i)], params[i]));
    }
    if (sampled_device.empty())
      return;

    // Transfer all active sampled token IDs in a single D2H sync.
    auto ids = at::cat(sampled_device, 0).to(at::kCPU, at::kLong, false, false).contiguous();
    const auto *src = ids.const_data_ptr<int64_t>();
    for (size_t k = 0; k < active_rows.size(); ++k) {
      if (src[k] < 0)
        throw std::runtime_error("Non-finite logits encountered during sampling");
      out_tokens[active_rows[k]] = src[k];
    }
  });
}
