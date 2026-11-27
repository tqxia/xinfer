// Tool-only GPU tensor check helpers for xinfer_validate (statically linked only
// into xinfer_validate; never exposed in production xinfer_aoti). Running these
// reductions on-device avoids copying full KV caches back to host every step.
#include "shim/torch_shim.h"

#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

#include <cstdio>
#include <stdexcept>

namespace {

template <class Function> int validate_call(XinferError *error, Function &&function) noexcept {
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

const at::Tensor &require_tensor(AtenTensorHandle handle) {
  const auto *tensor = torch::aot_inductor::tensor_handle_to_tensor_pointer(handle);
  if (!tensor || !tensor->defined())
    throw std::runtime_error("Null AtenTensorHandle");
  return *tensor;
}

} // namespace

// True iff every element in `tensor` is finite (no NaN or Inf).
extern "C" int xinfer_validate_all_finite(AtenTensorHandle tensor, bool *out,
                                          XinferError *error) noexcept {
  return validate_call(error,
                       [&] { *out = at::isfinite(require_tensor(tensor)).all().item<bool>(); });
}

// Element-wise exact equality (shape, dtype, and values must match).
extern "C" int xinfer_validate_equal(AtenTensorHandle a, AtenTensorHandle b, bool *out,
                                     XinferError *error) noexcept {
  return validate_call(error, [&] { *out = at::equal(require_tensor(a), require_tensor(b)); });
}

// True iff every element equals `value` under floating/numeric equality.
extern "C" int xinfer_validate_all_eq_scalar(AtenTensorHandle tensor, double value, bool *out,
                                             XinferError *error) noexcept {
  return validate_call(error, [&] { *out = require_tensor(tensor).eq(value).all().item<bool>(); });
}

// Bitwise byte comparison against `full_like(tensor, fill)` so future-KV checks
// catch sign-bit flips (-0.0 vs +0.0) and NaNs that numeric `.eq()` can mask.
extern "C" int xinfer_validate_all_eq_fill_bytes(AtenTensorHandle tensor, double fill, bool *out,
                                                 XinferError *error) noexcept {
  return validate_call(error, [&] {
    const auto &actual = require_tensor(tensor);
    const auto expected = at::full_like(actual, fill);
    *out = at::equal(actual.contiguous().view(at::kByte), expected.contiguous().view(at::kByte));
  });
}
