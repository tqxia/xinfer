// Minimal production-only C++ bridge for LibTorch APIs that lack a C ABI.
#include "torch_shim.h"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/version.h>

#include <cstdio>
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
