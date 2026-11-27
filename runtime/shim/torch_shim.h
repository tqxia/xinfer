#pragma once

// Minimal C ABI bridging Rust to C++-only LibTorch APIs (AOTIModelPackageLoader,
// process runtime settings, thread-local AOTI error strings, and GPU argmax).
// General tensor creation, views, copies, and fills call PyTorch's aoti_torch_*
// C ABI directly from Rust without going through this shim.
//
// Conventions:
// - Fallible functions return 0 on success or non-zero with a NUL-terminated
//   message written into `error->message` (if `error` is non-null).
// - All entry points are `noexcept` and execute under `c10::InferenceMode`.
// - Callers own `XinferPkg*` (release with `xinfer_shim_pkg_close`) and any
//   output `AtenTensorHandle` (release with `aoti_torch_delete_tensor_object`).

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define XINFER_NOEXCEPT noexcept
extern "C" {
#else
#define XINFER_NOEXCEPT
#endif

// Matches PyTorch's opaque handle in <torch/csrc/inductor/aoti_torch/c/shim.h>.
struct AtenTensorOpaque;
typedef struct AtenTensorOpaque *AtenTensorHandle;

// Opaque owner of one torch::inductor::AOTIModelPackageLoader instance.
typedef struct XinferPkg XinferPkg;

typedef struct {
  char message[2048];
} XinferError;

// Static build metadata of the linked LibTorch (used to validate metadata.json).
typedef struct {
  const char *torch_version;
  long cudart_version;
  bool cxx11_abi;
} XinferTorchInfo;

// Populates `out` with compile-time LibTorch version, CUDART version, and ABI flag.
void xinfer_shim_torch_info(XinferTorchInfo *out) XINFER_NOEXCEPT;

// Returns PyTorch's thread-local AOTI error message (empty string if unset).
const char *xinfer_shim_last_torch_error(void) XINFER_NOEXCEPT;

// Pins ATen to 1 CPU thread and disables cuBLAS/cuDNN TF32 for deterministic numerics.
int xinfer_shim_configure_runtime(XinferError *error) XINFER_NOEXCEPT;

// Opens a compiled `.pt2` AOTI package; caller must close `*out` via `xinfer_shim_pkg_close`.
int xinfer_shim_pkg_open(const char *pt2_path, XinferPkg **out, XinferError *error) XINFER_NOEXCEPT;
void xinfer_shim_pkg_close(XinferPkg *pkg) XINFER_NOEXCEPT;

// Invokes `callback(fqn, ctx)` once for each constant FQN declared by `pkg`.
int xinfer_shim_pkg_constant_fqns(XinferPkg *pkg, void (*callback)(const char *fqn, void *ctx),
                                  void *ctx, XinferError *error) XINFER_NOEXCEPT;

// Binds caller-owned weight views into `pkg` with `user_managed = true` and verifies aliasing.
int xinfer_shim_pkg_load_constants(XinferPkg *pkg, const char *const *names,
                                   const AtenTensorHandle *handles, size_t count,
                                   XinferError *error) XINFER_NOEXCEPT;

// Runs `pkg` with borrowed `inputs` and transfers ownership of the single logits output to
// `*out_logits` (caller releases via `aoti_torch_delete_tensor_object`).
int xinfer_shim_pkg_run(XinferPkg *pkg, const AtenTensorHandle *inputs, size_t num_inputs,
                        AtenTensorHandle *out_logits, XinferError *error) XINFER_NOEXCEPT;

// Computes `logits.argmax(-1).item<int64_t>()` on the tensor's resident device.
int xinfer_shim_argmax(AtenTensorHandle logits, int64_t *token, XinferError *error) XINFER_NOEXCEPT;

// Computes `logits.argmax(-1)` on GPU for a 2D `[count, vocab]` tensor and writes
// `count` token IDs to host `tokens`.
int xinfer_shim_argmax_batch(AtenTensorHandle logits, int64_t *tokens, size_t count,
                             XinferError *error) XINFER_NOEXCEPT;

typedef struct {
  float temperature; // <= 0.0 selects greedy argmax
  float top_p;       // (0.0, 1.0); >= 1.0 disables nucleus filtering
  int32_t top_k;     // <= 0 disables top-k filtering
  float uniform;     // deterministic draw in [0.0, 1.0) provided by Rust RNG
} XinferSampleRow;

// Samples one token ID per active row of 2D `[count, vocab]` logits (CPU or CUDA).
// Inactive rows (`active[i] == 0`) are skipped and receive `out_tokens[i] = 0`.
int xinfer_shim_sample_batch(AtenTensorHandle logits, const XinferSampleRow *params,
                             const uint8_t *active, int64_t *out_tokens, size_t count,
                             XinferError *error) XINFER_NOEXCEPT;

#ifdef __cplusplus
}
#endif
#undef XINFER_NOEXCEPT
