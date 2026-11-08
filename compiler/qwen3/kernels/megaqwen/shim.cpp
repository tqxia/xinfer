#include "op.h"

#include <torch/csrc/inductor/aoti_torch/utils.h>

extern "C" AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cuda_megaqwen_decode(
    AtenTensorHandle ids, const AtenTensorHandle *weights, int64_t weights_len,
    const AtenTensorHandle *kv, int64_t kv_len, AtenTensorHandle cos, AtenTensorHandle sin,
    const AtenTensorHandle *workspace, int64_t workspace_len, AtenTensorHandle extent,
    AtenTensorHandle *output) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto weight_tensors = torch::aot_inductor::pointer_to_list<at::Tensor>(weights, weights_len);
    auto kv_tensors = torch::aot_inductor::pointer_to_list<at::Tensor>(kv, kv_len);
    auto workspace_tensors =
        torch::aot_inductor::pointer_to_list<at::Tensor>(workspace, workspace_len);
    auto result = xinfer::megaqwen_decode(
        torch::aot_inductor::resolve_tensor_dispatch_flags(ids), weight_tensors, kv_tensors,
        torch::aot_inductor::resolve_tensor_dispatch_flags(cos),
        torch::aot_inductor::resolve_tensor_dispatch_flags(sin), workspace_tensors,
        torch::aot_inductor::resolve_tensor_dispatch_flags(extent));
    *output = torch::aot_inductor::new_tensor_handle(std::move(result));
  });
}
