#include "attention.h"

#include <torch/csrc/inductor/aoti_torch/utils.h>

extern "C" AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_cuda_flash_attention(AtenTensorHandle q, AtenTensorHandle k, AtenTensorHandle v,
                                AtenTensorHandle extent, double scale, AtenTensorHandle *output) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto result =
        xinfer::flash_attention(torch::aot_inductor::resolve_tensor_dispatch_flags(q),
                                torch::aot_inductor::resolve_tensor_dispatch_flags(k),
                                torch::aot_inductor::resolve_tensor_dispatch_flags(v),
                                torch::aot_inductor::resolve_tensor_dispatch_flags(extent), scale);
    *output = torch::aot_inductor::new_tensor_handle(std::move(result));
  });
}
