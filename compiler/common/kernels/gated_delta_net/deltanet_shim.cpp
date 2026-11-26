#include "deltanet.h"

#include <torch/csrc/inductor/aoti_torch/utils.h>

extern "C" AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cuda_batched_deltanet(
    AtenTensorHandle qkv, AtenTensorHandle b, AtenTensorHandle a, AtenTensorHandle conv,
    AtenTensorHandle state, AtenTensorHandle weight, AtenTensorHandle A_log,
    AtenTensorHandle dt_bias, AtenTensorHandle cu_query, AtenTensorHandle cuda_cu_query,
    AtenTensorHandle *output) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto result =
        xinfer::batched_deltanet(torch::aot_inductor::resolve_tensor_dispatch_flags(qkv),
                                 torch::aot_inductor::resolve_tensor_dispatch_flags(b),
                                 torch::aot_inductor::resolve_tensor_dispatch_flags(a),
                                 torch::aot_inductor::resolve_tensor_dispatch_flags(conv),
                                 torch::aot_inductor::resolve_tensor_dispatch_flags(state),
                                 torch::aot_inductor::resolve_tensor_dispatch_flags(weight),
                                 torch::aot_inductor::resolve_tensor_dispatch_flags(A_log),
                                 torch::aot_inductor::resolve_tensor_dispatch_flags(dt_bias),
                                 torch::aot_inductor::resolve_tensor_dispatch_flags(cu_query),
                                 torch::aot_inductor::resolve_tensor_dispatch_flags(cuda_cu_query));
    *output = torch::aot_inductor::new_tensor_handle(std::move(result));
  });
}
