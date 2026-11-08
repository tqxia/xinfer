#include "op.h"
#include "kernel.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <array>
#include <initializer_list>
#include <unordered_set>
#include <vector>

namespace xinfer {
namespace {

constexpr int kWeightsPerLayer = 11;
constexpr int kWeightCount = 3 + megaqwen::kDecoderLayers * kWeightsPerLayer;
constexpr int kWorkspaceCount = 12;
constexpr int64_t kVocabulary = 151936;

void check_tensor(const at::Tensor &tensor, const c10::Device &device, at::ScalarType dtype,
                  at::IntArrayRef shape, const char *name, uintptr_t alignment = 16) {
  TORCH_CHECK(tensor.defined() && tensor.device() == device && tensor.scalar_type() == dtype &&
                  tensor.sizes() == shape && tensor.is_contiguous(),
              "MegaQwen operator requires contiguous ", name, " with the supported shape/dtype");
  TORCH_CHECK(!tensor.requires_grad(), "MegaQwen operator supports inference only: ", name);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.const_data_ptr()) % alignment == 0,
              "MegaQwen operator received misaligned ", name);
}

bool overlaps(const at::Tensor &left, const at::Tensor &right) {
  const auto a = reinterpret_cast<uintptr_t>(left.const_data_ptr());
  const auto b = reinterpret_cast<uintptr_t>(right.const_data_ptr());
  return a <= b ? b - a < left.nbytes() : a - b < right.nbytes();
}

void check_aliases(const at::Tensor &ids, at::TensorList weights, at::TensorList kv,
                   const at::Tensor &cos, const at::Tensor &sin, at::TensorList workspace) {
  std::vector<const at::Tensor *> readonly{&ids, &cos, &sin};
  for (const auto &tensor : weights)
    readonly.push_back(&tensor);
  std::vector<const at::Tensor *> mutated;
  for (auto tensors : {kv, workspace})
    for (const auto &tensor : tensors) {
      for (const auto *prior : mutated)
        TORCH_CHECK(!overlaps(tensor, *prior), "MegaQwen mutable inputs must not overlap");
      for (const auto *input : readonly)
        TORCH_CHECK(!overlaps(tensor, *input),
                    "MegaQwen mutable inputs must not overlap read-only inputs");
      mutated.push_back(&tensor);
    }
}

void record_stream(const at::Tensor &ids, at::TensorList weights, at::TensorList kv,
                   const at::Tensor &cos, const at::Tensor &sin, at::TensorList workspace,
                   const at::Tensor &output, c10::cuda::CUDAStream stream) {
  std::unordered_set<const void *> recorded;
  auto record = [&](const at::Tensor &tensor) {
    const auto storage = tensor.storage();
    const auto &allocation = storage.data_ptr();
    if (recorded.insert(allocation.get()).second)
      c10::cuda::CUDACachingAllocator::recordStream(allocation, stream);
  };
  for (const auto *tensor : {&ids, &cos, &sin, &output})
    record(*tensor);
  for (auto tensors : {weights, kv, workspace})
    for (const auto &tensor : tensors)
      record(tensor);
}

} // namespace

at::Tensor megaqwen_decode(const at::Tensor &ids, at::TensorList weights, at::TensorList kv,
                           const at::Tensor &cos, const at::Tensor &sin, at::TensorList workspace,
                           const at::Tensor &extent) {
  TORCH_CHECK(ids.defined() && ids.is_cuda(), "MegaQwen operator requires CUDA token IDs");
  const auto device = ids.device();
  c10::cuda::CUDAGuard guard(device);
  check_tensor(ids, device, at::kLong, {1, 1}, "token IDs", alignof(int64_t));
  TORCH_CHECK(weights.size() == kWeightCount && kv.size() == 2 * megaqwen::kDecoderLayers &&
                  workspace.size() == kWorkspaceCount,
              "MegaQwen operator requires 311 weights, 56 KV tensors and 12 workspace tensors");
  TORCH_CHECK(kv[0].defined() && kv[0].dim() == 4,
              "MegaQwen operator requires full-capacity KV tensors");
  const auto capacity = kv[0].size(2);
  TORCH_CHECK(capacity >= 512 && capacity <= 40960,
              "MegaQwen operator requires context capacity in 512..40960");
  for (const auto &tensor : kv)
    check_tensor(tensor, device, at::kBFloat16, {1, 8, capacity, 128}, "KV state");
  check_tensor(weights[0], device, at::kBFloat16, {kVocabulary, 1024}, "embedding");
  check_tensor(weights[1], device, at::kBFloat16, {1024}, "final norm");
  check_tensor(weights[2], device, at::kBFloat16, {kVocabulary, 1024}, "LM head");
  const std::array<std::vector<int64_t>, kWeightsPerLayer> weight_shapes{{{1024},
                                                                          {2048, 1024},
                                                                          {1024, 1024},
                                                                          {1024, 1024},
                                                                          {128},
                                                                          {128},
                                                                          {1024, 2048},
                                                                          {1024},
                                                                          {3072, 1024},
                                                                          {3072, 1024},
                                                                          {1024, 3072}}};
  for (int layer = 0; layer < megaqwen::kDecoderLayers; ++layer)
    for (int item = 0; item < kWeightsPerLayer; ++item)
      check_tensor(weights[3 + layer * kWeightsPerLayer + item], device, at::kBFloat16,
                   weight_shapes[item], "layer weight");
  check_tensor(cos, device, at::kBFloat16, {capacity, 128}, "cosine table");
  check_tensor(sin, device, at::kBFloat16, {capacity, 128}, "sine table");
  const std::array<int64_t, 9> vector_sizes{1024, 1024, 1024, 2048, 1024, 1024, 2048, 3072, 1024};
  for (size_t i = 0; i < vector_sizes.size(); ++i)
    check_tensor(workspace[i], device, i == 0 ? at::kBFloat16 : at::kFloat, {vector_sizes[i]},
                 "workspace vector");
  for (int i : {9, 10})
    check_tensor(workspace[i], device, at::kFloat, {16, megaqwen::kMaxAttentionPartitions},
                 "partial softmax statistic");
  check_tensor(workspace[11], device, at::kFloat, {16, megaqwen::kMaxAttentionPartitions, 128},
               "partial weighted values");
  TORCH_CHECK(extent.defined() && extent.device().is_cpu() && extent.scalar_type() == at::kByte &&
                  extent.dim() == 1 && extent.size(0) >= 2 && extent.size(0) <= capacity + 1,
              "MegaQwen operator requires a CPU uint8 valid-prefix extent in 2..capacity+1");
  check_aliases(ids, weights, kv, cos, sin, workspace);

  megaqwen::LayerPointerTable table{};
  for (int layer = 0; layer < megaqwen::kDecoderLayers; ++layer) {
    const auto w = weights.slice(3 + layer * kWeightsPerLayer, kWeightsPerLayer);
    table.layers[layer] = {
        w[0].const_data_ptr(),       w[1].const_data_ptr(),  w[2].const_data_ptr(),
        w[3].const_data_ptr(),       w[4].const_data_ptr(),  w[5].const_data_ptr(),
        w[6].const_data_ptr(),       w[7].const_data_ptr(),  w[8].const_data_ptr(),
        w[9].const_data_ptr(),       w[10].const_data_ptr(), kv[2 * layer].data_ptr(),
        kv[2 * layer + 1].data_ptr()};
  }
  const auto launch = megaqwen::query_token_launch_info();
  const auto stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto output = at::empty({1, kVocabulary}, kv[0].options());
  megaqwen::Workspace buffers{workspace[0].data_ptr(),
                              workspace[1].data_ptr<float>(),
                              workspace[2].data_ptr<float>(),
                              workspace[3].data_ptr<float>(),
                              workspace[4].data_ptr<float>(),
                              workspace[5].data_ptr<float>(),
                              workspace[6].data_ptr<float>(),
                              workspace[7].data_ptr<float>(),
                              workspace[8].data_ptr<float>(),
                              output.data_ptr(),
                              {workspace[9].data_ptr<float>(), workspace[10].data_ptr<float>(),
                               workspace[11].data_ptr<float>()}};
  record_stream(ids, weights, kv, cos, sin, workspace, output, stream);
  megaqwen::launch_token_logits(
      ids.const_data_ptr<int64_t>(), int(extent.size(0) - 2), int(capacity), table,
      weights[0].const_data_ptr(), weights[1].const_data_ptr(), weights[2].const_data_ptr(),
      cos.const_data_ptr(), sin.const_data_ptr(), buffers, launch.grid_blocks, stream.stream());
  return output;
}

} // namespace xinfer

TORCH_LIBRARY_FRAGMENT(xinfer, m) {
  m.def("megaqwen_decode(Tensor ids, Tensor[] weights, Tensor(a!)[] kv, Tensor cos, Tensor sin, "
        "Tensor(b!)[] workspace, Tensor extent) -> Tensor");
}

TORCH_LIBRARY_IMPL(xinfer, CUDA, m) {
  m.impl("megaqwen_decode", TORCH_FN(xinfer::megaqwen_decode));
}
