#include "../../common/decode.h"
#include "../../common/weights.h"
#include "kernel.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <array>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace xinfer {
namespace {

constexpr const char *kUpstreamCommit = "0bc1fdedb82744481373000c5c7d3c58795c7a3a";
constexpr const char *kArithmeticMode = "eager_bf16_boundaries_fp32_accumulation";
constexpr const char *kRepairedSourceSha256 =
    "21d42a0423538b591dfe206c33127b1b3502214196dd6683714190206bb786f9";

uint64_t address(const at::Tensor &tensor) {
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor.const_data_ptr()));
}

class MegaQwenDecode final : public NativeDecode {
public:
  MegaQwenDecode(const SharedWeights &weights, const std::vector<at::Tensor> &state,
                 int64_t capacity, int64_t vocab)
      : state_(state), capacity_(capacity), vocab_(vocab) {
    require(capacity_ == 1024 && vocab_ == 151936,
            "MegaQwen requires Qwen3-0.6B capacity 1024 and vocabulary 151936");
    require(state_.size() == 56 && state_[0].defined() && state_[0].is_cuda(),
            "MegaQwen requires 56 CUDA KV state tensors");
    device_ = state_[0].device();
    c10::cuda::CUDAGuard guard(device_);
    for (const auto &tensor : state_)
      check_tensor(tensor, {1, 8, capacity_, 128}, "KV state");

    embedding_ = borrow(weights, "model.model.embed_tokens.weight", {151936, 1024});
    final_norm_ = borrow(weights, "model.model.norm.weight", {1024});
    head_ = borrow(weights, "model.lm_head.weight", {151936, 1024});
    std::vector<megaqwen::LayerPointers> pointers;
    pointers.reserve(28);
    for (int layer = 0; layer < 28; ++layer) {
      const std::string prefix = "model.model.layers." + std::to_string(layer) + ".";
      const std::string attention = prefix + "self_attn.original.";
      auto get = [&](const std::string &name, std::initializer_list<int64_t> shape) {
        return borrow(weights, name, shape).const_data_ptr();
      };
      pointers.push_back(
          {get(prefix + "input_layernorm.weight", {1024}),
           get(attention + "q_proj.weight", {2048, 1024}),
           get(attention + "k_proj.weight", {1024, 1024}),
           get(attention + "v_proj.weight", {1024, 1024}), get(attention + "q_norm.weight", {128}),
           get(attention + "k_norm.weight", {128}), get(attention + "o_proj.weight", {1024, 2048}),
           get(prefix + "post_attention_layernorm.weight", {1024}),
           get(prefix + "mlp.gate_proj.weight", {3072, 1024}),
           get(prefix + "mlp.up_proj.weight", {3072, 1024}),
           get(prefix + "mlp.down_proj.weight", {1024, 3072}), state_[2 * layer].data_ptr(),
           state_[2 * layer + 1].data_ptr()});
    }

    // Reject unsupported launch resources before allocating backend workspace.
    launch_ = megaqwen::query_launch_info();
    const auto bf16 = state_[0].options();
    const auto fp32 = bf16.dtype(at::kFloat);
    const auto table_bytes = int64_t(pointers.size() * sizeof(megaqwen::LayerPointers));
    pointer_table_ = at::empty({table_bytes}, bf16.dtype(at::kByte));
    C10_CUDA_CHECK(cudaMemcpy(pointer_table_.data_ptr(), pointers.data(), size_t(table_bytes),
                              cudaMemcpyHostToDevice));

    // Match the repaired Python loader's CPU FP32 RoPE construction, then copy
    // the small BF16 tables once. The bundle's model contract fixes theta=1e6.
    const auto cpu_float = at::TensorOptions().device(at::kCPU).dtype(at::kFloat);
    const auto exponents = at::arange(0, 128, 2, cpu_float) / 128.0;
    const auto inverse_frequency = at::reciprocal(at::pow(1000000.0, exponents));
    const auto positions = at::arange(capacity_, cpu_float);
    const auto frequencies = positions.unsqueeze(1) * inverse_frequency.unsqueeze(0);
    const auto cosine = at::cos(frequencies);
    const auto sine = at::sin(frequencies);
    cos_ = at::cat({cosine, cosine}, 1).to(at::kBFloat16).to(device_).contiguous();
    sin_ = at::cat({sine, sine}, 1).to(at::kBFloat16).to(device_).contiguous();

    hidden_ = at::empty({1024}, bf16);
    activations_ = at::empty({1024}, fp32);
    residual_ = at::empty({1024}, fp32);
    q_ = at::empty({2048}, fp32);
    key_ = at::empty({1024}, fp32);
    value_ = at::empty({1024}, fp32);
    attention_ = at::empty({2048}, fp32);
    mlp_ = at::empty({3072}, fp32);
    normalized_ = at::empty({1024}, fp32);
    workspace_ = {hidden_.data_ptr(),
                  activations_.data_ptr<float>(),
                  residual_.data_ptr<float>(),
                  q_.data_ptr<float>(),
                  key_.data_ptr<float>(),
                  value_.data_ptr<float>(),
                  attention_.data_ptr<float>(),
                  mlp_.data_ptr<float>(),
                  normalized_.data_ptr<float>(),
                  nullptr};
    workspace_bytes_ = 0;
    for (const auto &tensor : {pointer_table_, cos_, sin_, hidden_, activations_, residual_, q_,
                               key_, value_, attention_, mlp_, normalized_})
      workspace_bytes_ += tensor.nbytes();
    // The copies above are pointer/metadata tables and scratch only. Every
    // retained model/cache tensor is a shallow alias of the existing pool.
    for (size_t i = 0; i < weight_aliases_.size(); ++i)
      require(weight_aliases_[i].const_data_ptr() ==
                  weights.source_tensor(weight_fqns_[i]).const_data_ptr(),
              "MegaQwen weight must alias the retained SharedWeights pool");
    for (size_t i = 0; i < state_.size(); ++i)
      require(state_[i].const_data_ptr() == state[i].const_data_ptr(),
              "MegaQwen KV state must alias caller storage");
  }

  at::Tensor run(int64_t token, int64_t position) override {
    require(token >= 0 && token < vocab_, "MegaQwen token ID is outside the vocabulary");
    require(position >= 0 && position < capacity_, "MegaQwen position exceeds KV capacity");
    c10::cuda::CUDAGuard guard(device_);
    const auto stream = c10::cuda::getCurrentCUDAStream(device_.index());
    record_stream(stream);
    // Each returned Tensor owns independent output storage across later steps.
    auto logits = at::empty({1, vocab_}, state_[0].options());
    auto workspace = workspace_;
    workspace.logits = logits.data_ptr();
    megaqwen::launch_logits(
        int(token), int(position), int(capacity_),
        reinterpret_cast<const megaqwen::LayerPointers *>(pointer_table_.const_data_ptr()),
        embedding_.const_data_ptr(), final_norm_.const_data_ptr(), head_.const_data_ptr(),
        cos_.const_data_ptr(), sin_.const_data_ptr(), workspace, launch_.grid_blocks,
        stream.stream());
    return logits;
  }

  Json info() const override {
    Json cache_pointers = Json::array();
    Json weight_pointers = Json::array();
    for (const auto &tensor : state_)
      cache_pointers.push_back(address(tensor));
    for (const auto &tensor : weight_aliases_)
      weight_pointers.push_back(address(tensor));
    return Json{{"backend", "megaqwen"},
                {"source_version", kUpstreamCommit},
                {"repaired_source_sha256", kRepairedSourceSha256},
                {"arithmetic_mode", kArithmeticMode},
                {"grid_blocks", launch_.grid_blocks},
                {"sm_count", launch_.sm_count},
                {"threads_per_block", launch_.threads_per_block},
                {"active_blocks_per_sm", launch_.active_blocks_per_sm},
                {"registers_per_thread", launch_.registers_per_thread},
                {"static_shared_bytes", launch_.static_shared_bytes},
                {"workspace_bytes", workspace_bytes_},
                {"weights_alias", true},
                {"cache_alias", true},
                {"cache_pointers", std::move(cache_pointers)},
                {"weight_fqns", weight_fqns_},
                {"borrowed_weight_pointers", std::move(weight_pointers)},
                {"weight_source_count", weight_aliases_.size()},
                {"validated", true},
                {"validation_scope", "shape_dtype_stride_alignment_device_and_alias_contract"},
                {"numerical_parity_validated", false},
                {"capacity", capacity_},
                {"vocab_size", vocab_},
                {"rope_theta", 1000000.0},
                {"layers", 28},
                {"launches_per_decode", 2},
                {"logits_reused", false},
                {"output_bytes_per_decode", vocab_ * int64_t(sizeof(at::BFloat16))},
                {"internal_position_state", false},
                {"host_synchronization_in_run", false}};
  }

private:
  void record_stream(c10::cuda::CUDAStream stream) {
    if (recorded_streams_.contains(stream.id()))
      return;
    // Raw CUDA launches bypass ATen's allocator integration. Retain the
    // allocation on every stream that uses it, including when the runner is
    // destroyed before asynchronous work completes. This records lifetime;
    // the caller still orders work when switching streams.
    std::unordered_set<const void *> recorded_allocations;
    auto record = [&](const at::Tensor &tensor) {
      const auto storage = tensor.storage();
      const auto &allocation = storage.data_ptr();
      if (recorded_allocations.insert(allocation.get()).second)
        c10::cuda::CUDACachingAllocator::recordStream(allocation, stream);
    };
    for (const auto &tensor : weight_aliases_)
      record(tensor);
    for (const auto &tensor : state_)
      record(tensor);
    for (const auto &tensor : {pointer_table_, cos_, sin_, hidden_, activations_, residual_, q_,
                               key_, value_, attention_, mlp_, normalized_})
      record(tensor);
    // Mark complete only after all storage registrations succeeded.
    recorded_streams_.insert(stream.id());
  }

  void check_tensor(const at::Tensor &tensor, std::initializer_list<int64_t> shape,
                    const std::string &name) const {
    require(tensor.defined() && tensor.device() == device_ &&
                tensor.scalar_type() == at::kBFloat16 && tensor.is_contiguous() &&
                tensor.sizes() == at::IntArrayRef(shape),
            "MegaQwen requires contiguous BF16 " + name + " on the KV CUDA device");
    require(reinterpret_cast<uintptr_t>(tensor.const_data_ptr()) % 16 == 0,
            "MegaQwen requires 16-byte aligned " + name);
  }

  at::Tensor borrow(const SharedWeights &pool, const std::string &name,
                    std::initializer_list<int64_t> shape) {
    const auto &tensor = pool.source_tensor(name);
    check_tensor(tensor, shape, name);
    weight_aliases_.push_back(tensor);
    weight_fqns_.push_back(name);
    return tensor;
  }

  std::vector<at::Tensor> state_, weight_aliases_;
  std::vector<std::string> weight_fqns_;
  std::unordered_set<c10::StreamId> recorded_streams_;
  int64_t capacity_, vocab_;
  c10::Device device_{at::kCUDA, 0};
  megaqwen::LaunchInfo launch_{};
  megaqwen::Workspace workspace_{};
  size_t workspace_bytes_ = 0;
  at::Tensor embedding_, final_norm_, head_, pointer_table_, cos_, sin_;
  at::Tensor hidden_, activations_, residual_, q_, key_, value_, attention_, mlp_, normalized_;
};

} // namespace

bool megaqwen_decode_available() { return true; }

std::unique_ptr<NativeDecode> make_megaqwen_decode(const SharedWeights &weights,
                                                   const std::vector<at::Tensor> &state,
                                                   int64_t capacity, int64_t vocab) {
  return std::make_unique<MegaQwenDecode>(weights, state, capacity, vocab);
}

} // namespace xinfer
