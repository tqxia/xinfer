// State/dispatch integration probes for AOTI prefill and native MegaQwen decode.
// These deliberately synchronize and clone state; they are not benchmarks.
#include "qwen3/runtime.h"
#include "common/decode.h"
#include "common/settings.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {
using xinfer::Json;
using xinfer::require;
using Tensors = std::vector<at::Tensor>;
using Pointers = std::vector<uintptr_t>;

bool bits_equal(const at::Tensor &a, const at::Tensor &b) {
  return a.sizes() == b.sizes() && a.scalar_type() == b.scalar_type() &&
         at::equal(a.contiguous().view(at::kByte), b.contiguous().view(at::kByte));
}

Tensors snapshot(const qwen::Runtime &runtime, bool cpu = false) {
  Tensors result;
  for (const auto &tensor : runtime.state())
    result.push_back(cpu ? tensor.cpu().clone() : tensor.clone());
  return result;
}

Pointers pointers(const qwen::Runtime &runtime) {
  Pointers result;
  for (const auto &tensor : runtime.state())
    result.push_back(reinterpret_cast<uintptr_t>(tensor.const_data_ptr()));
  return result;
}

void equal_state(const Tensors &a, const Tensors &b, const std::string &label) {
  require(a.size() == b.size(), label + ": state count differs");
  for (size_t i = 0; i < a.size(); ++i)
    require(bits_equal(a[i], b[i]), label + ": state tensor " + std::to_string(i) + " differs");
}

void check_output(const at::Tensor &logits, const qwen::Spec &spec) {
  require(logits.sizes() == at::IntArrayRef({1, spec.vocab}) &&
              logits.scalar_type() == at::kBFloat16 && logits.is_cuda(),
          "Incorrect hybrid logit shape/dtype/device");
  require(at::isfinite(logits).all().item<bool>(), "Nonfinite hybrid logits");
}

void check_regions(const qwen::Runtime &runtime, const Tensors &before, int64_t start, int64_t end,
                   const Pointers &storage) {
  require(pointers(runtime) == storage, "Hybrid changed caller-owned KV storage");
  for (size_t i = 0; i < before.size(); ++i) {
    const auto &after = runtime.state()[i];
    require(before[i].sizes() == after.sizes() && before[i].strides() == after.strides() &&
                after.is_contiguous() && after.scalar_type() == at::kBFloat16,
            "Hybrid changed state shape/stride/dtype");
    require(bits_equal(before[i].slice(2, 0, start), after.slice(2, 0, start)),
            "Hybrid overwrote past KV");
    require(bits_equal(before[i].slice(2, end), after.slice(2, end)), "Hybrid overwrote future KV");
    require(!bits_equal(before[i].slice(2, start, end), after.slice(2, start, end)),
            "Hybrid omitted a layer's cache append");
    require(at::isfinite(after).all().item<bool>(), "Nonfinite hybrid KV");
  }
}

std::vector<int64_t> tokens(int64_t count, int64_t offset = 0) {
  const std::vector<int64_t> seed{42, 13, 151645, 0, 151643, 151935, 17};
  std::vector<int64_t> result;
  for (int64_t i = 0; i < count; ++i)
    result.push_back(seed[(offset + i) % seed.size()]);
  return result;
}

void reject_unchanged(qwen::Runtime &runtime, const std::vector<int64_t> &input) {
  const auto before = snapshot(runtime);
  const auto storage = pointers(runtime);
  const auto position = runtime.position();
  const auto prefill = runtime.prefill_calls(), decode = runtime.decode_calls();
  const auto mega = runtime.megaqwen_calls();
  bool rejected = false;
  try {
    runtime.step(input);
  } catch (const std::exception &) {
    rejected = true;
  }
  require(rejected, "Invalid hybrid input was accepted");
  require(position == runtime.position() && prefill == runtime.prefill_calls() &&
              decode == runtime.decode_calls() && mega == runtime.megaqwen_calls(),
          "Rejected input advanced hybrid state/counters");
  require(storage == pointers(runtime), "Rejected input changed state addresses");
  equal_state(snapshot(runtime), before, "Rejected input mutated KV");
}

Json partitions(qwen::Runtime &hybrid, qwen::Runtime &aoti, const qwen::Spec &spec,
                const std::vector<int64_t> &lengths, const std::string &name) {
  hybrid.reset();
  aoti.reset();
  const auto storage = pointers(hybrid);
  int64_t position = 0, prefill = 0, decode = 0, mega = 0, matched = 0;
  for (size_t index = 0; index < lengths.size(); ++index) {
    const auto input = tokens(lengths[index], position);
    const auto before = snapshot(hybrid);
    // For an AOTI stage after MegaQwen, seed the independent AOTI runner with
    // exactly the hybrid history. Equal results then prove the return handoff,
    // rather than incorrectly requiring two different decode engines to agree.
    const bool aoti_stage = index == 0 || lengths[index] > 1;
    if (aoti_stage)
      for (size_t i = 0; i < before.size(); ++i)
        aoti.state()[i].copy_(before[i]);
    const auto control = aoti.step(input);
    const auto actual = hybrid.step(input);
    check_output(actual, spec);
    check_regions(hybrid, before, position, position + lengths[index], storage);
    if (aoti_stage) {
      require(bits_equal(actual, control), "AOTI handoff logits differ on identical state");
      equal_state(snapshot(hybrid), snapshot(aoti), "AOTI handoff cache");
      ++matched;
    } else {
      ++mega;
    }
    position += lengths[index];
    prefill += lengths[index] > 1;
    decode += lengths[index] == 1;
    require(hybrid.position() == position && hybrid.prefill_calls() == prefill &&
                hybrid.decode_calls() == decode && hybrid.megaqwen_calls() == mega,
            "Hybrid dispatched to the wrong backend or advanced the wrong position");
  }
  return {{"name", name},
          {"passed", true},
          {"partitions", lengths},
          {"position", position},
          {"prefill_calls", prefill},
          {"decode_calls", decode},
          {"megaqwen_calls", mega},
          {"identical_state_aoti_stages", matched},
          {"stable_cache_pointers", storage}};
}

void native_libraries(Json &report) {
  std::ifstream maps("/proc/self/maps");
  require(maps.is_open(), "Cannot inspect native process mappings");
  bool attention = false, mega = false;
  for (std::string line; std::getline(maps, line);) {
    require(line.find("libpython") == std::string::npos &&
                line.find("libtorch_python") == std::string::npos,
            "Hybrid native probe loaded a Python library");
    attention = attention || line.find("libxinfer_attention_ops.so") != std::string::npos;
    mega = mega || line.find("libxinfer_megaqwen.so") != std::string::npos;
  }
  require(attention && mega, "Hybrid native probe is missing a required backend library");
  report["no_python_library"] = true;
  report["attention_library"] = attention;
  report["megaqwen_library"] = mega;
}

void validate(const xinfer::fs::path &packages, Json &report) {
  const qwen::Spec spec(packages);
  qwen::Runtime hybrid(packages, spec, xinfer::DecodeBackendKind::MegaQwen);
  qwen::Runtime aoti(packages, spec, xinfer::DecodeBackendKind::Aoti);
  const auto info = hybrid.decode_info();
  report["backend_info"] = info;
  report["cache_pointers"] = pointers(hybrid);
  require(hybrid.state().size() == 56, "Expected 56 shared layer cache views");
  // Backend ownership claims are checked against its retained pointer map by
  // the backend constructor. Require the exposed result, not just usable output.
  require(info.at("weights_alias").get<bool>() && info.at("cache_alias").get<bool>(),
          "MegaQwen did not borrow canonical weights and caller-owned cache");
  require(info.at("cache_pointers").get<Pointers>() == pointers(hybrid),
          "MegaQwen cache pointer table differs from the runner's state");
  const auto borrowed = info.at("borrowed_weight_pointers").get<Pointers>();
  require(!borrowed.empty() && borrowed.size() == info.at("weight_fqns").size(),
          "MegaQwen weight borrowing evidence is incomplete");
  for (auto pointer : borrowed)
    require(pointer != 0 && pointer % 16 == 0, "Invalid borrowed weight pointer");
  report["weights_and_cache_alias"] = true;
  for (auto length : {1, 2, 25, 128, 512})
    report["cases"].push_back(
        partitions(hybrid, aoti, spec, {length, 1, 1}, "prompt_" + std::to_string(length)));
  report["cases"].push_back(partitions(hybrid, aoti, spec, {2, 1, 13, 1}, "aoti_after_megaqwen"));
  report["cases"].push_back(partitions(hybrid, aoti, spec, {512, 510, 1, 1}, "context_1023_1024"));
  reject_unchanged(hybrid, {42});
  report["overflow_rejected_without_mutation"] = true;

  const auto storage = pointers(hybrid);
  hybrid.reset();
  require(hybrid.position() == 0 && hybrid.prefill_calls() == 0 && hybrid.decode_calls() == 0 &&
              hybrid.megaqwen_calls() == 0 && storage == pointers(hybrid),
          "Hybrid reset failed to preserve storage or clear counters");
  for (const auto &tensor : hybrid.state())
    require(tensor.eq(0).all().item<bool>(), "Hybrid reset left stale KV");
  report["reset_reuse"] = partitions(hybrid, aoti, spec, {25, 1}, "reuse_after_full");
  for (const auto &invalid : std::vector<std::vector<int64_t>>{{}, {-1}, {151936}, tokens(513)})
    reject_unchanged(hybrid, invalid);
  report["invalid_inputs_rejected_without_mutation"] = true;

  auto trace = [&](double fill) {
    hybrid.reset(fill);
    Tensors result;
    result.push_back(hybrid.step(tokens(2)).cpu().clone());
    result.push_back(hybrid.step({13}).cpu().clone());
    const auto state = snapshot(hybrid, true);
    result.insert(result.end(), state.begin(), state.end());
    return result;
  };
  const auto clean = trace(0);
  const auto poison = trace(3);
  require(bits_equal(clean[0], poison[0]) && bits_equal(clean[1], poison[1]),
          "Future poison affected prefill or MegaQwen logits");
  for (size_t i = 2; i < clean.size(); ++i) {
    require(bits_equal(clean[i].slice(2, 0, 3), poison[i].slice(2, 0, 3)),
            "Future poison affected valid hybrid KV");
    require(poison[i].slice(2, 3).eq(3).all().item<bool>(), "Hybrid overwrote poisoned future");
  }
  report["poisoned_future"] = true;
  const auto stream = c10::cuda::getStreamFromPool(false, 0);
  Tensors streamed;
  {
    c10::cuda::CUDAStreamGuard guard(stream);
    streamed = trace(0);
    C10_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
  }
  equal_state(streamed, clean, "Nondefault stream replay");
  report["nondefault_stream"] = true;

  // A raw CUDA launch must record allocator use of every borrowed/owned buffer.
  // Keep only its returned Tensor alive while destroying the complete runtime
  // on the allocation stream, before waiting for the decode stream to finish.
  const auto default_stream = c10::cuda::getDefaultCUDAStream(0);
  const auto lifetime_stream = c10::cuda::getStreamFromPool(false, 0);
  at::Tensor retained_logits;
  Tensors churn;
  size_t churn_bytes = 0;
  {
    c10::cuda::CUDAStreamGuard creation_guard(default_stream);
    {
      auto temporary =
          std::make_unique<qwen::Runtime>(packages, spec, xinfer::DecodeBackendKind::MegaQwen);
      const auto prefill = temporary->step(tokens(2)).cpu();
      require(bits_equal(prefill, clean[0]), "Temporary lifetime probe prefill differs");
      {
        c10::cuda::CUDAStreamGuard decode_guard(lifetime_stream);
        retained_logits = temporary->step({13});
      }
      temporary.reset(); // No synchronization before the runtime releases its buffers.
    }
    const auto bf16 = at::TensorOptions().device(at::kCUDA, 0).dtype(at::kBFloat16);
    auto fill = [&](const std::vector<int64_t> &shape, at::ScalarType dtype) {
      auto tensor = at::empty(shape, bf16.dtype(dtype));
      tensor.fill_(7);
      churn_bytes += tensor.nbytes();
      churn.push_back(std::move(tensor));
    };
    // Match small workspace allocations and the 2/4/6 MiB dense-weight buckets.
    // Retaining every allocation prevents reuse of one churn buffer by another.
    for (int repeat = 0; repeat < 10; ++repeat) {
      fill({1024}, at::kBFloat16);
      fill({1024}, at::kFloat);
      fill({2048}, at::kFloat);
      fill({3072}, at::kFloat);
      fill({28 * 11 * int64_t(sizeof(void *))}, at::kByte);
      fill({1024, 1024}, at::kBFloat16);
      fill({2048, 1024}, at::kBFloat16);
      fill({3072, 1024}, at::kBFloat16);
    }
    fill({8 * 1024 * 1024}, at::kByte);
    C10_CUDA_CHECK(cudaStreamSynchronize(lifetime_stream.stream()));
    require(bits_equal(retained_logits.cpu(), clean[1]),
            "Async runtime destruction or allocator reuse changed retained decode logits");
    // Keep churn storage alive until its own writes have also completed.
    C10_CUDA_CHECK(cudaStreamSynchronize(default_stream.stream()));
  }
  report["async_destruction_lifetime"] = true;
  report["async_destruction_churn_bytes"] = churn_bytes;
  require(storage == pointers(hybrid), "Hybrid reallocated state during contract probes");
  report["backend_info_after"] = hybrid.decode_info();
  native_libraries(report);
}
} // namespace

int main(int argc, char **argv) {
  Json report = {{"passed", false}, {"cases", Json::array()}, {"benchmark", false}};
  try {
    require(argc == 3, "Usage: qwen_hybrid_contract PACKAGE_DIR OUTPUT_JSON");
    require(!xinfer::fs::exists(argv[2]), "Output already exists");
    c10::InferenceMode inference;
    xinfer::configure_runtime();
    report["packages"] = xinfer::fs::absolute(argv[1]).string();
    report["torch"] = TORCH_VERSION;
    validate(argv[1], report);
    report["passed"] = true;
  } catch (const std::exception &error) {
    report["error"] = error.what();
    std::cerr << error.what() << '\n';
  }
  if (argc == 3 && !xinfer::fs::exists(argv[2])) {
    try {
      auto file = xinfer::output_file(argv[2]);
      file << report.dump(2) << '\n';
      file.close();
      require(bool(file), "Failed to write hybrid contract report");
    } catch (const std::exception &error) {
      std::cerr << error.what() << '\n';
      return 2;
    }
  }
  return report["passed"].get<bool>() ? 0 : 1;
}
