// State/dispatch integration probes for AOTI prefill and native MegaQwen decode.
// These deliberately synchronize and clone state; they are not benchmarks.
#include "qwen3/runtime.h"
#include "common/decode.h"
#include "common/settings.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
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

void reject_operation_unchanged(qwen::Runtime &runtime, const std::function<void()> &operation) {
  const auto before = snapshot(runtime);
  const auto storage = pointers(runtime);
  const auto position = runtime.position();
  const auto prefill = runtime.prefill_calls(), decode = runtime.decode_calls();
  const auto mega = runtime.megaqwen_calls();
  bool rejected = false;
  try {
    operation();
  } catch (const std::exception &) {
    rejected = true;
  }
  require(rejected, "Invalid hybrid input was accepted");
  require(position == runtime.position() && prefill == runtime.prefill_calls() &&
              decode == runtime.decode_calls() && mega == runtime.megaqwen_calls(),
          "Rejected input advanced hybrid state/counters");
  require(storage == pointers(runtime), "Rejected input changed state addresses");
  equal_state(runtime.state(), before, "Rejected input mutated KV");
}

void reject_unchanged(qwen::Runtime &runtime, const std::vector<int64_t> &input) {
  reject_operation_unchanged(runtime, [&] { runtime.step(input); });
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

std::vector<int64_t> chunk_lengths(int64_t length, int64_t maximum) {
  std::vector<int64_t> result;
  while (length > 0) {
    const auto count = std::min(length, maximum);
    result.push_back(count);
    length -= count;
  }
  return result;
}

void check_prefill_dispatch(const qwen::Runtime &runtime, int64_t prompt, int64_t maximum) {
  const auto lengths = chunk_lengths(prompt, maximum);
  const auto decode = int64_t(std::count(lengths.begin(), lengths.end(), 1));
  const auto prefill = int64_t(lengths.size()) - decode;
  const auto mega = decode - int64_t(prompt == 1);
  require(runtime.position() == prompt && runtime.prefill_calls() == prefill &&
              runtime.decode_calls() == decode && runtime.megaqwen_calls() == mega,
          "Complete-prompt prefill used incorrect chunk positions or singleton dispatch");
}

void check_future(const qwen::Runtime &runtime, double fill) {
  for (const auto &tensor : runtime.state()) {
    require(at::isfinite(tensor).all().item<bool>(), "Nonfinite long-context KV");
    require(tensor.slice(2, runtime.position()).eq(fill).all().item<bool>(),
            "Long-context execution overwrote future KV");
  }
}

// Long requests exercise many prefill chunks. Retain small past windows around
// the start, the former 1K boundary and the current append position, rather than
// cloning the complete multi-GiB cache after every chunk. The final two capacity
// steps below still check the entire past and future against full checkpoints.
struct PastWindow {
  int64_t start, end;
  Tensors values;
};

std::vector<PastWindow> past_windows(const qwen::Runtime &runtime) {
  const auto position = runtime.position();
  std::vector<PastWindow> result;
  for (const auto &[start, end] : std::vector<std::pair<int64_t, int64_t>>{
           {0, std::min<int64_t>(2, position)},
           {std::min<int64_t>(1023, position), std::min<int64_t>(1025, position)},
           {std::max<int64_t>(0, position - 2), position}}) {
    PastWindow window{start, end, {}};
    for (const auto &tensor : runtime.state())
      window.values.push_back(tensor.slice(2, start, end).clone());
    result.push_back(std::move(window));
  }
  return result;
}

void check_past_windows(const qwen::Runtime &runtime, const std::vector<PastWindow> &before) {
  for (const auto &window : before)
    for (size_t i = 0; i < window.values.size(); ++i)
      require(bits_equal(window.values[i], runtime.state()[i].slice(2, window.start, window.end)),
              "Long-context append overwrote a past KV checkpoint");
}

Tensors prefix_trace(qwen::Runtime &runtime, const qwen::Spec &spec, int64_t prompt, double fill) {
  runtime.reset(fill);
  const auto first = runtime.prefill(tokens(prompt));
  check_output(first, spec);
  check_prefill_dispatch(runtime, prompt, spec.max_prefill);
  const auto next = runtime.step({13});
  check_output(next, spec);
  check_future(runtime, fill);
  Tensors result{first.cpu().clone(), next.cpu().clone()};
  for (const auto &tensor : runtime.state())
    result.push_back(tensor.slice(2, 0, runtime.position()).cpu().clone());
  return result;
}

void check_async_lifetime(const xinfer::fs::path &packages, const qwen::Spec &spec, int64_t prompt,
                          const Tensors &expected, Json &report) {
  // Raw CUDA work must retain borrowed weights, KV and partial-result scratch
  // when the runtime is destroyed before its nondefault stream completes.
  const auto default_stream = c10::cuda::getDefaultCUDAStream(0);
  const auto decode_stream = c10::cuda::getStreamFromPool(false, 0);
  at::Tensor retained_logits;
  Tensors churn;
  size_t churn_bytes = 0;
  {
    c10::cuda::CUDAStreamGuard creation_guard(default_stream);
    {
      auto temporary =
          std::make_unique<qwen::Runtime>(packages, spec, xinfer::DecodeBackendKind::MegaQwen);
      const auto prefill = temporary->prefill(tokens(prompt)).cpu();
      require(bits_equal(prefill, expected[0]), "Temporary lifetime probe prefill differs");
      {
        c10::cuda::CUDAStreamGuard decode_guard(decode_stream);
        retained_logits = temporary->step({13});
      }
      temporary.reset();
    }
    const auto bf16 = at::TensorOptions().device(at::kCUDA, 0).dtype(at::kBFloat16);
    auto fill = [&](const std::vector<int64_t> &shape, at::ScalarType dtype) {
      auto tensor = at::empty(shape, bf16.dtype(dtype));
      tensor.fill_(7);
      churn_bytes += tensor.nbytes();
      churn.push_back(std::move(tensor));
    };
    // Retain each allocation to churn all workspace/partial-result and dense
    // weight bins instead of repeatedly reusing a single allocation.
    for (int repeat = 0; repeat < 10; ++repeat) {
      fill({1024}, at::kBFloat16);
      fill({1024}, at::kFloat);
      fill({2048}, at::kFloat);
      fill({3072}, at::kFloat);
      fill({16, 4}, at::kFloat);
      fill({16, 4}, at::kFloat);
      fill({16, 4, 128}, at::kFloat);
      fill({28 * 11 * int64_t(sizeof(void *))}, at::kByte);
      fill({1024, 1024}, at::kBFloat16);
      fill({2048, 1024}, at::kBFloat16);
      fill({3072, 1024}, at::kBFloat16);
    }
    // Match capacity-dependent KV and RoPE bins once, keeping the long-context
    // probe bounded instead of retaining another complete model cache.
    fill({1, 8, spec.capacity, 128}, at::kBFloat16);
    fill({spec.capacity, 128}, at::kBFloat16);
    fill({spec.capacity, 128}, at::kBFloat16);
    fill({8 * 1024 * 1024}, at::kByte);
    C10_CUDA_CHECK(cudaStreamSynchronize(decode_stream.stream()));
    require(bits_equal(retained_logits.cpu(), expected[1]),
            "Async runtime destruction or allocator reuse changed retained decode logits");
    C10_CUDA_CHECK(cudaStreamSynchronize(default_stream.stream()));
  }
  report["async_destruction_lifetime"] = true;
  report["async_destruction_prompt_tokens"] = prompt;
  report["async_destruction_churn_bytes"] = churn_bytes;
}

void validate_long(const xinfer::fs::path &packages, const qwen::Spec &spec, qwen::Runtime &hybrid,
                   qwen::Runtime &aoti, Json &report) {
  const auto storage = pointers(hybrid);
  report["validation_scope"] = "long_capacity_chunking_handoff_and_full_tail_state";
  report["past_checkpoints_per_long_request"] = "start,1024_boundary,append_boundary";
  for (int64_t prompt : {513, 1025, 8191, 8192, 8193}) {
    if (prompt + 3 > spec.capacity)
      continue;
    hybrid.reset(3);
    aoti.reset(3);
    const auto input = tokens(prompt);
    const auto control = aoti.prefill(input);
    const auto actual = hybrid.prefill(input);
    check_output(actual, spec);
    check_prefill_dispatch(hybrid, prompt, spec.max_prefill);
    const auto prefill_calls = hybrid.prefill_calls(), decode_calls = hybrid.decode_calls();
    const auto mega_calls = hybrid.megaqwen_calls();
    if (mega_calls == 0) {
      require(bits_equal(actual, control), "Chunked AOTI prefill logits differ");
      equal_state(hybrid.state(), aoti.state(), "Chunked AOTI prefill state");
    }
    check_future(hybrid, 3);
    const auto past = past_windows(hybrid);
    check_output(hybrid.step({13}), spec);
    aoti.step({13}); // Advance its position before borrowing the hybrid history.
    check_past_windows(hybrid, past);
    check_future(hybrid, 3);
    for (size_t i = 0; i < hybrid.state().size(); ++i)
      aoti.state()[i].copy_(hybrid.state()[i]);
    const auto appended = tokens(2, prompt + 1);
    const auto returned = hybrid.step(appended);
    require(bits_equal(returned, aoti.step(appended)),
            "Long-context MegaQwen to AOTI return logits differ on identical state");
    equal_state(hybrid.state(), aoti.state(), "Long-context AOTI return state");
    check_future(hybrid, 3);
    require(pointers(hybrid) == storage && hybrid.position() == prompt + 3 &&
                hybrid.prefill_calls() == prefill_calls + 1 &&
                hybrid.decode_calls() == decode_calls + 1 &&
                hybrid.megaqwen_calls() == mega_calls + 1,
            "Long-context handoff changed storage, position or dispatch counts");
    report["cases"].push_back({{"name", "complete_prompt_" + std::to_string(prompt)},
                               {"prompt_tokens", prompt},
                               {"prompt_chunks", chunk_lengths(prompt, spec.max_prefill)},
                               {"prompt_megaqwen_calls", mega_calls},
                               {"position", hybrid.position()},
                               {"passed", true}});
  }

  // Only these final two steps retain complete cache checkpoints. Their peak
  // extra allocation is one cache, independent of the number of prefill chunks.
  hybrid.reset(3);
  check_output(hybrid.prefill(tokens(spec.capacity - 2)), spec);
  check_prefill_dispatch(hybrid, spec.capacity - 2, spec.max_prefill);
  for (int step = 0; step < 2; ++step) {
    const auto start = hybrid.position();
    const auto before = snapshot(hybrid);
    check_output(hybrid.step({13}), spec);
    check_regions(hybrid, before, start, start + 1, storage);
  }
  reject_unchanged(hybrid, {13});
  report["full_capacity_tail"] = {{"capacity", spec.capacity},
                                  {"full_state_steps", 2},
                                  {"overflow_rejected_without_mutation", true}};
  const auto bounded_prompt = std::min<int64_t>(1025, spec.capacity - 1);
  const auto clean = prefix_trace(hybrid, spec, bounded_prompt, 0);
  equal_state(prefix_trace(hybrid, spec, bounded_prompt, 3), clean,
              "Long-context poisoned future replay");
  const auto stream = c10::cuda::getStreamFromPool(false, 0);
  {
    c10::cuda::CUDAStreamGuard guard(stream);
    equal_state(prefix_trace(hybrid, spec, bounded_prompt, 0), clean,
                "Long-context nondefault stream replay");
    C10_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
  }
  report["poisoned_future"] = report["nondefault_stream"] = true;
  report["replay_prefix_tokens"] = bounded_prompt + 1;
  check_async_lifetime(packages, spec, bounded_prompt, clean, report);

  // Request validation must reject before reset or the first compiled chunk.
  reject_operation_unchanged(hybrid, [&] { hybrid.prefill(tokens(spec.capacity + 1)); });
  reject_operation_unchanged(hybrid, [&] { hybrid.generate(tokens(spec.capacity), 1); });
  const int64_t generation_prompt = 513;
  const auto generated = hybrid.generate(tokens(generation_prompt), 1);
  require(generated.size() == 1, "Chunked generation did not return its one-token budget");
  check_prefill_dispatch(hybrid, generation_prompt, spec.max_prefill);
  require(hybrid.generate(tokens(generation_prompt), 1) == generated,
          "Long-context generation reset/reuse changed its token");
  report["generation_chunked_prompt"] = generation_prompt;
  report["generation_reset_reuse"] = true;
  hybrid.reset();
  require(hybrid.position() == 0 && hybrid.prefill_calls() == 0 && hybrid.decode_calls() == 0 &&
              hybrid.megaqwen_calls() == 0 && pointers(hybrid) == storage,
          "Long-context reset changed storage or left counters");
  for (const auto &tensor : hybrid.state())
    require(tensor.eq(0).all().item<bool>(), "Long-context reset left stale KV");
  report["reset_reuse"] = true;
}

void validate(const xinfer::fs::path &packages, Json &report) {
  const qwen::Spec spec(packages);
  qwen::Runtime hybrid(packages, spec, xinfer::DecodeBackendKind::MegaQwen);
  qwen::Runtime aoti(packages, spec, xinfer::DecodeBackendKind::Aoti);
  const auto info = hybrid.decode_info();
  report["backend_info"] = info;
  report["cache_pointers"] = pointers(hybrid);
  report["capacity"] = spec.capacity;
  report["max_prefill"] = spec.max_prefill;
  require(hybrid.state().size() == 56, "Expected 56 shared layer cache views");
  require(info.at("capacity").get<int64_t>() == spec.capacity,
          "MegaQwen capacity differs from package metadata");
  int64_t cache_bytes = 0;
  for (const auto &tensor : hybrid.state()) {
    require(tensor.sizes() == at::IntArrayRef({1, 8, spec.capacity, 128}) &&
                tensor.is_contiguous() && tensor.scalar_type() == at::kBFloat16,
            "MegaQwen KV allocation differs from the metadata capacity");
    cache_bytes += tensor.nbytes();
  }
  report["cache_bytes"] = cache_bytes;
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
  if (spec.capacity > 1024) {
    validate_long(packages, spec, hybrid, aoti, report);
    report["backend_info_after"] = hybrid.decode_info();
    native_libraries(report);
    return;
  }
  report["validation_scope"] = "short_capacity_full_state_contract";
  for (auto length : {1, 2, 25, 128, 512})
    if (length + 2 <= spec.capacity)
      report["cases"].push_back(
          partitions(hybrid, aoti, spec, {length, 1, 1}, "prompt_" + std::to_string(length)));
  for (auto boundary : {128, 256})
    report["cases"].push_back(partitions(hybrid, aoti, spec, {boundary - 1, 1, 1, 2, 1},
                                         "split_transition_" + std::to_string(boundary)));
  report["cases"].push_back(partitions(hybrid, aoti, spec, {2, 1, 13, 1}, "aoti_after_megaqwen"));
  auto capacity_steps = chunk_lengths(spec.capacity - 2, spec.max_prefill);
  capacity_steps.insert(capacity_steps.end(), {1, 1});
  report["cases"].push_back(
      partitions(hybrid, aoti, spec, capacity_steps, "context_capacity_tail"));
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

  auto trace = [&](double fill, int64_t prompt = 2) {
    hybrid.reset(fill);
    Tensors result;
    result.push_back(hybrid.step(tokens(prompt)).cpu().clone());
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
  report["split_poisoned_future"] = Json::array();
  for (auto prompt : {128, 256, 512}) {
    if (prompt + 1 > spec.capacity)
      continue;
    const auto clean_split = trace(0, prompt);
    const auto poisoned_split = trace(3, prompt);
    require(bits_equal(clean_split[0], poisoned_split[0]) &&
                bits_equal(clean_split[1], poisoned_split[1]),
            "Future poison affected split attention logits");
    for (size_t i = 2; i < clean_split.size(); ++i) {
      require(bits_equal(clean_split[i].slice(2, 0, prompt + 1),
                         poisoned_split[i].slice(2, 0, prompt + 1)),
              "Future poison affected valid split attention KV");
      require(poisoned_split[i].slice(2, prompt + 1).eq(3).all().item<bool>(),
              "Split attention overwrote poisoned future");
    }
    report["split_poisoned_future"].push_back({{"prompt_tokens", prompt}, {"passed", true}});
  }
  const auto stream = c10::cuda::getStreamFromPool(false, 0);
  report["nondefault_stream_cases"] = Json::array();
  for (auto prompt : {2, 128, 256}) {
    const auto expected = trace(0, prompt);
    Tensors streamed;
    {
      c10::cuda::CUDAStreamGuard guard(stream);
      streamed = trace(0, prompt);
      C10_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
    }
    equal_state(streamed, expected, "Nondefault stream replay");
    report["nondefault_stream_cases"].push_back({{"prompt_tokens", prompt}, {"passed", true}});
  }
  report["nondefault_stream"] = true;

  const int64_t lifetime_prompt = 256;
  check_async_lifetime(packages, spec, lifetime_prompt, trace(0, lifetime_prompt), report);
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
