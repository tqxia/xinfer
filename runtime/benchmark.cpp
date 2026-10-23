// Fixed-token benchmark using the same checked Runtime as the production CLI.
#include "runtime.h"
#include <ATen/Parallel.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <chrono>

using namespace qwen;
using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

Json memory_stats() {
  size_t free = 0, total = 0;
  C10_CUDA_CHECK(cudaMemGetInfo(&free, &total));
  const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
  const auto all = static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
  return {{"allocated_bytes", stats.allocated_bytes[all].current},
          {"reserved_bytes", stats.reserved_bytes[all].current},
          {"peak_allocated_bytes", stats.allocated_bytes[all].peak},
          {"peak_reserved_bytes", stats.reserved_bytes[all].peak},
          {"device_used_bytes", total - free}, {"device_total_bytes", total}};
}

Json request(Runtime& runtime, const std::vector<int64_t>& prompt,
             const std::vector<int64_t>& tokens) {
  std::vector<int64_t> predictions;
  predictions.reserve(tokens.size() + 1);
  torch::cuda::synchronize();
  const auto begin = Clock::now();
  runtime.reset();
  torch::cuda::synchronize();
  const auto prefill_start = Clock::now();
  predictions.push_back(runtime.step(prompt).argmax(-1).item<int64_t>());
  torch::cuda::synchronize();
  const auto decode_start = Clock::now();
  // Teacher-forced tokens keep all profiles on identical inputs. EOS is data.
  for (const auto token : tokens)
    predictions.push_back(runtime.step({token}).argmax(-1).item<int64_t>());
  torch::cuda::synchronize();
  const auto end = Clock::now();
  require(runtime.position() == static_cast<int64_t>(prompt.size()+tokens.size()),
          "Benchmark did not consume the expected context");
  const double decode_ms = milliseconds(decode_start, end);
  return {{"reset_ms", milliseconds(begin, prefill_start)},
          {"prefill_ms", milliseconds(prefill_start, decode_start)},
          {"decode_ms", decode_ms}, {"decode_ms_per_token", decode_ms / tokens.size()},
          {"request_ms", milliseconds(begin, end)}, {"prediction_ids", predictions}};
}

int main(int argc, char** argv) {
  try {
    require(argc == 4, "Usage: qwen_benchmark PACKAGE_DIR WORKLOAD_JSON OUTPUT_JSON");
    const Spec spec(argv[1]);
    const auto workload = read_json(argv[2]);
    require(!fs::exists(argv[3]), "Output already exists");
    const int warmup = workload.at("warmup"), repeats = workload.at("repeats");
    require(warmup >= 1 && repeats >= 1, "Warmup and repeats must be positive");
    for (const auto& item : workload.at("cases")) {
      const auto prompt = item.at("prompt").get<std::vector<int64_t>>();
      const auto tokens = item.at("decode_tokens").get<std::vector<int64_t>>();
      require(!prompt.empty() && prompt.size() <= static_cast<size_t>(spec.max_prompt) &&
              !tokens.empty() && prompt.size() + tokens.size() <= static_cast<size_t>(spec.cache_length),
              "Invalid benchmark token counts");
      for (const auto* ids : {&prompt, &tokens})
        require(std::all_of(ids->begin(), ids->end(), [&](int64_t t) { return t >= 0 && t < spec.vocab; }),
                "Benchmark token ID out of range");
    }
    c10::InferenceMode inference;
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    const auto init_start = Clock::now();
    at::empty({0}, at::TensorOptions().device(at::kCUDA));
    torch::cuda::synchronize();
    const auto init_end = Clock::now();
    const auto initial_memory = memory_stats();
    c10::cuda::CUDACachingAllocator::resetPeakStats(0);
    const auto load_start = Clock::now();
    Runtime runtime(argv[1], spec);
    torch::cuda::synchronize();
    const auto load_end = Clock::now();
    Json report = {{"kind", "native_aoti"}, {"torch_version", TORCH_VERSION},
                   {"cpu_threads", at::get_num_threads()},
                   {"cuda_init_ms", milliseconds(init_start, init_end)},
                   {"load_ms", milliseconds(load_start, load_end)},
                   {"initial_memory", initial_memory},
                   {"load_memory", memory_stats()}, {"cases", Json::array()}};
    for (const auto& item : workload.at("cases")) {
      const auto prompt = item.at("prompt").get<std::vector<int64_t>>();
      const auto tokens = item.at("decode_tokens").get<std::vector<int64_t>>();
      Json result = {{"name", item.at("name")}, {"prompt_tokens", prompt.size()},
                     {"decode_steps", tokens.size()}, {"first_request", request(runtime, prompt, tokens)},
                     {"samples", Json::array()}};
      auto peak_device = memory_stats().at("device_used_bytes").get<size_t>();
      const auto sample_memory = [&] {
        peak_device = std::max(peak_device, memory_stats().at("device_used_bytes").get<size_t>());
      };
      for (int i = 0; i < warmup; ++i) { request(runtime, prompt, tokens); sample_memory(); }
      c10::cuda::CUDACachingAllocator::resetPeakStats(0);
      for (int i = 0; i < repeats; ++i) {
        auto sample = request(runtime, prompt, tokens);
        require(sample.at("prediction_ids") == result.at("first_request").at("prediction_ids"),
                "Repeated request predictions differ");
        result["samples"].push_back(std::move(sample));
        sample_memory();
      }
      result["memory"] = memory_stats();
      result["memory"]["sampled_peak_device_used_bytes"] = peak_device;
      report["cases"].push_back(std::move(result));
      std::cerr << item.at("name") << ": " << repeats << " measured requests\n";
    }
    auto output = output_file(argv[3]);
    output << report.dump(2) << '\n';
    output.flush();
    require(bool(output), "Failed to write benchmark report");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
