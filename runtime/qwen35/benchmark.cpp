// Fixed-token timing of the same Runtime used by the text-generation CLI.
#include "runtime.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include <chrono>
#include <iostream>

using namespace qwen35;
using Clock = std::chrono::steady_clock;

double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

Json memory() {
  size_t free = 0, total = 0;
  C10_CUDA_CHECK(cudaMemGetInfo(&free, &total));
  const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
  const auto all = static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
  return {{"allocated_bytes", stats.allocated_bytes[all].current}, {"reserved_bytes", stats.reserved_bytes[all].current},
          {"peak_allocated_bytes", stats.allocated_bytes[all].peak}, {"peak_reserved_bytes", stats.reserved_bytes[all].peak},
          {"device_used_bytes", total - free}, {"device_total_bytes", total}};
}

Json request(Runtime& runtime, const std::vector<int64_t>& prompt, const std::vector<int64_t>& tokens) {
  std::vector<int64_t> predictions;
  predictions.reserve(tokens.size() + 1);
  torch::cuda::synchronize();
  const auto begin = Clock::now();
  runtime.reset();
  torch::cuda::synchronize();
  const auto prefill = Clock::now();
  predictions.push_back(runtime.step(prompt).argmax(-1).item<int64_t>());
  torch::cuda::synchronize();
  const auto decode = Clock::now();
  for (const auto token : tokens)  // EOS is data here: every profile does the same work.
    predictions.push_back(runtime.step({token}).argmax(-1).item<int64_t>());
  torch::cuda::synchronize();
  const auto end = Clock::now();
  require(runtime.position() == static_cast<int64_t>(prompt.size() + tokens.size()), "Consumed position mismatch");
  return {{"reset_ms", ms(begin, prefill)}, {"prefill_ms", ms(prefill, decode)},
          {"first_prediction_ms", ms(begin, decode)}, {"decode_ms", ms(decode, end)},
          {"decode_ms_per_token", ms(decode, end) / tokens.size()}, {"request_ms", ms(begin, end)},
          {"prediction_ids", predictions}, {"consumed_positions", runtime.position()}};
}

int main(int argc, char** argv) {
  try {
    require(argc == 4, "Usage: qwen35_benchmark PACKAGE_DIR WORKLOAD_JSON OUTPUT_JSON");
    const Spec spec(argv[1]);
    const auto workload = read_json(argv[2]);
    require(!fs::exists(argv[3]), "Output already exists");
    const int warmup = workload.at("warmup"), repeats = workload.at("repeats");
    require(warmup > 0 && repeats > 0 && !workload.at("cases").empty(), "Invalid benchmark counts");
    for (const auto& item : workload.at("cases")) {
      const auto prompt = item.at("prompt").get<std::vector<int64_t>>();
      const auto tokens = item.at("decode_tokens").get<std::vector<int64_t>>();
      spec.check_request(prompt, tokens.size());
      for (auto id : tokens) require(id >= 0 && id < spec.vocab, "Decode token ID out of range");
    }
    c10::InferenceMode inference;
    const auto init = Clock::now();
    configure(spec);
    at::empty({0}, at::TensorOptions().device(at::kCUDA));
    torch::cuda::synchronize();
    const double init_ms = ms(init, Clock::now());
    const auto initial_memory = memory();
    c10::cuda::CUDACachingAllocator::resetPeakStats(0);
    const auto load = Clock::now();
    Runtime runtime(argv[1], spec);
    torch::cuda::synchronize();
    Json report = {{"profile", "cpp_aoti"}, {"torch", TORCH_VERSION}, {"cpu_threads", at::get_num_threads()},
                   {"cuda_init_ms", init_ms}, {"load_ms", ms(load, Clock::now())},
                   {"initial_memory", initial_memory}, {"load_memory", memory()}, {"cases", Json::array()}};
    for (const auto& item : workload.at("cases")) {
      const auto prompt = item.at("prompt").get<std::vector<int64_t>>();
      const auto tokens = item.at("decode_tokens").get<std::vector<int64_t>>();
      const auto first = request(runtime, prompt, tokens);
      Json row = {{"name", item.at("name")}, {"first_request", first}, {"samples", Json::array()}};
      size_t peak_device = memory().at("device_used_bytes");
      for (int i = 0; i < warmup + repeats; ++i) {
        if (i == warmup) c10::cuda::CUDACachingAllocator::resetPeakStats(0);
        auto sample = request(runtime, prompt, tokens);
        require(sample.at("prediction_ids") == first.at("prediction_ids"), "Repeated predictions differ");
        peak_device = std::max(peak_device, memory().at("device_used_bytes").get<size_t>());
        if (i >= warmup) row["samples"].push_back(std::move(sample));
      }
      row["memory"] = memory();
      row["memory"]["sampled_peak_device_used_bytes"] = peak_device;
      report["cases"].push_back(std::move(row));
      std::cerr << item.at("name") << ": completed\n";
    }
    std::ofstream output(argv[3]);
    output << report.dump(2) << '\n';
    output.close();
    require(bool(output), "Failed to write benchmark report");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
