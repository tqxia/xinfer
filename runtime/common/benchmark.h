#pragma once
#include "io.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/cuda.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

namespace xinfer {
using Clock = std::chrono::steady_clock;

inline double milliseconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

inline Json memory_stats() {
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

template <class Runtime>
Json measure_request(Runtime& runtime, const std::vector<int64_t>& prompt,
                     const std::vector<int64_t>& tokens) {
  std::vector<int64_t> predictions;
  predictions.reserve(tokens.size() + 1);
  torch::cuda::synchronize();
  const auto begin = Clock::now();
  runtime.reset();
  torch::cuda::synchronize();
  const auto prefill = Clock::now();
  predictions.push_back(runtime.step(prompt).argmax(-1).template item<int64_t>());
  torch::cuda::synchronize();
  const auto decode = Clock::now();
  // EOS remains data: every profile consumes the identical fixed workload.
  for (auto token : tokens)
    predictions.push_back(runtime.step({token}).argmax(-1).template item<int64_t>());
  torch::cuda::synchronize();
  const auto end = Clock::now();
  require(runtime.position() == static_cast<int64_t>(prompt.size() + tokens.size()),
          "Consumed position mismatch");
  return {{"reset_ms", milliseconds(begin, prefill)}, {"prefill_ms", milliseconds(prefill, decode)},
          {"first_prediction_ms", milliseconds(begin, decode)}, {"decode_ms", milliseconds(decode, end)},
          {"decode_ms_per_token", milliseconds(decode, end) / tokens.size()},
          {"request_ms", milliseconds(begin, end)}, {"prediction_ids", predictions},
          {"consumed_positions", runtime.position()}};
}

template <class Runtime>
Json measure_cases(Runtime& runtime, const Json& workload) {
  const int warmup = workload.at("warmup"), repeats = workload.at("repeats");
  Json cases = Json::array();
  for (const auto& item : workload.at("cases")) {
    const auto prompt = item.at("prompt").get<std::vector<int64_t>>();
    const auto tokens = item.at("decode_tokens").get<std::vector<int64_t>>();
    const auto first = measure_request(runtime, prompt, tokens);
    Json row = {{"name", item.at("name")}, {"prompt_tokens", prompt.size()},
                {"decode_steps", tokens.size()}, {"first_request", first}, {"samples", Json::array()}};
    size_t peak_device = memory_stats().at("device_used_bytes");
    for (int i = 0; i < warmup + repeats; ++i) {
      if (i == warmup) c10::cuda::CUDACachingAllocator::resetPeakStats(0);
      auto sample = measure_request(runtime, prompt, tokens);
      require(sample.at("prediction_ids") == first.at("prediction_ids"), "Repeated predictions differ");
      peak_device = std::max(peak_device, memory_stats().at("device_used_bytes").get<size_t>());
      if (i >= warmup) row["samples"].push_back(std::move(sample));
    }
    row["memory"] = memory_stats();
    row["memory"]["sampled_peak_device_used_bytes"] = peak_device;
    cases.push_back(std::move(row));
    std::cerr << item.at("name") << ": completed\n";
  }
  return cases;
}

}  // namespace xinfer
