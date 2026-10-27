// Model setup stays here; timing and request loops are shared.
#include "runtime.h"
#include "../common/benchmark.h"
#include <ATen/Parallel.h>

using namespace qwen;
using xinfer::Clock;
using xinfer::milliseconds;
using xinfer::memory_stats;

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
    report["cases"] = xinfer::measure_cases(runtime, workload);
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
