// Model setup stays here; timing and request loops are shared.
#include "runtime.h"
#include "../common/benchmark.h"
#include <ATen/Parallel.h>

using namespace qwen35;
using xinfer::Clock;
using xinfer::milliseconds;
using xinfer::memory_stats;

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
    const double init_ms = milliseconds(init, Clock::now());
    const auto initial_memory = memory_stats();
    c10::cuda::CUDACachingAllocator::resetPeakStats(0);
    const auto load = Clock::now();
    Runtime runtime(argv[1], spec);
    torch::cuda::synchronize();
    Json report = {{"profile", "cpp_aoti"}, {"torch", TORCH_VERSION}, {"cpu_threads", at::get_num_threads()},
                   {"cuda_init_ms", init_ms}, {"load_ms", milliseconds(load, Clock::now())},
                   {"initial_memory", initial_memory}, {"load_memory", memory_stats()}, {"cases", Json::array()}};
    report["cases"] = xinfer::measure_cases(runtime, workload);
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
