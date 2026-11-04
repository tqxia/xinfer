#include "common/model.h"
#include "common/benchmark.h"
#include <ATen/Parallel.h>

using namespace xinfer;

int main(int argc, char **argv) {
  try {
    require(argc == 4, "Usage: xinfer_benchmark PACKAGE_DIR WORKLOAD_JSON OUTPUT_JSON");
    const auto workload = read_json(argv[2]);
    require(!fs::exists(argv[3]), "Output already exists");
    return with_model(argv[1], [&](const auto &spec, auto make_runtime) {
      const int warmup = workload.at("warmup"), repeats = workload.at("repeats");
      require(warmup > 0 && repeats > 0 && !workload.at("cases").empty(),
              "Invalid benchmark counts");
      for (const auto &item : workload.at("cases")) {
        const auto prompt = item.at("prompt").template get<std::vector<int64_t>>();
        const auto tokens = item.at("decode_tokens").template get<std::vector<int64_t>>();
        spec.check_request(prompt, tokens.size());
        for (auto id : tokens)
          require(id >= 0 && id < spec.vocab, "Decode token ID out of range");
      }
      c10::InferenceMode inference;
      const auto init = Clock::now();
      checked_device(spec);
      at::empty({0}, at::TensorOptions().device(at::kCUDA));
      torch::cuda::synchronize();
      const double init_ms = milliseconds(init, Clock::now());
      const auto initial_memory = memory_stats();
      c10::cuda::CUDACachingAllocator::resetPeakStats(0);
      const auto load = Clock::now();
      auto runtime = make_runtime();
      torch::cuda::synchronize();
      Json report = {{"profile", "cpp_aoti"},
                     {"torch", TORCH_VERSION},
                     {"cpu_threads", at::get_num_threads()},
                     {"cuda_init_ms", init_ms},
                     {"load_ms", milliseconds(load, Clock::now())},
                     {"initial_memory", initial_memory},
                     {"load_memory", memory_stats()},
                     {"cases", Json::array()}};
      report["cases"] = measure_cases(*runtime, workload);
      auto output = output_file(argv[3]);
      output << report.dump(2) << '\n';
      output.close();
      require(bool(output), "Failed to write benchmark report");
      return 0;
    });
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
