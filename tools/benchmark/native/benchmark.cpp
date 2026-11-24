#include "common/model.h"
#include "benchmark.h"

#include <ATen/Parallel.h>

using namespace xinfer;

namespace {
struct Options {
  fs::path packages, workload, output;
};

Options parse_options(int argc, char **argv) {
  require(argc == 4, "Usage: xinfer_benchmark PACKAGE_DIR WORKLOAD_JSON OUTPUT_JSON");
  return {argv[1], argv[2], argv[3]};
}

void validate_workload(const Json &workload, const SpecBase &spec) {
  const int warmup = workload.at("warmup"), repeats = workload.at("repeats");
  require(warmup > 0 && repeats > 0 && !workload.at("cases").empty(), "Invalid benchmark counts");

  for (const auto &item : workload.at("cases")) {
    const auto prompt = item.at("prompt").get<std::vector<int64_t>>();
    const auto tokens = item.at("decode_tokens").get<std::vector<int64_t>>();
    spec.check_request(prompt, tokens.size());
    for (auto id : tokens)
      require(id >= 0 && id < spec.vocab, "Decode token ID out of range");
  }
}

void write_report(const fs::path &path, const Json &report) {
  auto output = output_file(path);
  output << report.dump(2) << '\n';
  output.close();
  require(bool(output), "Failed to write benchmark report");
}

void run_benchmark(const Options &options) {
  const auto workload = read_json(options.workload);
  require(!path_exists(options.output), "Output already exists");
  require(!workload.contains("max_prompt_length"),
          "Benchmark workload no longer accepts max_prompt_length");

  with_model(options.packages, [&](const auto &spec, auto make_runtime) {
    validate_workload(workload, spec);
    c10::InferenceMode inference;

    const auto init = Clock::now();
    checked_device(spec);
    at::empty({0}, at::TensorOptions().device(at::kCUDA));
    torch::cuda::synchronize();
    const double init_ms = milliseconds(init, Clock::now());
    const auto initial_memory = memory_stats();
    c10::cuda::CUDACachingAllocator::resetPeakStats(0);

    const auto load = Clock::now();
    auto runtime = make_runtime(1);
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

    write_report(options.output, report);
  });
}

} // namespace

int main(int argc, char **argv) {
  try {
    run_benchmark(parse_options(argc, argv));
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
