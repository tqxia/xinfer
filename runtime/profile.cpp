// NVTX/CUDA-profiler instrumentation around the unchanged production Runtime.
#include "runtime.h"
#include <ATen/Parallel.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <chrono>

using namespace qwen;
using Clock = std::chrono::steady_clock;

struct Range {
  explicit Range(const char* name) { nvtxRangePushA(name); }
  ~Range() { nvtxRangePop(); }
  Range(const Range&) = delete;
  Range& operator=(const Range&) = delete;
};

double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b-a).count();
}

int64_t step(Runtime& runtime, const std::vector<int64_t>& ids) {
  at::Tensor logits;
  {
    Range range("forward");
    logits = runtime.step(ids);
  }
  Range range("argmax_host");
  return logits.argmax(-1).item<int64_t>();
}

Json request(Runtime& runtime, const Json& item) {
  const auto prompt = item.at("prompt").get<std::vector<int64_t>>();
  const auto tokens = item.at("decode_tokens").get<std::vector<int64_t>>();
  std::vector<int64_t> predictions;
  predictions.reserve(tokens.size()+1);
  torch::cuda::synchronize();
  const auto start = Clock::now();
  {
    Range range("reset");
    runtime.reset();
    torch::cuda::synchronize();
  }
  const auto prefill_start = Clock::now();
  {
    Range range("prefill");
    predictions.push_back(step(runtime, prompt));
    torch::cuda::synchronize();
  }
  const auto decode_start = Clock::now();
  for (auto token : tokens) {
    Range range("decode");
    predictions.push_back(step(runtime, {token}));
  }
  torch::cuda::synchronize();
  const auto end = Clock::now();
  require(runtime.position() == static_cast<int64_t>(prompt.size()+tokens.size()),
          "Unexpected consumed context");
  if (item.contains("expected_predictions"))
    require(Json(predictions) == item.at("expected_predictions"), "Profile predictions differ from baseline");
  return {{"reset_ms", ms(start, prefill_start)}, {"prefill_ms", ms(prefill_start, decode_start)},
          {"decode_ms_per_token", ms(decode_start, end)/tokens.size()},
          {"request_ms", ms(start, end)}, {"prediction_ids", predictions}};
}

int main(int argc, char** argv) {
  try {
    require(argc == 4, "Usage: qwen_profile PACKAGE_DIR WORKLOAD_JSON OUTPUT_JSON");
    require(!fs::exists(argv[3]), "Output already exists");
    const Spec spec(argv[1]);
    const auto workload = read_json(argv[2]);
    const int warmup = workload.at("warmup"), repeats = workload.at("repeats");
    require(warmup >= 1 && repeats >= 1 && !workload.at("cases").empty(), "Invalid profile workload");
    for (const auto& item : workload.at("cases")) {
      const auto prompt = item.at("prompt").get<std::vector<int64_t>>();
      const auto tokens = item.at("decode_tokens").get<std::vector<int64_t>>();
      require(!prompt.empty() && !tokens.empty() && prompt.size() <= static_cast<size_t>(spec.max_prompt) &&
              prompt.size()+tokens.size() <= static_cast<size_t>(spec.cache_length), "Invalid token counts");
      for (const auto* ids : {&prompt, &tokens})
        require(std::all_of(ids->begin(), ids->end(), [&](auto id) {return id >= 0 && id < spec.vocab;}),
                "Invalid token ID");
    }
    c10::InferenceMode inference;
    at::set_num_threads(1);
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    Runtime runtime(argv[1], spec);
    for (const auto& item : workload.at("cases"))
      for (int i=0; i<warmup; ++i) request(runtime, item);
    torch::cuda::synchronize();
    Json report = {{"torch_version", TORCH_VERSION}, {"cpu_threads", at::get_num_threads()},
                   {"warmup", warmup}, {"repeats", repeats}, {"cases", Json::array()}};
    C10_CUDA_CHECK(cudaProfilerStart());
    for (const auto& item : workload.at("cases")) {
      Json result = {{"name", item.at("name")}, {"samples", Json::array()}};
      const auto name = "case:" + item.at("name").get<std::string>();
      for (int i=0; i<repeats; ++i) {
        Range range(name.c_str());
        result["samples"].push_back(request(runtime, item));
      }
      report["cases"].push_back(std::move(result));
    }
    C10_CUDA_CHECK(cudaProfilerStop());
    auto output = output_file(argv[3]);
    output << report.dump(2) << '\n';
    output.flush();
    require(bool(output), "Failed to write profile results");
    std::cout << "Captured " << workload.at("cases").size() << " cases; predictions match baseline\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
