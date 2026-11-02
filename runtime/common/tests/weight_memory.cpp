// Run in a fresh process; cudaMemGetInfo also counts AOTI's raw CUDA allocations.
#include "qwen3/runtime.h"
#include "qwen35/runtime.h"
#include <cuda_runtime_api.h>

#include <chrono>
#include <memory>

using Json = nlohmann::json;

Json memory() {
  torch::cuda::synchronize();
  size_t available = 0, total = 0;
  const auto error = cudaMemGetInfo(&available, &total);
  xinfer::require(error == cudaSuccess, cudaGetErrorString(error));
  return {{"free_bytes", available}, {"total_bytes", total}, {"used_bytes", total - available}};
}

template <class Spec, class Runtime>
void measure(const std::filesystem::path& packages, Json& report) {
  const Spec spec(packages);
  // Initialize the context and GEMM before attributing allocations to a model.
  const auto options = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  auto warmup = at::mm(at::ones({16, 16}, options), at::ones({16, 16}, options));
  report["before_runtime"] = memory();
  auto runtime = std::make_unique<Runtime>(packages, spec);
  report["after_loading"] = memory();
  auto logits = runtime->step({42, 13, 42, 13, 42, 13, 42, 13});
  xinfer::require(at::isfinite(logits).all().template item<bool>(), "Nonfinite prefill logits");
  report["after_prefill"] = memory();
  logits = runtime->step({42});
  xinfer::require(at::isfinite(logits).all().template item<bool>(), "Nonfinite decode logits");
  report["after_decode"] = memory();
  report["loading_growth_bytes"] = report["after_loading"]["used_bytes"].template get<int64_t>() -
      report["before_runtime"]["used_bytes"].template get<int64_t>();
  report["after_decode_growth_bytes"] = report["after_decode"]["used_bytes"].template get<int64_t>() -
      report["before_runtime"]["used_bytes"].template get<int64_t>();
  runtime.reset();
  report["after_runtime_destroyed"] = memory();
  report["passed"] = true;
}

int main(int argc, char** argv) {
  try {
    xinfer::require(argc == 4, "Usage: qwen_weight_memory qwen3|qwen35 PACKAGE_DIR REPORT_JSON");
    c10::InferenceMode inference;
    at::set_num_threads(1);
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    const std::string model(argv[1]);
    Json report = {{"model", model}, {"packages", std::filesystem::absolute(argv[2]).string()},
                   {"torch", TORCH_VERSION}, {"measurement", "cudaMemGetInfo"}, {"passed", false}};
    if (model == "qwen3") measure<qwen::Spec, qwen::Runtime>(argv[2], report);
    else if (model == "qwen35") measure<qwen35::Spec, qwen35::Runtime>(argv[2], report);
    else throw std::runtime_error("Unknown model: " + model);
    auto file = xinfer::output_file(argv[3]);
    file << report.dump(2) << '\n';
    file.close();
    xinfer::require(bool(file), "Failed to save memory report");
    std::cout << report.dump(2) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
