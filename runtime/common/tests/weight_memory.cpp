// Fresh-process boundary samples; CUDA free memory also sees raw AOTI allocations.
#include "common/model.h"
#include "common/settings.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

using Json = nlohmann::json;

namespace {
using xinfer::require;

int64_t proc_bytes(const char *path, const char *field) {
  std::ifstream file(path);
  require(file.is_open(), std::string("Cannot read memory statistics: ") + path);
  for (std::string line; std::getline(file, line);) {
    std::istringstream parts(line);
    std::string key, unit;
    int64_t value = 0;
    if (parts >> key && key == std::string(field) + ":") {
      require(bool(parts >> value >> unit) && value >= 0 && unit == "kB",
              "Invalid process memory statistics");
      return value * 1024;
    }
  }
  throw std::runtime_error(std::string("Missing memory statistic: ") + field);
}

Json memory(const char *phase) {
  torch::cuda::synchronize();
  size_t available = 0, total = 0;
  const auto error = cudaMemGetInfo(&available, &total);
  require(error == cudaSuccess, cudaGetErrorString(error));
  const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
  const auto aggregate = static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
  return {{"phase", phase},
          {"cuda_free_bytes", available},
          {"cuda_total_bytes", total},
          {"cuda_used_bytes", total - available},
          {"aten_allocated_bytes", stats.allocated_bytes[aggregate].current},
          {"aten_reserved_bytes", stats.reserved_bytes[aggregate].current},
          {"aten_peak_allocated_bytes", stats.allocated_bytes[aggregate].peak},
          {"aten_peak_reserved_bytes", stats.reserved_bytes[aggregate].peak},
          {"vm_rss_bytes", proc_bytes("/proc/self/status", "VmRSS")},
          {"vm_hwm_bytes", proc_bytes("/proc/self/status", "VmHWM")},
          {"system_mem_available_bytes", proc_bytes("/proc/meminfo", "MemAvailable")}};
}

template <class Spec>
void measure(const std::filesystem::path &packages, const Spec &spec, Json &report) {
  using Runtime = xinfer::Runner<Spec>;
  xinfer::configure_runtime();
  xinfer::checked_device(spec);
  // Establish the context/GEMM baseline before attributing allocations to weights.
  {
    const auto options = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
    auto warmup = at::mm(at::ones({16, 16}, options), at::ones({16, 16}, options));
    torch::cuda::synchronize();
  }
  c10::cuda::CUDACachingAllocator::resetPeakStats(0);
  auto sample = [&](const char *phase) {
    auto row = memory(phase);
    report["samples"].push_back(row);
    std::cout << row.dump() << std::endl;
  };
  sample("before_runtime");
  auto runtime = std::make_unique<Runtime>(packages, spec, sample);
  // Both calls to SharedWeights::load checked every borrowed constant pointer,
  // dtype, shape, stride and device against this one retained canonical pool.
  report["loader_pairs"] = 1;
  report["shared_pool_alias_checks_passed"] = true;
  sample("runtime_ready");
  auto logits = runtime->step({42, 13, 42, 13, 42, 13, 42, 13});
  require(at::isfinite(logits).all().template item<bool>(), "Nonfinite prefill logits");
  sample("after_first_prefill");
  logits = runtime->step({42});
  require(at::isfinite(logits).all().template item<bool>(), "Nonfinite decode logits");
  sample("after_first_decode");
  require(runtime->position() == 9, "Incorrect position after prefill and decode");
  report["position"] = runtime->position();
  report["execution_passed"] = true;
  logits = at::Tensor();
  runtime.reset();
  sample("after_runtime_destroyed");
  uint64_t sampled_peak = 0;
  for (const auto &row : report["samples"])
    sampled_peak = std::max(sampled_peak, row.at("cuda_used_bytes").get<uint64_t>());
  report["boundary_sampled_cuda_peak_used_bytes"] = sampled_peak;
  const auto before = report["samples"].front().at("aten_allocated_bytes").get<int64_t>();
  const auto after = report["samples"].back().at("aten_allocated_bytes").get<int64_t>();
  report["post_destroy_aten_allocated_growth_bytes"] = after - before;
  report["returned_to_allocator_baseline"] = after <= before;
  report["passed"] = report["returned_to_allocator_baseline"];
}
} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 4, "Usage: qwen_weight_memory qwen3|qwen35|qwen36 PACKAGE_DIR REPORT_JSON");
    c10::InferenceMode inference;
    const std::string model(argv[1]);
    Json report = {
        {"model", model},
        {"packages", std::filesystem::absolute(argv[2]).string()},
        {"torch", TORCH_VERSION},
        {"measurement", "Synchronized native load/inference boundary samples"},
        {"scope", "One shared pool; first prefill/decode only; no accuracy or timing acceptance"},
        {"memory_caveat",
         "CUDA, allocator and RSS values overlap on UMA; do not add them. Boundary-sampled CUDA "
         "use is not an exact cudaMalloc peak. Allocator peaks exclude raw AOTI allocations."},
        {"samples", Json::array()},
        {"execution_passed", false},
        {"passed", false}};
    xinfer::with_model(argv[2], [&](const auto &spec, auto) {
      require(model == xinfer::model_name(spec.model_type), "Model argument differs from package");
      report["cache_length"] = spec.capacity;
      measure(argv[2], spec, report);
    });
    auto file = xinfer::output_file(argv[3]);
    file << report.dump(2) << '\n';
    file.close();
    require(bool(file), "Failed to save memory report");
    std::cout << report.dump(2) << '\n';
    return report["passed"].get<bool>() ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
