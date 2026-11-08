// Exercise the shipping ABI through its ordinary shared-weight AOTI runner.
#include "common/settings.h"
#include "qwen3/runtime.h"

#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
using xinfer::Json;
using xinfer::require;
using Tensors = std::vector<at::Tensor>;
using Pointers = std::vector<uintptr_t>;

uintptr_t address(const at::Tensor &tensor) {
  return reinterpret_cast<uintptr_t>(tensor.const_data_ptr());
}

Pointers addresses(const Tensors &tensors) {
  Pointers result;
  for (const auto &tensor : tensors)
    result.push_back(address(tensor));
  return result;
}

bool bits_equal(const at::Tensor &left, const at::Tensor &right) {
  return left.sizes() == right.sizes() && left.scalar_type() == right.scalar_type() &&
         at::equal(left.contiguous().view(at::kByte), right.contiguous().view(at::kByte));
}

Tensors snapshot(const Tensors &tensors) {
  Tensors result;
  for (const auto &tensor : tensors)
    result.push_back(tensor.clone());
  return result;
}

struct Case {
  std::string name, same_as;
  std::vector<std::vector<int64_t>> steps;
  double fill;
  bool dump_cache, nondefault_stream, check_overflow;
};

std::vector<Case> read_cases(const xinfer::fs::path &manifest, const qwen::Spec &spec) {
  const auto data = xinfer::read_json(manifest);
  require(data.contains("cases") && data.at("cases").is_array() && !data.at("cases").empty(),
          "Missing production package cases");
  std::vector<Case> result;
  std::unordered_set<std::string> names;
  for (const auto &item : data.at("cases")) {
    Case test{item.at("name").get<std::string>(),
              item.value("same_as", std::string()),
              item.at("steps").get<std::vector<std::vector<int64_t>>>(),
              item.value("poison_kv", 0.0),
              item.value("dump_cache", false),
              item.value("nondefault_stream", false),
              item.value("check_overflow", false)};
    require(
        !test.name.empty() &&
            std::all_of(test.name.begin(), test.name.end(),
                        [](unsigned char c) { return std::isalnum(c) || c == '_' || c == '-'; }) &&
            names.insert(test.name).second,
        "Invalid or duplicate package case name");
    require(!test.steps.empty() && std::isfinite(test.fill), "Invalid package case");
    int64_t position = 0;
    for (const auto &tokens : test.steps) {
      spec.check_step_tokens(tokens);
      require(int64_t(tokens.size()) <= spec.capacity - position, "Case exceeds context");
      position += int64_t(tokens.size());
    }
    require(!test.check_overflow || position == spec.capacity, "Overflow case must fill context");
    require(test.same_as.empty() || (test.same_as != test.name && names.contains(test.same_as)),
            "Missing package replay control");
    result.push_back(std::move(test));
  }
  return result;
}

void libraries(Json &report, bool compiled_megaqwen) {
  std::ifstream maps("/proc/self/maps");
  require(maps.is_open(), "Cannot inspect native mappings");
  bool attention = false, megaqwen = false;
  for (std::string line; std::getline(maps, line);) {
    require(line.find("libpython") == std::string::npos &&
                line.find("libtorch_python") == std::string::npos,
            "Production package loaded Python");
    attention = attention || line.find("libxinfer_attention_ops.so") != std::string::npos;
    megaqwen = megaqwen || line.find("libxinfer_megaqwen.so") != std::string::npos;
  }
  require(attention && (!compiled_megaqwen || megaqwen), "Missing native package dependency");
  report["no_python_library"] = true;
}

void async_lifetime(const xinfer::fs::path &packages, const qwen::Spec &spec, const Case &test,
                    const at::Tensor &expected, Json &report) {
  const auto device = xinfer::configured_device(spec);
  const auto creation = c10::cuda::getCurrentCUDAStream(device.index());
  const auto selected = c10::cuda::getStreamFromPool(false, device.index());
  at::Tensor retained;
  {
    auto temporary = std::make_unique<qwen::Runtime>(packages, spec);
    temporary->reset(test.fill);
    for (size_t step = 0; step + 1 < test.steps.size(); ++step)
      temporary->step(test.steps[step]).cpu(); // Complete producers before changing streams.
    C10_CUDA_CHECK(cudaStreamSynchronize(creation.stream()));
    {
      c10::cuda::CUDAStreamGuard guard(selected);
      retained = temporary->step(test.steps.back());
    }
    temporary.reset();
  }
  Tensors churn;
  const auto options = expected.options().device(device);
  for (int repeat = 0; repeat < 10; ++repeat) {
    for (const auto &shape : std::vector<std::vector<int64_t>>{
             {1024}, {2048}, {3072}, {16, 4}, {16, 4}, {16, 4, 128}, {1024, 1024}})
      churn.push_back(at::full(shape, 7, options.dtype(at::kFloat)));
    churn.push_back(at::full({1024}, 7, options.dtype(at::kBFloat16)));
  }
  churn.push_back(at::full({1, 8, spec.capacity, 128}, 7, options));
  churn.push_back(at::full({spec.capacity, 128}, 7, options));
  C10_CUDA_CHECK(cudaStreamSynchronize(selected.stream()));
  require(bits_equal(retained.cpu(), expected), "Async destruction changed retained decode output");
  C10_CUDA_CHECK(cudaStreamSynchronize(creation.stream()));
  report["async_destruction_lifetime"] = true;
  report["async_destruction_case"] = test.name;
}

void validate(const xinfer::fs::path &packages, const xinfer::fs::path &manifest,
              const xinfer::fs::path &output, Json &report) {
  const qwen::Spec spec(packages);
  const auto cases = read_cases(manifest, spec);
  qwen::Runtime runtime(packages, spec);
  const auto device = runtime.state().front().device();
  c10::cuda::CUDAGuard device_guard(device);
  const auto persistent = addresses(runtime.state());
  report["capacity"] = spec.capacity;
  report["compiled_decode_backend"] = spec.compiled_megaqwen ? "megaqwen" : "aoti";
  report["caller_kv_pointers"] = persistent;
  std::unordered_map<std::string, at::Tensor> controls;
  std::unordered_map<std::string, Tensors> cache_controls;
  std::unordered_set<std::string> replay_controls;
  for (const auto &test : cases)
    if (!test.same_as.empty())
      replay_controls.insert(test.same_as);
  const Case *lifetime_case = nullptr;
  at::Tensor lifetime_expected;
  for (const auto &test : cases) {
    const auto current = c10::cuda::getCurrentCUDAStream(device.index());
    C10_CUDA_CHECK(cudaStreamSynchronize(current.stream()));
    const auto selected =
        test.nondefault_stream ? c10::cuda::getStreamFromPool(false, device.index()) : current;
    c10::cuda::CUDAStreamGuard stream_guard(selected);
    runtime.reset(test.fill);
    require(addresses(runtime.state()) == persistent, "Reset replaced caller KV storage");
    auto file = xinfer::output_file(output / (test.name + ".f32"), true);
    Tensors retained, expected;
    Json row{{"name", test.name}};
    for (const auto &tokens : test.steps) {
      const int64_t start = runtime.position(), end = start + int64_t(tokens.size());
      const auto before = snapshot(runtime.state());
      const auto score = runtime.step(tokens);
      require(at::isfinite(score).all().item<bool>(), "Nonfinite compiled logits");
      require(runtime.position() == end && addresses(runtime.state()) == persistent,
              "Compiled step changed position or KV ownership");
      for (size_t i = 0; i < before.size(); ++i) {
        const auto &after = runtime.state()[i];
        require(bits_equal(before[i].slice(2, 0, start), after.slice(2, 0, start)) &&
                    bits_equal(before[i].slice(2, end), after.slice(2, end)),
                "Compiled step overwrote past or future KV");
        require(!bits_equal(before[i].slice(2, start, end), after.slice(2, start, end)),
                "Compiled step omitted a layer's KV append");
        require(at::isfinite(after).all().item<bool>(), "Nonfinite compiled KV");
        const auto cache = address(after), output = address(score);
        require(output + score.nbytes() <= cache || cache + after.nbytes() <= output,
                "Compiled logits overlap caller-owned KV storage");
      }
      for (size_t i = 0; i < retained.size(); ++i)
        require(address(retained[i]) != address(score) &&
                    bits_equal(retained[i].cpu(), expected[i]),
                "A later call reused or changed retained logits");
      retained.push_back(score);
      expected.push_back(score.cpu().clone());
      xinfer::write_tensor(file, expected.back().to(at::kFloat));
    }
    file.close();
    require(bool(file), "Failed to write compiled package logits");
    if (!test.same_as.empty())
      require(controls.contains(test.same_as) &&
                  bits_equal(at::stack(expected), controls.at(test.same_as)),
              "Poison/reset changed compiled package logits");
    controls.emplace(test.name, at::stack(expected));
    if (!test.same_as.empty()) {
      require(cache_controls.contains(test.same_as), "Missing valid-KV replay control");
      const auto &original = cache_controls.at(test.same_as);
      for (size_t i = 0; i < original.size(); ++i)
        require(bits_equal(runtime.state()[i].slice(2, 0, runtime.position()).cpu(), original[i]),
                "Poison/reset changed valid compiled KV");
      row["poison_valid_kv_bitwise"] = true;
    }
    if (replay_controls.contains(test.name)) {
      Tensors valid;
      for (const auto &tensor : runtime.state())
        valid.push_back(tensor.slice(2, 0, runtime.position()).cpu().clone());
      cache_controls.emplace(test.name, std::move(valid));
    }
    if (test.dump_cache) {
      auto cache = xinfer::output_file(output / (test.name + ".kv.bf16"), true);
      for (const auto &tensor : runtime.state())
        xinfer::write_tensor(cache, tensor.slice(2, 0, runtime.position()));
      cache.close();
      require(bool(cache), "Failed to write compiled package KV");
    }
    if (test.check_overflow) {
      const auto before = snapshot(runtime.state());
      bool rejected = false;
      try {
        runtime.step({13});
      } catch (const std::runtime_error &error) {
        rejected = std::string(error.what()) == "Context capacity exceeded";
      }
      require(rejected && runtime.position() == spec.capacity &&
                  addresses(runtime.state()) == persistent,
              "Context overflow changed position or caller storage");
      for (size_t i = 0; i < before.size(); ++i)
        require(bits_equal(before[i], runtime.state()[i]), "Rejected overflow changed KV");
      row["overflow_rejected_without_mutation"] = true;
    }
    row["position"] = runtime.position();
    row["rows"] = test.steps.size();
    row["prefill_calls"] = runtime.prefill_calls();
    row["decode_calls"] = runtime.decode_calls();
    const auto singleton_calls =
        std::count_if(test.steps.begin(), test.steps.end(),
                      [](const auto &tokens) { return tokens.size() == 1; });
    require(runtime.decode_calls() == singleton_calls &&
                runtime.prefill_calls() == int64_t(test.steps.size()) - singleton_calls,
            "Compiled package prefill/decode counts differ from input lengths");
    row["caller_owned_storage_preserved"] = true;
    row["nondefault_stream"] = test.nondefault_stream;
    row["stream_id"] = selected.id();
    row["passed"] = true;
    report["cases"].push_back(row);
    C10_CUDA_CHECK(cudaStreamSynchronize(selected.stream()));
    if (test.nondefault_stream && test.steps.back().size() == 1) {
      lifetime_case = &test;
      lifetime_expected = expected.back();
    }
  }
  require(lifetime_case != nullptr, "Missing ordered-stream/lifetime case");
  async_lifetime(packages, spec, *lifetime_case, lifetime_expected, report);
  libraries(report, spec.compiled_megaqwen);
}
} // namespace

int main(int argc, char **argv) {
  Json report{{"passed", false}, {"benchmark", false}, {"cases", Json::array()}};
  xinfer::fs::path output;
  bool created = false;
  try {
    require(argc == 4, "Usage: qwen_megaqwen_package_contract PACKAGE_DIR CASES_JSON OUTPUT_DIR");
    output = argv[3];
    require(!xinfer::fs::exists(output), "Output directory already exists");
    created = xinfer::fs::create_directories(output);
    require(created, "Cannot create package contract directory");
    c10::InferenceMode inference;
    xinfer::configure_runtime();
    validate(argv[1], argv[2], output, report);
    report["passed"] = true;
  } catch (const std::exception &error) {
    report["error"] = error.what();
    std::cerr << error.what() << '\n';
  }
  if (created) {
    auto file = xinfer::output_file(output / "summary.json");
    file << report.dump(2) << '\n';
    file.close();
    if (!file)
      return 2;
  }
  return report["passed"].get<bool>() ? 0 : 1;
}
