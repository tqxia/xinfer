// Regression for caller-independent Spec ownership and constructor preflight.
#include "qwen3/runtime.h"
#include "qwen35/runtime.h"

#include <iostream>
#include <memory>
#include <vector>

namespace {
using Json = nlohmann::json;
using xinfer::require;
using Trace = std::vector<at::Tensor>;

const std::vector<at::Tensor>& state(const qwen::Runtime& runtime) { return runtime.cache(); }
const std::vector<at::Tensor>& state(const qwen35::Runtime& runtime) { return runtime.state(); }

template <class Spec>
void mutate(Spec& spec) {
  spec.max_prompt = spec.capacity = spec.vocab = 0;
  spec.eos = {-1};
  spec.state.clear();
  spec.prefill_metadata = Json();
  spec.metadata = Json();
}

template <class Runtime>
Trace trace(Runtime& runtime) {
  runtime.reset();
  Trace result;
  for (const auto& tokens : {std::vector<int64_t>{42, 13}, std::vector<int64_t>{42}}) {
    result.push_back(runtime.step(tokens).cpu());
    for (const auto& tensor : state(runtime)) result.push_back(tensor.cpu());
  }
  require(runtime.position() == 3, "Incorrect position after prefill and decode");
  for (const auto& tensor : result)
    require(at::isfinite(tensor).all().template item<bool>(), "Non-finite runtime output");
  return result;
}

void compare(const Trace& actual, const Trace& expected) {
  require(actual.size() == expected.size(), "Runtime trace length differs");
  for (size_t i = 0; i < actual.size(); ++i)
    require(at::equal(actual[i], expected[i]), "Spec ownership changed logits or state");
}

template <class Spec, class Runtime>
void check_ownership(const std::filesystem::path& packages, const Spec& validated, Json& report) {
  Trace expected;
  std::vector<int64_t> expected_ids;
  {
    Spec caller(validated);
    Runtime runtime(packages, caller);
    expected = trace(runtime);
    expected_ids = runtime.generate({42, 13}, 3, nullptr);
    // This fails deterministically with the old borrowed Spec, before the
    // temporary/scoped cases below can encounter a dangling reference.
    mutate(caller);
    compare(trace(runtime), expected);
    require(runtime.generate({42, 13}, 3, nullptr) == expected_ids,
            "Caller Spec mutation changed generation");
    report["cases"].push_back({{"name", "caller_spec_mutation"}, {"passed", true}});
  }
  {
    Runtime runtime(packages, Spec(validated));
    compare(trace(runtime), expected);
    require(runtime.generate({42, 13}, 3, nullptr) == expected_ids,
            "Temporary Spec changed generation");
    report["cases"].push_back({{"name", "temporary_spec"}, {"passed", true}});
  }
  {
    auto runtime = [&] {
      Spec local(validated);
      return std::make_unique<Runtime>(packages, local);
    }();
    compare(trace(*runtime), expected);
    require(runtime->generate({42, 13}, 3, nullptr) == expected_ids,
            "Scoped Spec changed generation");
    report["cases"].push_back({{"name", "scoped_spec_destroyed"}, {"passed", true}});
  }
  report["trace_tensors_per_case"] = expected.size();
  report["generated_ids"] = expected_ids;
}

void check_gpu_preflight(const std::filesystem::path& packages, const qwen::Spec& validated) {
  auto wrong_gpu = validated;
  require(wrong_gpu.compute_capability.size() == 2, "Invalid test compute capability");
  wrong_gpu.compute_capability[0] = 99;
  // Loading this descriptor would fail. The architecture error must win,
  // proving that rejection occurs before weight construction/package loading.
  wrong_gpu.shared_weights = Json();
  bool rejected = false;
  try { qwen::Runtime runtime(packages, wrong_gpu); }
  catch (const std::runtime_error& error) {
    rejected = std::string(error.what()) == "GPU architecture mismatch";
  }
  require(rejected, "GPU architecture was not checked before loading weights");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 4, "Usage: qwen_runtime_contract qwen3|qwen35 PACKAGE_DIR OUTPUT_JSON");
    c10::InferenceMode inference;
    at::set_num_threads(1);
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    const std::string model(argv[1]);
    Json report = {{"model", model}, {"packages", std::filesystem::absolute(argv[2]).string()},
                   {"torch", TORCH_VERSION}, {"cases", Json::array()}, {"passed", false}};
    if (model == "qwen3") {
      const qwen::Spec spec(argv[2]);
      check_gpu_preflight(argv[2], spec);
      report["gpu_guard_before_weights"] = true;
      check_ownership<qwen::Spec, qwen::Runtime>(argv[2], spec, report);
    } else if (model == "qwen35") {
      const qwen35::Spec spec(argv[2]);
      qwen35::configure(spec);
      check_ownership<qwen35::Spec, qwen35::Runtime>(argv[2], spec, report);
    } else {
      throw std::runtime_error("Unknown model: " + model);
    }
    report["passed"] = true;
    auto file = xinfer::output_file(argv[3]);
    file << report.dump(2) << '\n';
    file.close();
    require(bool(file), "Failed to write runtime contract report");
    std::cout << report.dump(2) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
