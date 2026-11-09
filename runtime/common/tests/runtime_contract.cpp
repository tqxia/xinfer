// Regression for caller-independent Spec ownership and constructor preflight.
#include "common/model.h"
#include "common/settings.h"

#include <iostream>
#include <memory>
#include <type_traits>
#include <vector>

namespace {
using Json = nlohmann::json;
using xinfer::require;
using Trace = std::vector<at::Tensor>;

template <class Spec> void mutate(Spec &spec) {
  spec.max_prefill = spec.capacity = spec.vocab = 0;
  spec.eos = {-1};
  spec.state.clear();
  spec.prefill_metadata = Json();
  spec.metadata = Json();
}

template <class Runtime> Trace trace(Runtime &runtime) {
  runtime.reset();
  Trace result;
  for (const auto &tokens : {std::vector<int64_t>{42, 13}, std::vector<int64_t>{42}}) {
    result.push_back(runtime.step(tokens).cpu());
    for (const auto &tensor : runtime.state())
      result.push_back(tensor.cpu());
  }
  require(runtime.position() == 3, "Incorrect position after prefill and decode");
  for (const auto &tensor : result)
    require(at::isfinite(tensor).all().template item<bool>(), "Non-finite runtime output");
  return result;
}

void compare(const Trace &actual, const Trace &expected) {
  require(actual.size() == expected.size(), "Runtime trace length differs");
  for (size_t i = 0; i < actual.size(); ++i)
    require(at::equal(actual[i], expected[i]), "Spec ownership changed logits or state");
}

template <class Spec, class Runtime>
void check_ownership(const std::filesystem::path &packages, const Spec &validated, Json &report) {
  Trace expected;
  std::vector<int64_t> expected_ids;
  {
    Spec caller(validated);
    // Library callers may start with unrelated process-wide settings.
    at::set_num_threads(2);
    at::globalContext().setAllowTF32CuBLAS(true);
    at::globalContext().setAllowTF32CuDNN(true);
    std::vector<std::string> phases;
    auto callback_owner = std::make_shared<int>(0);
    std::weak_ptr<int> callback_lifetime = callback_owner;
    Runtime runtime(packages, caller, [owner = callback_owner, &phases](const char *phase) {
      (void)owner;
      phases.emplace_back(phase);
    });
    callback_owner.reset();
    require(callback_lifetime.expired(), "Runtime retained its load observer");
    require(phases == std::vector<std::string>{"cpu_archive_validated",
                                               "canonical_pool_allocated_cpu_archive_alive",
                                               "loaders_constructed", "prefill_bound",
                                               "decode_bound"},
            "Runtime load phases differ");
    report["load_observer_released"] = true;
    report["load_phases"] = phases;
    require(at::get_num_threads() == 1 && !at::globalContext().allowTF32CuBLAS() &&
                !at::globalContext().allowTF32CuDNN(),
            "Runtime did not configure threads and TF32");
    report["runtime_settings_applied"] = true;
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

// File checks are fresh for every load, even when a caller reuses a Spec.
template <class Spec, class Runtime>
void check_file_preflight(const std::filesystem::path &packages, const Spec &validated,
                          const std::filesystem::path &output, Json &report) {
  namespace fs = std::filesystem;
  const auto fixture = output.parent_path() / (output.filename().string() + ".bundle");
  require(!fs::exists(fixture), "Preflight fixture already exists");
  fs::create_directories(fixture);
  struct Cleanup {
    fs::path directory;
    ~Cleanup() {
      std::error_code ignored;
      fs::remove_all(directory, ignored);
    }
  } cleanup{fixture};
  for (const auto *name : {"prefill.json", "decode.json", "prefill.pt2", "decode.pt2"})
    fs::copy_file(packages / name, fixture / name);
  // The large archive is immutable; only the copied packages are modified.
  fs::create_symlink(fs::absolute(packages / "weights.pt"), fixture / "weights.pt");
  const Spec reused(fixture);
  {
    auto verified = reused.validate_files(fixture);
    (void)verified;
  }
  auto corrupt = [&](const char *kind) {
    std::fstream file(fixture / (std::string(kind) + ".pt2"),
                      std::ios::in | std::ios::out | std::ios::binary);
    require(file.is_open(), "Cannot open preflight fixture");
    char byte = 0;
    file.read(&byte, 1);
    require(bool(file), "Cannot read preflight fixture");
    byte ^= 1;
    file.seekp(0);
    file.write(&byte, 1);
    file.close();
    require(bool(file), "Cannot corrupt preflight fixture");
  };
  auto reject = [&](const Spec &descriptor, const char *kind, const char *name) {
    bool rejected = false;
    try {
      Runtime runtime(fixture, descriptor);
    } catch (const std::runtime_error &error) {
      rejected = std::string(error.what()) == std::string("Package checksum mismatch: ") + kind;
    }
    require(rejected, std::string("File preflight was bypassed: ") + name);
    report["cases"].push_back({{"name", name}, {"passed", true}});
  };
  corrupt("prefill");
  reject(reused, "prefill", "changed_after_spec");
  reject(validated, "prefill", "different_package_directory");
  fs::copy_file(packages / "prefill.pt2", fixture / "prefill.pt2",
                fs::copy_options::overwrite_existing);
  corrupt("decode");
  reject(reused, "decode", "reused_spec_after_verification");
}

template <class Spec, class Runtime>
void check_gpu_preflight(const std::filesystem::path &packages, const Spec &validated) {
  auto wrong_gpu = validated;
  require(wrong_gpu.compute_capability.size() == 2, "Invalid test compute capability");
  wrong_gpu.compute_capability[0] = 99;
  // Loading this descriptor would fail. The architecture error must win,
  // proving that rejection occurs before weight construction/package loading.
  wrong_gpu.shared_weights = Json();
  bool rejected = false;
  try {
    Runtime runtime(packages, wrong_gpu);
  } catch (const std::runtime_error &error) {
    rejected = std::string(error.what()) == "GPU architecture mismatch";
  }
  require(rejected, "GPU architecture was not checked before loading weights");
}
} // namespace

int main(int argc, char **argv) {
  try {
    require(
        argc == 4 || (argc == 5 && std::string(argv[4]) == "--files-only"),
        "Usage: qwen_runtime_contract qwen3|qwen35|qwen36 PACKAGE_DIR OUTPUT_JSON [--files-only]");
    const bool files_only = argc == 5;
    c10::InferenceMode inference;
    const std::string model(argv[1]);
    Json report = {{"model", model},
                   {"packages", std::filesystem::absolute(argv[2]).string()},
                   {"torch", TORCH_VERSION},
                   {"cases", Json::array()},
                   {"passed", false}};
    xinfer::with_model(argv[2], [&](const auto &spec, auto) {
      using Spec = std::decay_t<decltype(spec)>;
      using Runtime = xinfer::Runner<Spec>;
      require(model == xinfer::model_name(spec.model_type), "Model argument differs from package");
      if (files_only) {
        check_file_preflight<Spec, Runtime>(argv[2], spec, argv[3], report);
      } else {
        check_gpu_preflight<Spec, Runtime>(argv[2], spec);
        report["gpu_guard_before_weights"] = true;
        check_ownership<Spec, Runtime>(argv[2], spec, report);
      }
    });
    report["passed"] = true;
    auto file = xinfer::output_file(argv[3]);
    file << report.dump(2) << '\n';
    file.close();
    require(bool(file), "Failed to write runtime contract report");
    std::cout << report.dump(2) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
