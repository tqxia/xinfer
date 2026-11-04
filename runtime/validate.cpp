// Shared native validator with adapters for the retained trace file layouts.
#include "common/model.h"
#include <iostream>
#include <unordered_set>

using namespace xinfer;

namespace {
struct Case {
  std::string name;
  bool greedy, dump_cache, check_overflow;
  std::vector<int64_t> prompt, eos;
  std::vector<std::vector<int64_t>> steps;
  int64_t limit = 0;
  double fill = 0;
  bool override_eos = false;
};

template <class Spec>
std::vector<Case> parse_cases(const Json &manifest, const Spec &spec, const fs::path &output) {
  require(!fs::exists(output) || fs::is_directory(output), "Output must be a directory");
  require(!fs::exists(output / "native.json"), "Output already exists");
  std::unordered_set<std::string> names;
  std::vector<Case> cases;
  for (const auto &item : manifest.at("cases")) {
    Case c;
    c.name = item.at("name").get<std::string>();
    require(!c.name.empty() && c.name != "." && c.name != ".." &&
                fs::path(c.name).filename() == c.name && names.insert(c.name).second,
            "Invalid or duplicate case name");
    c.greedy = item.contains("prompt");
    if (item.contains("mode")) {
      const auto mode = item.at("mode").get<std::string>();
      require(mode == "greedy" || mode == "trace", "Unknown validation mode");
      require((mode == "greedy") == c.greedy, "Validation mode disagrees with inputs");
    }
    c.dump_cache = item.value("dump_cache", false);
    c.check_overflow = item.value("check_overflow", false);
    c.fill = item.value("fill", item.value("poison_kv", 0.0));
    c.eos = item.value("eos", spec.eos);
    c.override_eos = item.contains("eos");
    require(!c.eos.empty(), "Missing EOS tokens");
    for (auto token : c.eos)
      require(token >= 0 && token < spec.vocab, "EOS token ID out of range");
    if (c.greedy) {
      c.prompt = item.at("prompt").get<std::vector<int64_t>>();
      c.limit = item.at("limit");
      spec.check_request(c.prompt, c.limit);
    } else {
      require(item.contains("chunks") != item.contains("steps"),
              "Expected validation chunks or steps");
      c.steps = item.at(item.contains("steps") ? "steps" : "chunks")
                    .get<std::vector<std::vector<int64_t>>>();
      require(!c.steps.empty(), "Expected validation steps");
      int64_t position = 0;
      for (const auto &tokens : c.steps) {
        spec.check_tokens(tokens);
        require(static_cast<int64_t>(tokens.size()) <= spec.capacity - position,
                "Context capacity exceeded");
        position += tokens.size();
      }
      require(!c.check_overflow || position == spec.capacity,
              "Overflow case must fill the context");
    }
    const bool qwen3 = spec.model_type == ModelType::Qwen3;
    std::vector<std::string> suffixes;
    if (qwen3 || c.greedy)
      suffixes.push_back(".f32");
    if (!qwen3)
      suffixes.push_back(".states");
    if (qwen3 && c.dump_cache)
      suffixes.push_back(".kv.bf16");
    for (const auto &suffix : suffixes)
      require(!fs::exists(output / (c.name + suffix)), "Output already exists");
    cases.push_back(std::move(c));
  }
  require(!cases.empty(), "Expected validation cases");
  return cases;
}

template <class Runtime> std::vector<at::Tensor> snapshot(const Runtime &runtime) {
  std::vector<at::Tensor> copies;
  for (const auto &tensor : runtime.state())
    copies.push_back(tensor.clone());
  return copies;
}

void finish(std::ofstream &file, const std::string &message) {
  if (!file.is_open())
    return;
  file.close();
  require(bool(file), message);
}

void check_native_libraries(Json &report) {
  std::ifstream maps("/proc/self/maps");
  require(maps.is_open(), "Cannot inspect process mappings");
  bool attention = false;
  for (std::string line; std::getline(maps, line);) {
    require(line.find("libpython") == std::string::npos &&
                line.find("libtorch_python") == std::string::npos,
            "Python library loaded");
    const bool expected = line.find("libxinfer_attention_ops.so") != std::string::npos;
    require(line.find("libxinfer_") == std::string::npos || expected,
            "Unexpected custom-op library loaded");
    attention = attention || expected;
  }
  require(attention, "Native attention library was not loaded");
  report["no_python_library"] = true;
  report["attention_library"] = true;
}
} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 4, "Usage: xinfer_validate PACKAGE_DIR CASES_JSON OUTPUT_DIR");
    const auto manifest = read_json(argv[2]);
    const fs::path output(argv[3]);
    return with_model(argv[1], [&](const auto &spec, auto make_runtime) {
      const auto cases = parse_cases(manifest, spec, output);
      const bool qwen3 = spec.model_type == ModelType::Qwen3;
      c10::InferenceMode inference;
      auto runtime = make_runtime();
      fs::create_directories(output);
      std::vector<const void *> persistent;
      for (const auto &tensor : runtime->state())
        persistent.push_back(tensor.const_data_ptr());
      const auto check_persistent = [&] {
        if (qwen3)
          for (size_t i = 0; i < persistent.size(); ++i)
            require(runtime->state()[i].const_data_ptr() == persistent[i],
                    "KV cache storage changed");
      };
      Json report = {
          {"cases", Json::array()}, {"loader_pairs", 1}, {"state_tensors", spec.state.size()}};
      if (qwen3)
        report["cache_buffers"] = persistent.size();
      for (const auto &c : cases) {
        Json row = {{"name", c.name}};
        std::ofstream logits, states;
        if (qwen3 || c.greedy)
          logits = output_file(output / (c.name + ".f32"), true);
        if (!qwen3)
          states = output_file(output / (c.name + ".states"), true);
        if (c.greedy) {
          const auto ids = c.override_eos
                               ? xinfer::generate(*runtime, c.prompt, c.limit, c.eos, &logits)
                               : runtime->generate(c.prompt, c.limit, &logits);
          row["ids"] = ids;
          row["rows"] = ids.size();
          row["stop"] =
              std::find(c.eos.begin(), c.eos.end(), ids.back()) != c.eos.end() ? "eos" : "limit";
          if (!qwen3)
            for (const auto &tensor : runtime->state())
              write_tensor(states, tensor);
        } else {
          runtime->reset(c.fill);
          for (size_t i = 0; i < spec.state.size(); ++i) {
            const double fill = spec.state[i].inplace ? c.fill : 0;
            require(runtime->state()[i].eq(fill).all().template item<bool>(),
                    "State reset left stale data");
          }
          for (const auto &tokens : c.steps) {
            const auto inputs = runtime->state();
            const auto before = snapshot(*runtime);
            const int64_t start = runtime->position(), end = start + tokens.size();
            const auto score = runtime->step(tokens);
            require(at::isfinite(score).all().template item<bool>(), "Non-finite logits");
            require(runtime->position() == end, "Incorrect position advance");
            if (qwen3)
              write_logits(logits, score);
            else
              write_tensor(states, score);
            require(runtime->state().size() == spec.state.size(), "Incorrect state count");
            for (size_t i = 0; i < inputs.size(); ++i) {
              const auto &after = runtime->state()[i];
              if (spec.state[i].inplace) {
                require(inputs[i].const_data_ptr() == after.const_data_ptr() &&
                            inputs[i].strides() == after.strides(),
                        "KV output lost caller-owned storage");
                require(at::equal(before[i].slice(2, 0, start), after.slice(2, 0, start)),
                        "Past KV overwritten");
                require(at::equal(before[i].slice(2, end), after.slice(2, end)),
                        "Future KV overwritten");
                if (qwen3)
                  require(!at::equal(before[i].slice(2, start, end), after.slice(2, start, end)),
                          "Cache update missing");
              } else {
                require(at::equal(inputs[i], before[i]), "Functional input state mutated");
              }
              require(at::isfinite(after).all().template item<bool>(), "Non-finite state");
              if (!qwen3)
                write_tensor(states, after);
            }
            check_persistent();
          }
          row["rows"] = c.steps.size();
          row["steps"] = c.steps.size();
          row["write_region_checks"] = row["reset_check"] = true;
          row["kv_inputs_updated_in_place"] = row["non_kv_inputs_unchanged"] =
              row["kv_write_regions_preserved"] = true;
          if (qwen3 && c.dump_cache) {
            auto cache = output_file(output / (c.name + ".kv.bf16"), true);
            for (const auto &tensor : runtime->state())
              write_tensor(cache, tensor);
            finish(cache, "Failed to write cache dump");
          }
        }
        if (c.check_overflow) {
          require(runtime->position() == spec.capacity, "Context not full");
          const auto inputs = runtime->state();
          const auto before = snapshot(*runtime);
          bool rejected = false;
          try {
            runtime->step({1});
          } catch (const std::runtime_error &error) {
            rejected = std::string(error.what()) == "Context capacity exceeded";
          }
          require(rejected && runtime->position() == spec.capacity, "Overflow not rejected");
          for (size_t i = 0; i < before.size(); ++i)
            require(inputs[i].const_data_ptr() == runtime->state()[i].const_data_ptr() &&
                        at::equal(before[i], runtime->state()[i]),
                    "Overflow changed state");
          row["overflow_rejected"] = row["overflow_rejected_without_mutation"] = true;
        }
        finish(logits, "Failed to write logits trace");
        finish(states, "Failed to write state dump");
        check_persistent();
        row["position"] = runtime->position();
        if (qwen3)
          row["stable_storage"] = true;
        row["stable_kv_storage"] = true;
        row["prefill_calls"] = runtime->prefill_calls();
        row["decode_calls"] = runtime->decode_calls();
        report["cases"].push_back(row);
        std::cerr << c.name << ": position " << runtime->position() << '\n';
      }
      check_native_libraries(report);
      auto file = output_file(output / "native.json");
      file << report.dump(2) << '\n';
      finish(file, "Failed to write native validation report");
      return 0;
    });
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
