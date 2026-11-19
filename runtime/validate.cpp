// Shared native validator with adapters for the retained trace file layouts.
#include "common/model.h"

#include <array>
#include <iostream>
#include <unordered_set>

using namespace xinfer;

namespace {
struct Options {
  fs::path packages, cases, output;
};

Options parse_options(int argc, char **argv) {
  require(argc == 4, "Usage: xinfer_validate PACKAGE_DIR CASES_JSON OUTPUT_DIR");
  return {argv[1], argv[2], argv[3]};
}

struct TraceLayout {
  bool qwen3, selected_states, compact;
};

struct Case {
  std::string name;
  bool greedy, dump_cache, check_overflow;
  std::vector<int64_t> prompt, eos;
  std::vector<std::vector<int64_t>> steps;
  std::vector<std::vector<std::vector<int64_t>>> batch_steps;
  int64_t limit = 0;
  double fill = 0;
  bool override_eos = false;
  std::vector<int64_t> state_steps;
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
      if (item.contains("batch_steps")) {
        require(!item.contains("chunks") && !item.contains("steps"),
                "Expected one validation input form");
        c.batch_steps = item.at("batch_steps").get<decltype(c.batch_steps)>();
        for (const auto &rows : c.batch_steps) {
          spec.check_batch_tokens(rows);
          c.steps.push_back(rows.front());
        }
      } else {
        require(item.contains("chunks") != item.contains("steps"),
                "Expected validation chunks or steps");
        c.steps = item.at(item.contains("steps") ? "steps" : "chunks")
                      .get<std::vector<std::vector<int64_t>>>();
      }
      require(!c.steps.empty(), "Expected validation steps");

      std::vector<int64_t> positions(spec.max_batch_size, 0);
      for (size_t step = 0; step < c.steps.size(); ++step) {
        const auto rows = c.batch_steps.empty() ? std::vector<std::vector<int64_t>>{c.steps[step]}
                                                : c.batch_steps[step];
        spec.check_batch_tokens(rows);
        for (size_t row = 0; row < rows.size(); ++row) {
          require(static_cast<int64_t>(rows[row].size()) <= spec.capacity - positions[row],
                  "Context capacity exceeded");
          positions[row] += rows[row].size();
          require(positions[row] > 0, "Each row must start with tokens");
        }
      }
      require(!c.check_overflow ||
                  *std::max_element(positions.begin(), positions.end()) == spec.capacity,
              "Overflow case must fill the context");

      if (manifest.value("trace_format", 1) == 2) {
        c.state_steps = item.value("state_steps", std::vector<int64_t>{});
        int64_t previous = -1;
        for (auto step : c.state_steps) {
          require(step > previous && step < static_cast<int64_t>(c.steps.size()),
                  "Invalid state checkpoint indices");
          previous = step;
        }
      }
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
    for (auto step : c.state_steps)
      require(!fs::exists(output / (c.name + "." + std::to_string(step) + ".states")),
              "Output already exists");

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

template <class Runtime>
void write_state(std::ofstream &file, const Runtime &runtime, bool compact, double fill) {
  const auto &state = runtime.state();
  for (size_t i = 0; i < state.size(); ++i) {
    const auto &tensor = state[i];
    require(at::isfinite(tensor).all().template item<bool>(), "Non-finite state");

    if (compact && runtime.spec().state[i].inplace) {
      const auto future = tensor.slice(2, runtime.position());
      const auto expected = at::full_like(future, fill);
      require(at::equal(future.contiguous().view(at::kByte), expected.contiguous().view(at::kByte)),
              "Future KV overwritten");
      write_tensor(file, tensor.slice(2, 0, runtime.position()));
    } else {
      write_tensor(file, tensor);
    }
  }
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

template <class Runtime>
void check_persistent(const Runtime &runtime, const std::vector<const void *> &persistent) {
  for (size_t i = 0; i < persistent.size(); ++i)
    if (runtime.spec().state[i].inplace)
      require(runtime.state()[i].const_data_ptr() == persistent[i], "KV cache storage changed");
}

template <class Runtime> void check_reset(const Runtime &runtime, double fill) {
  for (size_t i = 0; i < runtime.spec().state.size(); ++i) {
    const double expected = runtime.spec().state[i].inplace ? fill : 0;
    require(runtime.state()[i].eq(expected).all().template item<bool>(),
            "State reset left stale data");
  }
}

void check_state_update(const StateSpec &spec, const at::Tensor &input, const at::Tensor &before,
                        const at::Tensor &after, int64_t start, int64_t end, bool qwen3) {
  if (spec.inplace) {
    require(input.const_data_ptr() == after.const_data_ptr() && input.strides() == after.strides(),
            "KV output lost caller-owned storage");
    require(at::equal(before.slice(2, 0, start), after.slice(2, 0, start)), "Past KV overwritten");
    require(at::equal(before.slice(2, end), after.slice(2, end)), "Future KV overwritten");
    if (qwen3 && end > start)
      require(!at::equal(before.slice(2, start, end), after.slice(2, start, end)),
              "Cache update missing");
  } else {
    require(at::equal(input, before), "Functional input state mutated");
  }

  require(at::isfinite(after).all().item<bool>(), "Non-finite state");
}

template <class Runtime> void check_overflow(Runtime &runtime) {
  const auto capacity = runtime.spec().capacity;
  require(runtime.position() == capacity, "Context not full");

  const auto inputs = runtime.state();
  const auto before = snapshot(runtime);
  const auto positions = runtime.positions();
  bool rejected = false;
  try {
    runtime.step_batch(std::vector<std::vector<int64_t>>(runtime.spec().max_batch_size, {1}));
  } catch (const std::runtime_error &error) {
    rejected = std::string(error.what()) == "Context capacity exceeded";
  }

  require(rejected && runtime.position() == capacity && runtime.positions() == positions,
          "Overflow not rejected");
  for (size_t i = 0; i < before.size(); ++i)
    require(inputs[i].const_data_ptr() == runtime.state()[i].const_data_ptr() &&
                at::equal(before[i], runtime.state()[i]),
            "Overflow changed state");
}

template <class Runtime>
void run_greedy_case(Runtime &runtime, const Case &c, const TraceLayout &layout,
                     std::ofstream &logits, std::ofstream &states, Json &row) {
  const auto ids = c.override_eos ? xinfer::generate(runtime, c.prompt, c.limit, c.eos, &logits)
                                  : runtime.generate(c.prompt, c.limit, &logits);
  row["ids"] = ids;
  row["rows"] = ids.size();
  row["stop"] = std::find(c.eos.begin(), c.eos.end(), ids.back()) != c.eos.end() ? "eos" : "limit";

  if (!layout.qwen3)
    write_state(states, runtime, layout.compact, 0);
}

template <class Runtime>
void run_trace_case(Runtime &runtime, const Case &c, const TraceLayout &layout,
                    const fs::path &output, const std::vector<const void *> &persistent,
                    std::ofstream &logits, std::ofstream &states, Json &row) {
  runtime.reset(c.fill);
  check_reset(runtime, c.fill);

  for (size_t step = 0; step < c.steps.size(); ++step) {
    const auto &tokens = c.steps[step];
    const auto inputs = runtime.state();
    const auto before = snapshot(runtime);
    const auto starts = runtime.positions();
    const auto rows =
        c.batch_steps.empty() ? std::vector<std::vector<int64_t>>{tokens} : c.batch_steps[step];

    const auto score =
        c.batch_steps.empty() ? runtime.step(tokens) : runtime.step_batch(c.batch_steps[step]);
    require(at::isfinite(score).all().template item<bool>(), "Non-finite logits");
    for (size_t row = 0; row < rows.size(); ++row)
      require(runtime.positions()[row] == starts[row] + static_cast<int64_t>(rows[row].size()),
              "Incorrect position advance");
    for (size_t row = rows.size(); row < starts.size(); ++row)
      require(runtime.positions()[row] == starts[row], "Unused row position changed");

    if (layout.qwen3)
      write_logits(logits, score);
    else
      write_tensor(states, score);

    require(runtime.state().size() == runtime.spec().state.size(), "Incorrect state count");
    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto &after = runtime.state()[i];
      for (size_t row = 0; row < starts.size(); ++row)
        check_state_update(runtime.spec().state[i], inputs[i].narrow(0, row, 1),
                           before[i].narrow(0, row, 1), after.narrow(0, row, 1), starts[row],
                           runtime.positions()[row], layout.qwen3);
      if (!layout.qwen3 && !layout.selected_states)
        write_tensor(states, after);
    }
    check_persistent(runtime, persistent);

    if (layout.selected_states &&
        std::find(c.state_steps.begin(), c.state_steps.end(), step) != c.state_steps.end()) {
      auto checkpoint =
          output_file(output / (c.name + "." + std::to_string(step) + ".states"), true);
      write_state(checkpoint, runtime, layout.compact, c.fill);
      finish(checkpoint, "Failed to write state checkpoint");
    }
  }

  row["rows"] = c.steps.size();
  row["steps"] = c.steps.size();
  row["write_region_checks"] = row["reset_check"] = true;
  row["kv_inputs_updated_in_place"] = row["non_kv_inputs_unchanged"] =
      row["kv_write_regions_preserved"] = true;

  if (layout.qwen3 && c.dump_cache && !layout.selected_states) {
    auto cache = output_file(output / (c.name + ".kv.bf16"), true);
    for (const auto &tensor : runtime.state())
      write_tensor(cache, tensor);
    finish(cache, "Failed to write cache dump");
  }
}

template <class Runtime>
Json run_case(Runtime &runtime, const Case &c, const TraceLayout &layout, const fs::path &output,
              const std::vector<const void *> &persistent) {
  Json row = {{"name", c.name}};
  if (layout.selected_states) {
    row["state_steps"] = c.state_steps;
    row["state_prefix"] = layout.compact;
  }

  std::ofstream logits, states;
  if (layout.qwen3 || c.greedy)
    logits = output_file(output / (c.name + ".f32"), true);
  if (!layout.qwen3)
    states = output_file(output / (c.name + ".states"), true);

  if (c.greedy)
    run_greedy_case(runtime, c, layout, logits, states, row);
  else
    run_trace_case(runtime, c, layout, output, persistent, logits, states, row);

  if (c.check_overflow) {
    check_overflow(runtime);
    row["overflow_rejected"] = row["overflow_rejected_without_mutation"] = true;
  }

  finish(logits, "Failed to write logits trace");
  finish(states, "Failed to write state dump");
  check_persistent(runtime, persistent);

  row["position"] = runtime.position();
  row["positions"] = runtime.positions();
  if (layout.qwen3)
    row["stable_storage"] = true;
  row["stable_kv_storage"] = true;
  row["prefill_calls"] = runtime.prefill_calls();
  row["decode_calls"] = runtime.decode_calls();
  return row;
}

void write_report(const fs::path &output, const Json &report) {
  auto file = output_file(output / "native.json");
  file << report.dump(2) << '\n';
  finish(file, "Failed to write native validation report");
}

void compare_files(const fs::path &actual, const fs::path &expected) {
  std::ifstream a(actual, std::ios::binary), b(expected, std::ios::binary);
  require(a.is_open() && b.is_open(), "Cannot open parity traces: " + actual.string());
  std::array<char, 65536> left{}, right{};
  for (;;) {
    a.read(left.data(), left.size());
    b.read(right.data(), right.size());
    require((a.good() || a.eof()) && (b.good() || b.eof()), "Parity trace read failed");
    const auto size = a.gcount();
    require(size == b.gcount() && std::equal(left.begin(), left.begin() + size, right.begin()),
            "Native bitwise mismatch: " + actual.string());
    if (size == 0)
      break;
  }
}

void check_parity(const Case &c, const fs::path &output, const fs::path &expected, Json &row) {
  compare_files(output / (c.name + ".f32"), expected / (c.name + ".f32"));
  for (auto step : c.state_steps) {
    const auto name = c.name + "." + std::to_string(step) + ".states";
    compare_files(output / name, expected / name);
  }
  row["bitwise_parity"] = true;
  // Keep failed traces; discard large state dumps only after exact comparisons.
  for (auto step : c.state_steps) {
    const auto name = c.name + "." + std::to_string(step) + ".states";
    fs::remove(output / name);
    fs::remove(expected / name);
  }
}

void run_validation(const Options &options) {
  const auto manifest = read_json(options.cases);
  const auto trace_format = manifest.value("trace_format", 1);
  require(trace_format == 1 || trace_format == 2, "Unsupported validation trace format");
  const bool selected_states = trace_format == 2;
  const fs::path expected = manifest.value("expected_directory", std::string());

  with_model(options.packages, [&](const auto &spec, auto make_runtime) {
    const auto cases = parse_cases(manifest, spec, options.output);
    const TraceLayout layout{spec.model_type == ModelType::Qwen3, selected_states,
                             selected_states && spec.capacity > 1024};
    require(expected.empty() || (layout.qwen3 && selected_states),
            "Expected traces require Qwen3 trace format 2");

    c10::InferenceMode inference;
    auto runtime = make_runtime();
    fs::create_directories(options.output);

    std::vector<const void *> persistent;
    for (const auto &tensor : runtime->state())
      persistent.push_back(tensor.const_data_ptr());

    Json report = {
        {"cases", Json::array()}, {"loader_pairs", 1}, {"state_tensors", spec.state.size()}};
    if (layout.qwen3)
      report["cache_buffers"] = persistent.size();

    for (const auto &c : cases) {
      auto row = run_case(*runtime, c, layout, options.output, persistent);
      if (!expected.empty())
        check_parity(c, options.output, expected, row);
      report["cases"].push_back(std::move(row));
      if (!expected.empty())
        write_report(options.output, report);
      std::cerr << c.name << ": position " << runtime->position() << '\n';
    }

    check_native_libraries(report);
    write_report(options.output, report);
  });
}

} // namespace

int main(int argc, char **argv) {
  try {
    run_validation(parse_options(argc, argv));
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
