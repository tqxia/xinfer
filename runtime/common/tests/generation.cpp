// CPU-only checks of complete-prompt partitioning and generation boundaries.
#include "common/generation.h"
#include "common/model.h"

#include <chrono>
#include <functional>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

namespace {
using xinfer::Json;
using xinfer::require;

// Sidecars exercise the production request checks without loading GPU weights.
struct Bundle {
  xinfer::fs::path directory;

  explicit Bundle(int64_t capacity = 16384) {
    directory = xinfer::fs::temp_directory_path() /
                ("xinfer-generation-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    require(xinfer::fs::create_directory(directory), "Cannot create test metadata directory");
    const Json state = Json::array({
        {{"layer", 0},
         {"kind", "key"},
         {"shape", {1, 8, capacity, 128}},
         {"dtype", "bfloat16"},
         {"update", "inplace"}},
        {{"layer", 0},
         {"kind", "value"},
         {"shape", {1, 8, capacity, 128}},
         {"dtype", "bfloat16"},
         {"update", "inplace"}},
    });
    Json metadata = {
        {"schema_version", 5},
        {"runtime_abi_version", 1},
        {"model_type", "qwen3"},
        {"torch_version", TORCH_VERSION},
        {"cxx11_abi", bool(_GLIBCXX_USE_CXX11_ABI)},
        {"cuda_compute_capability", {8, 9}},
        {"cache_length", capacity},
        {"vocab_size", 16},
        {"eos_token_ids", {1}},
        {"checkpoint", {{"model_id", "test"}, {"revision", "test"}, {"manifest", Json::object()}}},
        {"state", state},
        {"export_options", {{"backed_size_oblivious", false}}},
        {"tf32", false},
        {"triton_enable_fp_fusion", true},
        {"inductor_configs",
         {{"emulate_precision_casts", true},
          {"aot_inductor.package_constants_in_so", false},
          {"always_keep_tensor_constants", true}}},
        {"package_bytes", 1},
        {"package_sha256", std::string(64, 'a')},
        {"shared_weights",
         {{"format", 2},
          {"file", "weights.pt"},
          {"identity", std::string(64, 'b')},
          {"bytes", 1},
          {"sha256", std::string(64, 'c')}}},
        {"model_specific", Json::object()},
    };
    for (const auto *kind : {"prefill", "decode"}) {
      metadata["entry_point"] = kind;
      metadata["token_range"] = std::string(kind) == "prefill" ? Json{2, 512} : Json{1, 1};
      auto file = xinfer::output_file(directory / (std::string(kind) + ".json"));
      file << metadata.dump();
    }
  }

  ~Bundle() {
    std::error_code ignored;
    xinfer::fs::remove_all(directory, ignored);
  }
};

struct Logits {
  at::Tensor tensor;
  std::vector<int64_t> *selections = nullptr;
  int64_t position = 0;

  at::Tensor argmax(int64_t dim) const {
    selections->push_back(position);
    return tensor.argmax(dim);
  }
  operator const at::Tensor &() const { return tensor; }
};

struct Runtime {
  explicit Runtime(const xinfer::SpecBase &contract) : spec(contract) {}

  void reset() {
    ++resets;
    position = prefill_calls = decode_calls = 0;
    calls.clear();
    consumed.clear();
    selections.clear();
  }

  Logits step(const std::vector<int64_t> &tokens) {
    spec.check_step_tokens(tokens);
    require(static_cast<int64_t>(tokens.size()) <= spec.capacity - position,
            "Context capacity exceeded");
    calls.emplace_back(position, tokens.size());
    consumed.insert(consumed.end(), tokens.begin(), tokens.end());
    position += static_cast<int64_t>(tokens.size());
    ++(tokens.size() == 1 ? decode_calls : prefill_calls);
    auto logits = at::zeros({1, spec.vocab});
    logits.select(0, 0).select(0, position % 7).fill_(1);
    return {logits, &selections, position};
  }

  Logits prefill(const std::vector<int64_t> &prompt) {
    spec.check_tokens(prompt);
    require(static_cast<int64_t>(prompt.size()) <= spec.capacity - position,
            "Context capacity exceeded");
    return xinfer::prefill(*this, prompt, spec.max_prefill);
  }

  std::vector<int64_t> generate(const std::vector<int64_t> &prompt, int64_t limit,
                                std::ostream *trace = nullptr) {
    spec.check_request(prompt, limit);
    return xinfer::generate(*this, prompt, limit, spec.eos, trace);
  }

  const xinfer::SpecBase &spec;
  int64_t position = 0, prefill_calls = 0, decode_calls = 0, resets = 0;
  std::vector<std::pair<int64_t, int64_t>> calls;
  std::vector<int64_t> consumed, selections;
};

void reject_without_mutation(Runtime &runtime, const std::function<void()> &operation) {
  const auto position = runtime.position, resets = runtime.resets;
  const auto calls = runtime.calls;
  const auto consumed = runtime.consumed, selections = runtime.selections;
  bool rejected = false;
  try {
    operation();
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  require(rejected, "Invalid request was accepted");
  require(runtime.position == position && runtime.resets == resets && runtime.calls == calls &&
              runtime.consumed == consumed && runtime.selections == selections,
          "Rejected request changed state or executed a chunk");
}
} // namespace

int main() {
  try {
    c10::InferenceMode inference;
    Bundle bundle;
    const xinfer::SpecBase spec(bundle.directory);
    Runtime runtime(spec);
    Json report = {{"device", "cpu"}, {"cases", Json::array()}, {"passed", false}};
    auto check = [&](const char *name, const std::function<void()> &operation) {
      try {
        operation();
      } catch (const std::exception &error) {
        throw std::runtime_error(std::string(name) + ": " + error.what());
      }
      report["cases"].push_back({{"name", name}, {"passed", true}});
    };

    check("qwen36_returns_all_hybrid_state_outputs", [&] {
      Bundle fixture(1024);
      for (const auto *kind : {"prefill", "decode"}) {
        const auto path = fixture.directory / (std::string(kind) + ".json");
        auto metadata = xinfer::read_json(path);
        metadata["model_type"] = "qwen36";
        metadata["checkpoint"] = {{"model_id", "Qwen/Qwen3.6-35B-A3B-FP8"},
                                  {"revision", "c653bba46dfa06e9658dcc0b0dc87fe4685f14a2"},
                                  {"manifest", Json::object()}};
        metadata["vocab_size"] = 248320;
        metadata["eos_token_ids"] = {248046, 248044};
        metadata["model_specific"] = {{"delta_backend", "triton_op"},
                                      {"attention_backend", "flash_attention"},
                                      {"fp8_backend", "triton_op"},
                                      {"fp8_format", "e4m3fn"},
                                      {"fp8_block_size", {128, 128}},
                                      {"fp8_scale_dtype", "bfloat16"},
                                      {"fp8_kernel",
                                       {{"repo", "kernels-community/finegrained-fp8"},
                                        {"revision", "03ae31933313f38fae2b050dbe2731024386c369"}}},
                                      {"linear_packing", {{"enabled", false}}}};
        metadata["state"] = Json::array();
        for (int64_t layer = 0; layer < 40; ++layer) {
          const bool attention = layer % 4 == 3;
          for (int index = 0; index < 2; ++index) {
            const auto name =
                attention ? (index ? "value" : "key") : (index ? "recurrent" : "conv");
            const Json shape = attention ? Json::array({1, 2, 1024, 256})
                               : index   ? Json::array({1, 32, 128, 128})
                                         : Json::array({1, 8192, 4});
            metadata["state"].push_back({{"layer", layer},
                                         {"kind", name},
                                         {"shape", shape},
                                         {"dtype", !attention && index ? "float32" : "bfloat16"},
                                         {"update", attention ? "inplace" : "replace"}});
          }
        }
        xinfer::output_file(path) << metadata.dump();
      }
      const qwen36::Spec spec(fixture.directory);
      require(spec.state.size() == 80 &&
                  xinfer::model_name(spec.model_type) == std::string("qwen36"),
              "Qwen3.6 model dispatch or state count differs");
      for (size_t i = 0; i < spec.state.size(); ++i)
        require(spec.state[i].output_index == i + 1, "Qwen3.6 returned-state output is missing");
      require(xinfer::model_from_identity(Json{{"model", "qwen3.6"}}) == spec.model_type,
              "Qwen3.6 archive identity differs");
    });

    check("schema5_context_contract_rejects_retired_prompt_field", [&] {
      require(spec.capacity == 16384 && spec.max_prefill == 512,
              "Compiled context and per-call bounds differ from the metadata");
      const auto metadata = xinfer::read_json(bundle.directory / "prefill.json");
      xinfer::validate_metadata(metadata, "prefill");
      for (const auto *kind : {"prefill", "decode"}) {
        auto small = xinfer::read_json(bundle.directory / (std::string(kind) + ".json"));
        small["cache_length"] = 511;
        bool rejected = false;
        try {
          xinfer::validate_metadata(small, kind);
        } catch (const std::runtime_error &error) {
          rejected = std::string(error.what()) == "Unsupported package metadata: cache_length";
        }
        require(rejected, "Sub-512 capacity was not rejected as a capacity error");
        small["cache_length"] = 512;
        for (auto &state : small["state"])
          state["shape"][2] = 512;
        xinfer::validate_metadata(small, kind);
      }
      for (const auto version : {3, 4}) {
        auto old = metadata;
        old["schema_version"] = version;
        bool rejected = false;
        try {
          xinfer::validate_metadata(old, "prefill");
        } catch (const std::runtime_error &error) {
          rejected =
              std::string(error.what()).find("Unsupported package metadata schema version") == 0;
        }
        require(rejected, "Retired metadata schema was accepted");
      }
      auto retired_field = metadata;
      retired_field["max_prompt_length"] = 8192;
      bool rejected = false;
      try {
        xinfer::validate_metadata(retired_field, "prefill");
      } catch (const std::runtime_error &error) {
        rejected = std::string(error.what()) == "Unsupported package metadata: max_prompt_length";
      }
      require(rejected, "A retired prompt field was accepted as package metadata");
    });

    check("default_requests_above_8192_and_exact_generation_budget", [&] {
      const std::vector<int64_t> prompt(8193, 2);
      spec.check_request(prompt, spec.capacity - prompt.size());
      runtime.generate(prompt, 1);
      require(runtime.position == 8193 && runtime.calls.size() == 17 &&
                  runtime.calls.back().second == 1,
              "Request validation retained an 8192-token cap");
      runtime.generate(std::vector<int64_t>(spec.capacity - 1, 2), 1);
      require(runtime.position == spec.capacity - 1,
              "Prompt plus one generated token did not fit the exact budget");
      reject_without_mutation(runtime,
                              [&] { runtime.generate(std::vector<int64_t>(spec.capacity, 2), 1); });
      // Even a prompt that predicts EOS immediately must reserve the full budget.
      reject_without_mutation(
          runtime, [&] { runtime.generate(std::vector<int64_t>(512, 2), spec.capacity); });
    });

    check("shared_model_api_checks_the_complete_context_before_loading", [&] {
      bool entered = false;
      xinfer::with_model(bundle.directory, [&](const auto &selected, auto) {
        entered = true;
        require(selected.capacity == spec.capacity && selected.max_prefill == 512,
                "Shared API changed compiled context or per-call bounds");
        selected.check_request(std::vector<int64_t>(8193, 2), 1);
        bool rejected = false;
        try {
          selected.check_request(std::vector<int64_t>(selected.capacity, 2), 1);
        } catch (const std::runtime_error &error) {
          rejected = std::string(error.what()).find("Generation budget") == 0;
        }
        require(rejected, "Over-budget request passed the pre-load descriptor check");
        // The factory is deliberately unused: these checks need no GPU or weights.
        return 0;
      });
      require(entered, "Shared API did not dispatch the model");
    });

    check("complete_request_and_compiled_call_bounds_are_independent", [&] {
      runtime.reset();
      spec.check_tokens(std::vector<int64_t>(513, 2));
      reject_without_mutation(runtime, [&] { runtime.step(std::vector<int64_t>(513, 2)); });
      runtime.prefill(std::vector<int64_t>(513, 2));
      require(runtime.position == 513 && runtime.calls.size() == 2,
              "A legal request was constrained to a single compiled call");
      runtime.step(std::vector<int64_t>(512, 2));
      require(runtime.position == 1025, "A legal appended compiled call was rejected");
      std::vector<int64_t> invalid(512, 2);
      invalid.back() = spec.vocab;
      reject_without_mutation(runtime, [&] { runtime.step(invalid); });
      reject_without_mutation(runtime,
                              [&] { runtime.prefill(std::vector<int64_t>(spec.capacity + 1, 2)); });
    });

    check("minimum_compiled_context_preserves_strict_output_budget", [&] {
      Bundle small_bundle(512);
      const xinfer::SpecBase small_spec(small_bundle.directory);
      Runtime small(small_spec);
      small.generate(std::vector<int64_t>(511, 2), 1);
      require(small.position == 511, "Exact minimum-context request budget was rejected");
      reject_without_mutation(small, [&] { small.generate(std::vector<int64_t>(511, 2), 2); });
      reject_without_mutation(small, [&] { small.generate(std::vector<int64_t>(512, 2), 1); });
      small.reset();
      small.prefill(std::vector<int64_t>(512, 2));
      require(small.position == small_spec.capacity && small.prefill_calls == 1,
              "Full-context prefill incorrectly reserved an output token");
    });

    check("request_partitions_and_singleton_dispatch", [&] {
      const std::vector<std::pair<int64_t, std::vector<int64_t>>> cases{
          {1, {1}},
          {511, {511}},
          {512, {512}},
          {513, {512, 1}},
          {8191, {512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 511}},
          {8192, {512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512}},
      };
      for (const auto &[length, sizes] : cases) {
        runtime.reset();
        std::vector<int64_t> prompt;
        for (int64_t i = 0; i < length; ++i)
          prompt.push_back(i % spec.vocab);
        runtime.prefill(prompt);
        require(runtime.consumed == prompt && runtime.position == length,
                "Prompt partition lost, reordered or repeated tokens");
        require(runtime.calls.size() == sizes.size(), "Incorrect compiled call count");
        int64_t position = 0, prefill = 0, decode = 0;
        for (size_t i = 0; i < sizes.size(); ++i) {
          require(runtime.calls[i] == std::pair(position, sizes[i]),
                  "Incorrect appended chunk position or length");
          position += sizes[i];
          ++(sizes[i] == 1 ? decode : prefill);
        }
        require(runtime.prefill_calls == prefill && runtime.decode_calls == decode,
                "Singleton prompt remainder did not use decode");
        require(runtime.selections.empty(), "Prefill selected an intermediate token");
      }
    });

    check("nonzero_offset_and_exact_context", [&] {
      runtime.reset();
      runtime.step({3, 4, 5});
      runtime.prefill(std::vector<int64_t>(513, 2));
      require(runtime.calls[1] == std::pair<int64_t, int64_t>(3, 512) &&
                  runtime.calls[2] == std::pair<int64_t, int64_t>(515, 1) &&
                  runtime.position == 516,
              "Chunking discarded the existing prefix");
      runtime.reset();
      const std::vector<int64_t> prompt(8192, 2);
      runtime.prefill(prompt);
      runtime.prefill(prompt);
      require(runtime.position == spec.capacity, "Exact context boundary was not reached");
      reject_without_mutation(runtime, [&] { runtime.step({1}); });
      reject_without_mutation(runtime, [&] { runtime.prefill({1, 2}); });
    });

    check("full_prompt_precedes_prediction_and_reset_reuse", [&] {
      const std::vector<int64_t> prompt(513, 2);
      std::ostringstream trace(std::ios::out | std::ios::binary);
      const auto ids = runtime.generate(prompt, 3, &trace);
      // The first 512-token chunk predicts EOS=1. It must not stop the request
      // before the singleton remainder, whose final logits instead predict 2.
      require(ids == std::vector<int64_t>({2, 3, 4}), "Generation used intermediate prompt logits");
      require(runtime.selections == std::vector<int64_t>({513, 514, 515}),
              "Generation performed an intermediate prompt argmax");
      require(trace.str().size() == ids.size() * spec.vocab * sizeof(float),
              "Trace contains intermediate prompt rows");
      const auto calls = runtime.calls;
      const auto resets = runtime.resets;
      require(runtime.generate(prompt, 3) == ids && runtime.calls == calls &&
                  runtime.resets == resets + 1,
              "Reset/reuse changed chunked generation");
      require(runtime.generate(std::vector<int64_t>(512, 2), 3) == std::vector<int64_t>{1} &&
                  runtime.position == 512,
              "EOS stopping performed an extra decode");
    });

    check("whole_request_validation_precedes_state_changes", [&] {
      runtime.reset();
      runtime.step({3, 4, 5});
      std::vector<int64_t> prompt(8192, 2);
      spec.check_request(prompt, 8192); // Exactly the advertised 16K budget.
      reject_without_mutation(runtime, [&] { runtime.generate(prompt, 8193); });
      reject_without_mutation(runtime, [&] { runtime.generate(prompt, 0); });
      reject_without_mutation(runtime, [&] { runtime.generate({}, 1); });
      reject_without_mutation(
          runtime, [&] { runtime.generate(std::vector<int64_t>(spec.capacity + 1, 2), 1); });
      prompt.back() = spec.vocab;
      reject_without_mutation(runtime, [&] { runtime.prefill(prompt); });
      reject_without_mutation(runtime, [&] { runtime.generate(prompt, 1); });
      reject_without_mutation(runtime, [&] { runtime.step(std::vector<int64_t>(513, 2)); });
      reject_without_mutation(runtime, [&] { xinfer::prefill(runtime, {1, 2}, 1); });
    });

    report["passed"] = true;
    report["case_count"] = report["cases"].size();
    std::cout << report.dump(2) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
