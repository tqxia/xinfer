// CPU-only checks of complete-prompt partitioning and generation boundaries.
#include "common/generation.h"
#include "common/model.h"

#include <chrono>
#include <functional>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using xinfer::Json;
using xinfer::require;

// Sidecars exercise the production request checks without loading GPU weights.
struct Bundle {
  xinfer::fs::path directory;

  explicit Bundle(int64_t capacity = 16384, std::string model_type = "qwen3") {
    directory = xinfer::fs::temp_directory_path() /
                ("xinfer-generation-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    require(xinfer::fs::create_directory(directory), "Cannot create test metadata directory");

    const Json state = Json::array({
        {{"layer", 0},
         {"kind", "key"},
         {"shape", {8, 8, capacity, 128}},
         {"dtype", "bfloat16"},
         {"update", "inplace"}},
        {{"layer", 0},
         {"kind", "value"},
         {"shape", {8, 8, capacity, 128}},
         {"dtype", "bfloat16"},
         {"update", "inplace"}},
    });
    Json metadata = {
        {"model_type", model_type},
        {"torch_version", TORCH_VERSION},
        {"cxx11_abi", bool(_GLIBCXX_USE_CXX11_ABI)},
        {"cuda_compute_capability", {8, 9}},
        {"cache_length", capacity},
        {"max_prefill_tokens", 512},
        {"vocab_size", 16},
        {"eos_token_ids", {1}},
        {"checkpoint", {{"model", "test"}, {"revision", "test"}}},
        {"state", state},
        {"entries",
         {{"prefill", {{"bytes", 1}, {"sha256", std::string(64, 'a')}}},
          {"decode", {{"bytes", 1}, {"sha256", std::string(64, 'a')}}}}},
        {"shared_weights",
         {{"identity", std::string(64, 'b')}, {"bytes", 1}, {"sha256", std::string(64, 'c')}}},
    };

    auto file = xinfer::output_file(directory / xinfer::kMetadataFile);
    file << metadata.dump();
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
    spec.check_batch_tokens({tokens});
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

    check("qwen38_identity_uses_shared_qwen35_runtime", [&] {
      for (const std::string type : {"qwen35", "qwen38"}) {
        Bundle hybrid(512, type);
        xinfer::with_model(hybrid.directory, [&](const auto &selected, auto create) {
          require((std::is_same_v<std::decay_t<decltype(selected)>, qwen35::Spec>),
                  "Hybrid package selected another spec implementation");
          require((std::is_same_v<typename decltype(create())::element_type, qwen35::Runtime>),
                  "Hybrid package selected another runtime implementation");
          require(xinfer::model_name(selected.model_type) == type, "Package identity was lost");
          require(xinfer::parse_model_type(type) == selected.model_type,
                  "Model type did not roundtrip");
          const Json identity = {{"model", type == "qwen38" ? "qwen3.8" : "qwen3.5"}};
          require(xinfer::model_from_identity(identity) == selected.model_type,
                  "Weight identity selected another model type");
          for (size_t i = 0; i < selected.state.size(); ++i)
            require(selected.state[i].output_index == i + 1, "Hybrid state output index missing");
        });
        const Json other_identity = {{"model", type == "qwen38" ? "qwen3.5" : "qwen3.8"}};
        require(xinfer::model_from_identity(other_identity) != xinfer::parse_model_type(type),
                "Qwen3.5 and Qwen3.8 weight identities collided");
      }
    });

    check("context_contract_rejects_retired_prompt_field", [&] {
      require(spec.capacity == 16384 && spec.max_prefill == 512,
              "Compiled context and per-call bounds differ from the metadata");
      const auto metadata = xinfer::read_json(bundle.directory / xinfer::kMetadataFile);
      xinfer::validate_metadata(metadata);

      auto small = metadata;
      small["cache_length"] = 511;
      bool rejected = false;
      try {
        xinfer::validate_metadata(small);
      } catch (const std::runtime_error &error) {
        rejected = std::string(error.what()) == "Unsupported package metadata: cache_length";
      }
      require(rejected, "Sub-512 capacity was not rejected as a capacity error");

      small["cache_length"] = 512;
      for (auto &state : small["state"])
        state["shape"][2] = 512;
      xinfer::validate_metadata(small);

      auto too_large = small;
      too_large["cache_length"] = xinfer::kMaxCacheLength + 1;
      for (auto &state : too_large["state"])
        state["shape"][2] = xinfer::kMaxCacheLength + 1;
      rejected = false;
      try {
        xinfer::validate_metadata(too_large);
      } catch (const std::runtime_error &error) {
        rejected = std::string(error.what()) == "Unsupported package metadata: cache_length";
      }
      require(rejected, "Over-int32-safe capacity was not rejected as a capacity error");

      too_large["cache_length"] = xinfer::kMaxCacheLength;
      for (auto &state : too_large["state"])
        state["shape"][2] = xinfer::kMaxCacheLength;
      xinfer::validate_metadata(too_large);

      auto bad_prefill = metadata;
      bad_prefill["max_prefill_tokens"] = 1024;
      rejected = false;
      try {
        xinfer::validate_metadata(bad_prefill);
      } catch (const std::runtime_error &error) {
        rejected = std::string(error.what()) == "Unsupported package metadata: max_prefill_tokens";
      }
      require(rejected, "Unsupported max_prefill_tokens was accepted");

      for (const auto *field : {"max_prompt_length", "entry_point", "token_range", "export_options",
                                "tf32", "triton_enable_fp_fusion", "inductor_configs",
                                "package_bytes", "package_sha256", "model_specific"}) {
        auto retired_field = metadata;
        retired_field[field] = 1;
        rejected = false;
        try {
          xinfer::validate_metadata(retired_field);
        } catch (const std::runtime_error &error) {
          rejected =
              std::string(error.what()) == std::string("Unsupported package metadata: ") + field;
        }
        require(rejected,
                std::string("A retired field was accepted as package metadata: ") + field);
      }
    });

    check("independent_row_admission", [&] {
      const qwen::Spec batch_spec(bundle.directory);
      for (int64_t batch = 1; batch <= 8; ++batch) {
        std::vector<std::vector<int64_t>> rows(batch, {2});
        rows[0] = {2, 3};
        require(batch_spec.max_batch_size == 8 && batch_spec.check_batch_tokens(rows) == 2,
                "Per-row lengths not accepted");
        for (int64_t active = 0; active < batch; ++active) {
          rows.assign(batch, {});
          rows[active] = {4};
          require(batch_spec.check_batch_tokens(rows) == 1, "Inactive rows not accepted");
        }
        for (int variant = 0; variant < 5; ++variant) {
          rows.assign(batch, {2});
          if (variant == 0)
            rows.assign(9, {2});
          if (variant == 1)
            rows.assign(batch, {});
          if (variant == 2)
            rows.back() = {16};
          if (variant == 3)
            rows.back().assign(513, 3);
          if (variant == 4)
            rows.clear();
          bool rejected = false;
          try {
            batch_spec.check_batch_tokens(rows);
          } catch (const std::runtime_error &) {
            rejected = true;
          }
          require(rejected, "Invalid batch admitted");
        }
      }
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

    check("safetensors_shared_weights_cpu_load_and_rejection", [&] {
      Bundle weights_bundle;
      const auto write_safetensors = [](const xinfer::fs::path &path, const Json &header,
                                        const std::vector<uint8_t> &payload) {
        const std::string header_text = header.dump();
        const uint64_t length = header_text.size();
        uint8_t prefix[8]{};
        for (int i = 0; i < 8; ++i)
          prefix[i] = static_cast<uint8_t>((length >> (8 * i)) & 0xffu);
        auto file = xinfer::output_file(path, true);
        file.write(reinterpret_cast<const char *>(prefix), 8);
        file.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
        if (!payload.empty())
          file.write(reinterpret_cast<const char *>(payload.data()),
                     static_cast<std::streamsize>(payload.size()));
        file.close();
        require(bool(file), "Failed to write test safetensors file");
      };

      const std::string identity(64, 'b');
      Json meta = xinfer::read_json(weights_bundle.directory / xinfer::kMetadataFile);
      for (const auto *kind : {"prefill", "decode"}) {
        const auto pkg = weights_bundle.directory / (std::string(kind) + ".pt2");
        auto file = xinfer::output_file(pkg, true);
        file << kind;
        file.close();
        meta["entries"][kind]["bytes"] = xinfer::fs::file_size(pkg);
        meta["entries"][kind]["sha256"] = xinfer::sha256_file(pkg);
      }

      require(xinfer::shared_weights_dtype("float8_e4m3fn") == at::kFloat8_e4m3fn &&
                  std::string(xinfer::safetensors_dtype_name(at::kFloat8_e4m3fn)) == "F8_E4M3",
              "Unexpected float8_e4m3fn safetensors dtype mapping");

      // Pack U8 (1) + I64 (16 at odd offset 1) + BF16 (12) + F8_E4M3 (6) + empty F32.
      const std::vector<uint8_t> payload(35, 0x2a);
      const Json base_manifest = {
          {"identity", {{"kind", "cpu_test"}}},
          {"identity_sha256", identity},
          {"entries",
           {{"prefill",
             {{"package_sha256", meta["entries"]["prefill"]["sha256"]},
              {"constants",
               {{"c_u8", "v_u8"}, {"c_i64", "v_i64"}, {"c_bf16", "v_bf16"}, {"c_fp8", "v_fp8"}}}}},
            {"decode",
             {{"package_sha256", meta["entries"]["decode"]["sha256"]},
              {"constants",
               {{"c_bf16_alias", "v_bf16_slice"},
                {"c_fp8_alias", "v_fp8_slice"},
                {"c_empty", "v_empty"}}}}}}},
          {"sources", Json::object()},
          {"storages",
           {{"w_u8", {{"dtype", "uint8"}, {"device", "cpu"}, {"bytes", 1}}},
            {"w_i64", {{"dtype", "int64"}, {"device", "cpu"}, {"bytes", 16}}},
            {"w_bf16", {{"dtype", "bfloat16"}, {"device", "cpu"}, {"bytes", 12}}},
            {"w_fp8", {{"dtype", "float8_e4m3fn"}, {"device", "cpu"}, {"bytes", 6}}},
            {"w_empty", {{"dtype", "float32"}, {"device", "cpu"}, {"bytes", 0}}}}},
          {"tensors",
           {{"v_u8",
             {{"storage", "w_u8"},
              {"dtype", "uint8"},
              {"device", "cpu"},
              {"shape", {1}},
              {"stride", {1}},
              {"offset", 0}}},
            {"v_i64",
             {{"storage", "w_i64"},
              {"dtype", "int64"},
              {"device", "cpu"},
              {"shape", {2}},
              {"stride", {1}},
              {"offset", 0}}},
            {"v_bf16",
             {{"storage", "w_bf16"},
              {"dtype", "bfloat16"},
              {"device", "cpu"},
              {"shape", {2, 3}},
              {"stride", {3, 1}},
              {"offset", 0}}},
            {"v_bf16_slice",
             {{"storage", "w_bf16"},
              {"dtype", "bfloat16"},
              {"device", "cpu"},
              {"shape", {2, 2}},
              {"stride", {2, 1}},
              {"offset", 1}}},
            {"v_fp8",
             {{"storage", "w_fp8"},
              {"dtype", "float8_e4m3fn"},
              {"device", "cpu"},
              {"shape", {2, 3}},
              {"stride", {3, 1}},
              {"offset", 0}}},
            {"v_fp8_slice",
             {{"storage", "w_fp8"},
              {"dtype", "float8_e4m3fn"},
              {"device", "cpu"},
              {"shape", {2, 2}},
              {"stride", {2, 1}},
              {"offset", 1}}},
            {"v_empty",
             {{"storage", "w_empty"},
              {"dtype", "float32"},
              {"device", "cpu"},
              {"shape", {0, 4}},
              {"stride", {4, 1}},
              {"offset", 0}}}}},
      };
      const Json base_header = {
          {"__metadata__", {{xinfer::kSharedWeightsManifestKey, base_manifest.dump()}}},
          {"w_u8", {{"dtype", "U8"}, {"shape", {1}}, {"data_offsets", {0, 1}}}},
          {"w_i64", {{"dtype", "I64"}, {"shape", {2}}, {"data_offsets", {1, 17}}}},
          {"w_bf16", {{"dtype", "BF16"}, {"shape", {6}}, {"data_offsets", {17, 29}}}},
          {"w_fp8", {{"dtype", "F8_E4M3"}, {"shape", {6}}, {"data_offsets", {29, 35}}}},
          {"w_empty", {{"dtype", "F32"}, {"shape", {0}}, {"data_offsets", {35, 35}}}},
      };

      const auto archive = weights_bundle.directory / xinfer::kSharedWeightsFile;
      auto update_contract = [&] {
        const Json contract = {
            {"identity", identity},
            {"bytes", xinfer::fs::file_size(archive)},
            {"sha256", xinfer::sha256_file(archive)},
        };
        meta["shared_weights"] = contract;
        return contract;
      };

      write_safetensors(archive, base_header, payload);
      const auto contract = update_contract();
      {
        auto verified = xinfer::ValidatedBundle::verify(weights_bundle.directory, meta);
        xinfer::SharedWeights loaded(std::move(verified), contract);
        require(loaded.package_path("prefill").filename() == "prefill.pt2" &&
                    loaded.package_path("decode").filename() == "decode.pt2",
                "Validated bundle returned unexpected package paths");
      }

      auto expect_error = [&](const std::function<void()> &operation, const std::string &expected) {
        bool matched = false;
        try {
          operation();
        } catch (const std::runtime_error &error) {
          matched = std::string(error.what()).find(expected) != std::string::npos;
        }
        require(matched, "Expected error containing: " + expected);
      };

      // Legacy file and retired format fields must be rejected.
      {
        auto legacy = meta;
        legacy["shared_weights"]["format"] = 2;
        legacy["shared_weights"]["file"] = "weights.pt";
        expect_error([&] { xinfer::validate_metadata(legacy); },
                     "Unsupported package metadata: shared_weights");
        expect_error([&] { xinfer::ValidatedBundle::verify(weights_bundle.directory, legacy); },
                     "Shared weights metadata is required");

        auto retired = meta;
        retired["shared_weights"]["file"] = xinfer::kSharedWeightsFile;
        expect_error([&] { xinfer::validate_metadata(retired); },
                     "Unsupported package metadata: shared_weights");
        expect_error([&] { xinfer::ValidatedBundle::verify(weights_bundle.directory, retired); },
                     "Shared weights metadata is required");
      }

      // Retired format key inside embedded manifest must be rejected during preflight.
      {
        Json bad_manifest = base_manifest;
        bad_manifest["format"] = 3;
        Json header = base_header;
        header["__metadata__"][xinfer::kSharedWeightsManifestKey] = bad_manifest.dump();
        write_safetensors(archive, header, payload);
        update_contract();
        expect_error([&] { xinfer::ValidatedBundle::verify(weights_bundle.directory, meta); },
                     "Shared weights archive metadata is invalid");
      }

      // Missing __metadata__ in safetensors header must be rejected during preflight.
      {
        Json header = base_header;
        header.erase("__metadata__");
        write_safetensors(archive, header, payload);
        update_contract();
        expect_error([&] { xinfer::ValidatedBundle::verify(weights_bundle.directory, meta); },
                     "Shared weights archive metadata is invalid");
      }

      // Truncated safetensors file must be rejected during preflight.
      {
        auto file = xinfer::output_file(archive, true);
        file.write("bad", 3);
        file.close();
        update_contract();
        expect_error([&] { xinfer::ValidatedBundle::verify(weights_bundle.directory, meta); },
                     "Shared weights archive metadata is invalid");
      }

      // Extra tensor in safetensors header must be rejected on load.
      {
        Json header = base_header;
        header["w_extra"] = {{"dtype", "U8"}, {"shape", {0}}, {"data_offsets", {35, 35}}};
        write_safetensors(archive, header, payload);
        const auto bad_contract = update_contract();
        expect_error(
            [&] {
              auto verified = xinfer::ValidatedBundle::verify(weights_bundle.directory, meta);
              xinfer::SharedWeights loaded(std::move(verified), bad_contract);
            },
            "Shared weights archive buffer set mismatch");
      }

      // Out-of-bounds data_offsets must be rejected on load.
      {
        Json header = base_header;
        header["w_fp8"]["data_offsets"] = {30, 36};
        write_safetensors(archive, header, payload);
        const auto bad_contract = update_contract();
        expect_error(
            [&] {
              auto verified = xinfer::ValidatedBundle::verify(weights_bundle.directory, meta);
              xinfer::SharedWeights loaded(std::move(verified), bad_contract);
            },
            "Shared weights archive tensor metadata mismatch");
      }

      // Header dtype differing from manifest dtype (including FP8 tags) must be rejected on load.
      for (const auto &[tensor_name, wrong_dtype] :
           std::vector<std::pair<std::string, std::string>>{
               {"w_bf16", "F16"}, {"w_fp8", "U8"}, {"w_fp8", "F8_E5M2"}}) {
        Json header = base_header;
        header[tensor_name]["dtype"] = wrong_dtype;
        write_safetensors(archive, header, payload);
        const auto bad_contract = update_contract();
        expect_error(
            [&] {
              auto verified = xinfer::ValidatedBundle::verify(weights_bundle.directory, meta);
              xinfer::SharedWeights loaded(std::move(verified), bad_contract);
            },
            "Shared weights archive tensor metadata mismatch");
      }

      expect_error([&] { (void)xinfer::shared_weights_dtype("float8_e5m2"); },
                   "Shared weights dtype is unsupported");

      // Out-of-bounds view into storage must be rejected before allocation.
      {
        Json bad_manifest = base_manifest;
        bad_manifest["tensors"]["v_bf16_slice"]["offset"] = 3;
        Json header = base_header;
        header["__metadata__"][xinfer::kSharedWeightsManifestKey] = bad_manifest.dump();
        write_safetensors(archive, header, payload);
        const auto bad_contract = update_contract();
        expect_error(
            [&] {
              auto verified = xinfer::ValidatedBundle::verify(weights_bundle.directory, meta);
              xinfer::SharedWeights loaded(std::move(verified), bad_contract);
            },
            "Shared weights tensor layout is invalid");
      }
    });

    check("sha256_matches_nist_vectors_and_scalar_fallback", [&] {
      Bundle temp_bundle;
      const auto hash_bytes = [&](const std::string &name, const std::string &bytes) {
        const auto path = temp_bundle.directory / name;
        auto file = xinfer::output_file(path, true);
        if (!bytes.empty())
          file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        file.close();
        require(bool(file), "Failed to write SHA-256 fixture");
        return xinfer::sha256_file(path);
      };

      require(hash_bytes("empty.bin", "") ==
                  "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
              "SHA-256 mismatch on empty input");
      require(hash_bytes("abc.bin", "abc") ==
                  "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
              "SHA-256 mismatch on 'abc'");
      require(hash_bytes("56.bin", "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
                  "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
              "SHA-256 mismatch on 56-byte NIST vector");
      require(hash_bytes("million.bin", std::string(1000000, 'a')) ==
                  "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
              "SHA-256 mismatch on 1,000,000 'a' bytes");

      // Exercise exact 256 KiB buffer boundary and 64-byte block boundaries against scalar
      // transform.
      for (const size_t length : {size_t{55}, size_t{63}, size_t{64}, size_t{65}, size_t{262144},
                                  size_t{262144 + 55}, size_t{262144 + 63}}) {
        std::string data(length, '\0');
        for (size_t i = 0; i < length; ++i)
          data[i] = static_cast<char>((i * 131 + 17) & 0xff);
        const auto actual = hash_bytes("len_" + std::to_string(length) + ".bin", data);

        std::array<uint32_t, 8> scalar_state{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                             0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
        std::vector<unsigned char> padded(data.begin(), data.end());
        padded.push_back(0x80);
        while (padded.size() % 64 != 56)
          padded.push_back(0);
        const uint64_t bits = uint64_t{length} * 8;
        for (int shift = 56; shift >= 0; shift -= 8)
          padded.push_back(static_cast<unsigned char>((bits >> shift) & 0xffu));
        xinfer::detail::sha256_transform_scalar(scalar_state, padded.data(), padded.size() / 64);

        std::ostringstream expected;
        expected << std::hex << std::setfill('0');
        for (auto word : scalar_state)
          expected << std::setw(8) << word;
        require(actual == expected.str(),
                "Hardware/buffered SHA-256 diverged from scalar transform at length " +
                    std::to_string(length));
      }
    });

    check("spec_checks_torch_cuda_suffix_and_dangling_symlinks", [&] {
      Bundle version_bundle;
      const auto write_version = [&](const std::string &version) {
        auto meta = xinfer::read_json(version_bundle.directory / xinfer::kMetadataFile);
        meta["torch_version"] = version;
        auto file = xinfer::output_file(version_bundle.directory / xinfer::kMetadataFile);
        file << meta.dump();
      };

      const long cudart = at::Context::versionCUDART();
      const std::string cuda_tag =
          "cu" + std::to_string(cudart / 1000) + std::to_string((cudart % 1000) / 10);
      for (const auto &accepted :
           {std::string(TORCH_VERSION), std::string(TORCH_VERSION) + "+" + cuda_tag,
            std::string(TORCH_VERSION) + "+gitabcdef"}) {
        write_version(accepted);
        const xinfer::SpecBase ok(version_bundle.directory);
        require(ok.capacity == 16384, "Accepted torch_version failed to load SpecBase");
      }

      for (const auto &bad :
           {std::string(TORCH_VERSION) + "+cu999", std::string(TORCH_VERSION) + "+cpu",
            std::string(TORCH_VERSION) + "+", std::string("0.0.0+") + cuda_tag}) {
        write_version(bad);
        bool rejected = false;
        try {
          const xinfer::SpecBase fail(version_bundle.directory);
          (void)fail;
        } catch (const std::runtime_error &error) {
          rejected = std::string(error.what()) == "Torch version mismatch";
        }
        require(rejected, "Mismatched torch_version was accepted: " + bad);
      }

      const auto missing = version_bundle.directory / "nonexistent_target";
      const auto dangling = version_bundle.directory / "dangling_output";
      xinfer::fs::create_symlink(missing, dangling);
      require(!xinfer::path_exists(missing), "Missing path reported as existing");
      require(!xinfer::fs::exists(dangling) && xinfer::path_exists(dangling),
              "Dangling symlink was not detected by path_exists");
    });

    check("qwen3_spec_accepts_general_kv_shapes_and_size_free_identity", [&] {
      require(xinfer::model_from_identity(Json{{"model", "qwen3"}}) == xinfer::ModelType::Qwen3,
              "qwen3 identity was not accepted");
      require(xinfer::model_from_identity(Json{{"model", "qwen3.5"}}) == xinfer::ModelType::Qwen35,
              "qwen3.5 identity was not accepted");
      bool rejected = false;
      try {
        (void)xinfer::model_from_identity(Json{{"model", "qwen3-0.6b"}});
      } catch (const std::runtime_error &) {
        rejected = true;
      }
      require(rejected, "Size-qualified qwen3-0.6b identity was still accepted");

      Bundle geometry_bundle;
      for (const auto &shape :
           std::vector<Json>{{8, 4, 16384, 128}, {8, 8, 16384, 64}, {8, 8, 16384, 256}}) {
        auto meta = xinfer::read_json(geometry_bundle.directory / xinfer::kMetadataFile);
        for (auto &item : meta["state"])
          item["shape"] = shape;
        auto file = xinfer::output_file(geometry_bundle.directory / xinfer::kMetadataFile);
        file << meta.dump();
        file.close();
        const qwen::Spec ok(geometry_bundle.directory);
        require(ok.state.size() == 2 && ok.state[0].shape == ok.state[1].shape,
                "Alternate Qwen3 KV geometry failed to load");
      }

      {
        auto meta = xinfer::read_json(geometry_bundle.directory / xinfer::kMetadataFile);
        meta["state"][0]["shape"] = {8, 4, 16384, 128};
        meta["state"][1]["shape"] = {8, 8, 16384, 128};
        auto file = xinfer::output_file(geometry_bundle.directory / xinfer::kMetadataFile);
        file << meta.dump();
      }
      rejected = false;
      try {
        const qwen::Spec mismatch(geometry_bundle.directory);
        (void)mismatch;
      } catch (const std::runtime_error &error) {
        rejected = std::string(error.what()) == "K/V shapes differ";
      }
      require(rejected, "Mismatched Qwen3 K/V shapes were accepted");
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
