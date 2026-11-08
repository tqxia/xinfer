// Correctness-only comparison of the MegaQwen dispatcher op and bare AOTI.
// The retained package ABI is unchanged: this probe owns the experimental router.
#include "common/settings.h"
#include "common/state.h"
#include "qwen3/runtime.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <caffe2/serialize/inline_container.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/version.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
using xinfer::Json;
using xinfer::require;
using Tensors = std::vector<at::Tensor>;
using Pointers = std::vector<uintptr_t>;
using Loader = torch::inductor::AOTIModelPackageLoader;

// Probe constructs its bundled AOTI loader before any dispatcher lookup.
at::Tensor call_megaqwen_decode(const at::Tensor &ids, at::TensorList weights, at::TensorList kv,
                                const at::Tensor &cos, const at::Tensor &sin,
                                at::TensorList workspace, const at::Tensor &extent) {
  using Signature =
      at::Tensor(const at::Tensor &, at::TensorList, at::TensorList, const at::Tensor &,
                 const at::Tensor &, at::TensorList, const at::Tensor &);
  const auto op = c10::Dispatcher::singleton()
                      .findSchemaOrThrow("xinfer::megaqwen_decode", "")
                      .typed<Signature>();
  return op.call(ids, weights, kv, cos, sin, workspace, extent);
}

bool bits_equal(const at::Tensor &left, const at::Tensor &right) {
  return left.sizes() == right.sizes() && left.scalar_type() == right.scalar_type() &&
         at::equal(left.contiguous().view(at::kByte), right.contiguous().view(at::kByte));
}

uintptr_t address(const at::Tensor &tensor) {
  return reinterpret_cast<uintptr_t>(tensor.const_data_ptr());
}

Pointers addresses(const Tensors &tensors) {
  Pointers result;
  for (const auto &tensor : tensors)
    result.push_back(address(tensor));
  return result;
}

void equal_tensors(const Tensors &actual, const Tensors &expected, const std::string &label) {
  require(actual.size() == expected.size(), label + ": tensor count differs");
  for (size_t i = 0; i < actual.size(); ++i)
    require(bits_equal(actual[i], expected[i]),
            label + ": tensor " + std::to_string(i) + " differs");
}

std::vector<std::string> weight_names() {
  std::vector<std::string> result{"model.model.embed_tokens.weight", "model.model.norm.weight",
                                  "model.lm_head.weight"};
  for (int layer = 0; layer < 28; ++layer) {
    const auto prefix = "model.model.layers." + std::to_string(layer) + ".";
    for (const auto *suffix :
         {"input_layernorm.weight", "self_attn.original.q_proj.weight",
          "self_attn.original.k_proj.weight", "self_attn.original.v_proj.weight",
          "self_attn.original.q_norm.weight", "self_attn.original.k_norm.weight",
          "self_attn.original.o_proj.weight", "post_attention_layernorm.weight",
          "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"})
      result.push_back(prefix + suffix);
  }
  return result;
}

// Borrow the prefill loader's verified canonical views for explicit-input tests.
Tensors canonical_weights(Loader &prefill, const xinfer::fs::path &packages) {
  caffe2::serialize::PyTorchStreamReader reader((packages / "weights.pt").string());
  const auto record = reader.getRecord("extra/weights.json");
  const auto *text = static_cast<const char *>(std::get<0>(record).get());
  const auto manifest = Json::parse(text, text + std::get<1>(record));
  const auto &mapping = manifest.at("entries").at("prefill").at("constants");
  std::unordered_map<std::string, at::Tensor> views;
  for (const auto &[name, tensor] : prefill.get_runner()->extract_constants_map(false))
    views.emplace(mapping.at(name).get<std::string>(), tensor);
  Tensors result;
  for (const auto &name : weight_names())
    result.push_back(views.at(manifest.at("sources").at(name).get<std::string>()));
  return result;
}

Tensors workspace(const at::TensorOptions &options) {
  const auto fp32 = options.dtype(at::kFloat);
  return {at::empty({1024}, options), at::empty({1024}, fp32),  at::empty({1024}, fp32),
          at::empty({2048}, fp32),    at::empty({1024}, fp32),  at::empty({1024}, fp32),
          at::empty({2048}, fp32),    at::empty({3072}, fp32),  at::empty({1024}, fp32),
          at::empty({16, 4}, fp32),   at::empty({16, 4}, fp32), at::empty({16, 4, 128}, fp32)};
}

struct Case {
  std::string name;
  std::vector<std::vector<int64_t>> steps;
  double fill;
  bool dump_cache, nondefault_stream, check_overflow;
  std::string same_as;
};

std::vector<Case> read_cases(const xinfer::fs::path &path, const qwen::Spec &spec) {
  const auto manifest = xinfer::read_json(path);
  require(manifest.contains("cases") && manifest.at("cases").is_array() &&
              !manifest.at("cases").empty(),
          "Missing custom-op validation cases");
  std::vector<Case> result;
  std::unordered_set<std::string> names;
  for (const auto &item : manifest.at("cases")) {
    Case value{item.at("name").get<std::string>(),
               item.at("steps").get<std::vector<std::vector<int64_t>>>(),
               item.value("poison_kv", 0.0),
               item.value("dump_cache", false),
               item.value("nondefault_stream", false),
               item.value("check_overflow", false),
               item.value("same_as", std::string())};
    require(
        !value.name.empty() &&
            std::all_of(value.name.begin(), value.name.end(),
                        [](unsigned char c) { return std::isalnum(c) || c == '_' || c == '-'; }) &&
            names.insert(value.name).second,
        "Invalid or duplicate custom-op case name");
    require(!value.steps.empty() && std::isfinite(value.fill), "Invalid custom-op case");
    int64_t position = 0;
    for (const auto &tokens : value.steps) {
      spec.check_step_tokens(tokens);
      require(int64_t(tokens.size()) <= spec.capacity - position, "Case exceeds context capacity");
      position += int64_t(tokens.size());
    }
    if (!value.same_as.empty())
      require(value.same_as != value.name && names.contains(value.same_as),
              "Missing replay control");
    result.push_back(std::move(value));
  }
  return result;
}

struct Trace {
  Tensors logits, state;
};

class Probe {
public:
  Probe(const xinfer::fs::path &packages, const xinfer::fs::path &prototype, const qwen::Spec &spec)
      : spec_(spec), device_(xinfer::configured_device(spec_)),
        weights_(spec_.validate_files(packages), spec_.shared_weights),
        prefill_(weights_.package_path("prefill").string()), prototype_(prototype.string()),
        state_(spec_.state, device_) {
    require(spec_.state.size() == 56 && spec_.vocab == 151936,
            "Custom-op probe requires the Qwen3-0.6B state and vocabulary");
    c10::cuda::CUDAGuard guard(device_);
    weights_.load(prefill_, "prefill");
    require(prototype_.get_constant_fqns().empty(),
            "Bare custom-op AOTI package must take every weight explicitly");
    state_.reset();
    parameters_ = canonical_weights(prefill_, packages);
    require(parameters_.size() == 311, "Incorrect custom-op weight order");
    scratch_ = workspace(state_.tensors()[0].options());
    const auto cpu = at::TensorOptions().device(at::kCPU).dtype(at::kFloat);
    const auto exponents = at::arange(0, 128, 2, cpu) / 128.0;
    const auto inverse = at::reciprocal(at::pow(1000000.0, exponents));
    const auto frequency = at::arange(spec_.capacity, cpu).unsqueeze(1) * inverse.unsqueeze(0);
    const auto cosine = at::cos(frequency), sine = at::sin(frequency);
    cos_ = at::cat({cosine, cosine}, 1).to(at::kBFloat16).to(device_).contiguous();
    sin_ = at::cat({sine, sine}, 1).to(at::kBFloat16).to(device_).contiguous();
    extent_ = at::empty({spec_.capacity + 1}, cpu.dtype(at::kByte));
    weight_addresses_ = addresses(parameters_);
    state_addresses_ = addresses(state_.tensors());
    scratch_addresses_ = addresses(scratch_);
  }

  void describe(Json &report) const {
    report["weight_names"] = weight_names();
    report["caller_weight_pointers"] = weight_addresses_;
    report["caller_kv_pointers"] = state_addresses_;
    report["caller_workspace_pointers"] = scratch_addresses_;
    report["caller_cos_pointer"] = address(cos_);
    report["caller_sin_pointer"] = address(sin_);
    report["bare_aoti_tensor_inputs"] = 383;
    report["bare_aoti_constants"] = 0;
  }

  Trace run(const Case &test, const std::string &path, const xinfer::fs::path &output, Json &row) {
    c10::cuda::CUDAGuard guard(device_);
    const auto current = c10::cuda::getCurrentCUDAStream(device_.index());
    C10_CUDA_CHECK(cudaStreamSynchronize(current.stream()));
    const auto selected =
        test.nondefault_stream ? c10::cuda::getStreamFromPool(false, device_.index()) : current;
    c10::cuda::CUDAStreamGuard stream_guard(selected);
    state_.reset(test.fill);
    for (auto &tensor : scratch_)
      tensor.fill_(std::numeric_limits<float>::quiet_NaN());
    Trace trace;
    std::vector<std::pair<at::Tensor, at::Tensor>> retained;
    int64_t position = 0, prefill_calls = 0, decode_calls = 0, selected_calls = 0;
    const auto filename = test.name + "." + path + ".f32";
    auto logits_file = xinfer::output_file(output / filename, true);
    for (size_t step = 0; step < test.steps.size(); ++step) {
      const auto &tokens = test.steps[step];
      const auto length = int64_t(tokens.size());
      const bool use_selected = length == 1;
      const auto options = at::TensorOptions().device(device_).dtype(at::kLong);
      const auto ids = at::tensor(tokens, at::kLong).view({1, length}).to(options);
      const auto extent = extent_.narrow(0, 0, position + length + 1);
      const auto original_ids = ids.clone();
      extent.fill_(255);
      Tensors before;
      if (use_selected)
        for (const auto &tensor : state_.tensors())
          before.push_back(tensor.clone());
      at::Tensor score;
      if (!use_selected) {
        Tensors inputs{ids, at::arange(position, position + length, options)};
        inputs.insert(inputs.end(), state_.tensors().begin(), state_.tensors().end());
        inputs.push_back(extent);
        auto result = prefill_.run(inputs);
        state_.accept(result);
        score = result[0];
      } else if (path == "op") {
        score =
            call_megaqwen_decode(ids, parameters_, state_.tensors(), cos_, sin_, scratch_, extent);
        state_.accept({score});
      } else {
        auto result = prototype_.run(prototype_inputs(ids, extent));
        state_.accept(result);
        score = result[0];
      }
      require(score.sizes() == at::IntArrayRef({1, spec_.vocab}) &&
                  score.scalar_type() == at::kBFloat16 && score.device() == device_,
              "Incorrect custom-op logits");
      require(at::isfinite(score).all().item<bool>(), "Nonfinite custom-op logits");
      require(bits_equal(ids, original_ids) && extent.eq(255).all().item<bool>(),
              "Custom-op route changed token IDs or shape-only extent data");
      require(addresses(state_.tensors()) == state_addresses_ &&
                  addresses(parameters_) == weight_addresses_ &&
                  addresses(scratch_) == scratch_addresses_,
              "Custom-op route changed caller-owned storage");
      if (use_selected) {
        check_output_storage(ids, score);
        check_regions(before, position, position + length);
        for (const auto &[previous, _] : retained)
          require(address(previous) != address(score), "Selected decode reused retained logits");
        ++selected_calls;
      }
      for (const auto &[previous, expected] : retained)
        require(bits_equal(previous.to(at::kCPU, at::kFloat), expected),
                "A later call changed retained selected-decode output");
      const auto cpu = score.to(at::kCPU, at::kFloat).clone();
      trace.logits.push_back(cpu);
      xinfer::write_tensor(logits_file, cpu);
      if (use_selected)
        retained.emplace_back(score, cpu);
      position += length;
      prefill_calls += length > 1;
      decode_calls += length == 1;
    }
    logits_file.close();
    require(bool(logits_file), "Failed to write custom-op logits");
    for (const auto &tensor : state_.tensors()) {
      require(at::isfinite(tensor).all().item<bool>(), "Nonfinite custom-op KV state");
      require(tensor.slice(2, position).eq(test.fill).all().item<bool>(),
              "Custom-op route overwrote future KV");
      trace.state.push_back(tensor.slice(2, 0, position).to(at::kCPU).contiguous().clone());
    }
    Json details{{"logits_file", filename},
                 {"prefill_calls", prefill_calls},
                 {"decode_calls", decode_calls},
                 {"selected_decode_calls", selected_calls},
                 {"singleton_backend", path},
                 {"caller_owned_storage_preserved", true},
                 {"token_and_extent_inputs_unchanged", true},
                 {"fresh_output_storage", true},
                 {"stream_id", selected.id()},
                 {"nondefault_stream", test.nondefault_stream}};
    if (test.check_overflow) {
      require(position == spec_.capacity, "Overflow fixture must fill the context");
      const auto ids = at::full({1, 1}, 13, state_.tensors()[0].options().dtype(at::kLong));
      const auto invalid_extent = at::empty({position + 2}, extent_.options());
      bool rejected = false;
      try {
        if (path == "op")
          call_megaqwen_decode(ids, parameters_, state_.tensors(), cos_, sin_, scratch_,
                               invalid_extent);
        else
          prototype_.run(prototype_inputs(ids, invalid_extent));
      } catch (const std::exception &) {
        rejected = true;
      }
      require(rejected, "Selected decode accepted an extent beyond KV capacity");
      for (size_t i = 0; i < trace.state.size(); ++i)
        require(bits_equal(state_.tensors()[i].to(at::kCPU), trace.state[i]),
                "Rejected extent changed KV state");
      require(addresses(state_.tensors()) == state_addresses_,
              "Rejected extent replaced caller-owned KV");
      details["overflow_rejected_without_mutation"] = true;
    }
    if (test.dump_cache) {
      const auto cache_name = test.name + "." + path + ".kv.bf16";
      auto cache_file = xinfer::output_file(output / cache_name, true);
      for (const auto &tensor : trace.state)
        xinfer::write_tensor(cache_file, tensor);
      cache_file.close();
      require(bool(cache_file), "Failed to write custom-op cache trace");
      details["cache_file"] = cache_name;
    }
    row["paths"][path] = std::move(details);
    row["position"] = position;
    C10_CUDA_CHECK(cudaStreamSynchronize(selected.stream()));
    return trace;
  }

private:
  Tensors prototype_inputs(const at::Tensor &ids, const at::Tensor &extent) const {
    Tensors inputs{ids};
    inputs.insert(inputs.end(), parameters_.begin(), parameters_.end());
    inputs.insert(inputs.end(), state_.tensors().begin(), state_.tensors().end());
    inputs.push_back(cos_);
    inputs.push_back(sin_);
    inputs.insert(inputs.end(), scratch_.begin(), scratch_.end());
    inputs.push_back(extent);
    require(inputs.size() == 383, "Incorrect bare AOTI input flattening");
    return inputs;
  }

  void check_regions(const Tensors &before, int64_t start, int64_t end) const {
    for (size_t i = 0; i < before.size(); ++i) {
      const auto &after = state_.tensors()[i];
      require(bits_equal(before[i].slice(2, 0, start), after.slice(2, 0, start)),
              "Selected decode overwrote past KV");
      require(bits_equal(before[i].slice(2, end), after.slice(2, end)),
              "Selected decode overwrote future KV");
      require(!bits_equal(before[i].slice(2, start, end), after.slice(2, start, end)),
              "Selected decode omitted a KV append");
    }
  }

  void check_output_storage(const at::Tensor &ids, const at::Tensor &score) const {
    for (const auto &tensors : {parameters_, state_.tensors(), scratch_, Tensors{ids, cos_, sin_}})
      for (const auto &tensor : tensors) {
        const auto input = address(tensor), output = address(score);
        require(output + score.nbytes() <= input || input + tensor.nbytes() <= output,
                "Custom-op output overlaps caller-owned input storage");
      }
  }

  const qwen::Spec spec_;
  const c10::Device device_;
  xinfer::SharedWeights weights_;
  Loader prefill_, prototype_;
  xinfer::StatePolicy state_;
  Tensors parameters_, scratch_;
  at::Tensor cos_, sin_, extent_;
  Pointers weight_addresses_, state_addresses_, scratch_addresses_;
};

void native_libraries(Json &report) {
  std::ifstream maps("/proc/self/maps");
  require(maps.is_open(), "Cannot inspect native process mappings");
  bool attention = false, megaqwen = false;
  for (std::string line; std::getline(maps, line);) {
    require(line.find("libpython") == std::string::npos &&
                line.find("libtorch_python") == std::string::npos,
            "Custom-op native probe loaded a Python library");
    attention = attention || line.find("libxinfer_attention_ops.so") != std::string::npos;
    megaqwen = megaqwen || line.find("libxinfer_megaqwen.so") != std::string::npos;
  }
  require(attention && megaqwen, "Missing native operator dependency");
  report["no_python_library"] = true;
  report["attention_library"] = attention;
  report["megaqwen_library"] = megaqwen;
}

void validate(const xinfer::fs::path &packages, const xinfer::fs::path &prototype,
              const xinfer::fs::path &manifest, const xinfer::fs::path &output, Json &report) {
  const qwen::Spec spec(packages);
  const auto cases = read_cases(manifest, spec);
  Probe probe(packages, prototype, spec);
  report["capacity"] = spec.capacity;
  report["torch"] = TORCH_VERSION;
  report["original_packages"] = xinfer::fs::absolute(packages).string();
  report["prototype"] = xinfer::fs::absolute(prototype).string();
  probe.describe(report);
  std::unordered_set<std::string> controls;
  for (const auto &test : cases)
    if (!test.same_as.empty())
      controls.insert(test.same_as);
  std::unordered_map<std::string, Trace> saved;
  bool ordered_stream = false;
  int64_t selected_decode_calls = 0;
  for (const auto &test : cases) {
    report["cases"].push_back({{"name", test.name}, {"lengths", Json::array()}});
    auto &row = report["cases"].back();
    for (const auto &tokens : test.steps)
      row["lengths"].push_back(tokens.size());
    auto original = probe.run(test, "op", output, row);
    const auto candidate = probe.run(test, "aoti", output, row);
    equal_tensors(candidate.logits, original.logits, "AOTI versus direct-op logits");
    equal_tensors(candidate.state, original.state, "AOTI versus direct-op KV");
    row["op_vs_aoti_bitwise"] = true;
    if (!test.same_as.empty()) {
      require(saved.contains(test.same_as), "Missing custom-op replay control");
      equal_tensors(original.logits, saved.at(test.same_as).logits, "Poison/reset replay logits");
      equal_tensors(original.state, saved.at(test.same_as).state, "Poison/reset valid KV");
      row["replay_bitwise"] = true;
    }
    if (controls.contains(test.name))
      saved.emplace(test.name, std::move(original));
    row["cache_bitwise"] = true;
    row["passed"] = true;
    ordered_stream = ordered_stream ||
                     (test.nondefault_stream && row["paths"]["op"]["selected_decode_calls"] > 0);
    selected_decode_calls += row["paths"]["op"]["selected_decode_calls"].get<int64_t>();
  }
  report["ordered_stream_cases"] = ordered_stream;
  require(selected_decode_calls > 0, "No custom-op decode case was exercised");
  report["selected_decode_calls"] = selected_decode_calls;
  report["caller_owned_storage_preserved"] = true;
  report["case_count"] = cases.size();
  native_libraries(report);
}
} // namespace

int main(int argc, char **argv) {
  Json report{{"passed", false}, {"benchmark", false}, {"cases", Json::array()}};
  xinfer::fs::path output;
  bool created = false;
  try {
    require(argc == 5,
            "Usage: qwen_megaqwen_op_contract ORIGINAL_PACKAGES DIRECT_PT2 CASES_JSON OUTPUT_DIR");
    output = argv[4];
    require(!xinfer::fs::exists(output), "Output directory already exists");
    require(xinfer::fs::create_directories(output), "Cannot create a fresh output directory");
    created = true;
    c10::InferenceMode inference;
    xinfer::configure_runtime();
    validate(argv[1], argv[2], argv[3], output, report);
    report["passed"] = true;
  } catch (const std::exception &error) {
    report["error"] = error.what();
    std::cerr << error.what() << '\n';
  }
  if (created) {
    try {
      auto file = xinfer::output_file(output / "summary.json");
      file << report.dump(2) << '\n';
      file.close();
      require(bool(file), "Failed to write custom-op validation report");
    } catch (const std::exception &error) {
      std::cerr << error.what() << '\n';
      return 2;
    }
  }
  return report["passed"].get<bool>() ? 0 : 1;
}
