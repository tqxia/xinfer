// CPU-only checks of state ownership and returned-state validation.
#include "common/state.h"

#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {
using xinfer::require;
using xinfer::StatePolicy;
using xinfer::StateSpec;
using Tensors = std::vector<at::Tensor>;

StateSpec descriptor(const std::string &kind, bool inplace, std::optional<size_t> output) {
  const auto shape =
      kind == "conv" ? std::vector<int64_t>{1, 2, 3} : std::vector<int64_t>{1, 2, 3, 4};
  std::vector<int64_t> stride(shape.size());
  int64_t step = 1;
  for (size_t i = shape.size(); i-- > 0;) {
    stride[i] = step;
    step *= shape[i];
  }
  return {
      "layer_0." + kind, kind,  0, shape, stride, kind == "recurrent" ? at::kFloat : at::kBFloat16,
      inplace,           output};
}

std::vector<StateSpec> kv_specs() {
  return {descriptor("key", true, std::nullopt), descriptor("value", true, std::nullopt)};
}

std::vector<StateSpec> mixed_specs() {
  return {descriptor("conv", false, 1), descriptor("recurrent", false, 2),
          descriptor("key", true, 3), descriptor("value", true, 4)};
}

Tensors valid_outputs(const StatePolicy &state) {
  const auto &tensors = state.tensors();
  return {at::zeros({1}), at::full_like(tensors[0], 2), at::full_like(tensors[1], 3), tensors[2],
          tensors[3]};
}

void reject(const std::function<void()> &operation) {
  bool rejected = false;
  try {
    operation();
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  require(rejected, "Invalid state contract was accepted");
}

void unchanged(const Tensors &actual, const Tensors &expected) {
  require(actual.size() == expected.size(), "State count changed after rejection");
  for (size_t i = 0; i < actual.size(); ++i)
    require(actual[i].const_data_ptr() == expected[i].const_data_ptr() &&
                actual[i].strides() == expected[i].strides() && at::equal(actual[i], expected[i]),
            "State was adopted before all outputs were validated");
}
} // namespace

int main() {
  try {
    c10::InferenceMode inference;
    const c10::Device cpu(c10::kCPU);
    xinfer::Json report = {{"device", "cpu"}, {"cases", xinfer::Json::array()}, {"passed", false}};
    auto check = [&](const char *name, const std::function<void()> &operation) {
      try {
        operation();
      } catch (const std::exception &error) {
        throw std::runtime_error(std::string(name) + ": " + error.what());
      }
      report["cases"].push_back({{"name", name}, {"passed", true}});
    };

    check("qwen3_logits_only", [&] {
      StatePolicy state(kv_specs(), cpu);
      state.reset(7);
      const auto before = state.tensors();
      state.accept({at::zeros({1})});
      unchanged(state.tensors(), before);
      state.reset(-4);
      for (size_t i = 0; i < before.size(); ++i)
        require(state.tensors()[i].const_data_ptr() == before[i].const_data_ptr() &&
                    state.tensors()[i].eq(-4).all().item<bool>(),
                "KV reset did not preserve storage or fill value");
      reject([&] { state.accept({}); });
      reject([&] { state.accept({at::zeros({1}), state.tensors()[0]}); });
    });

    check("qwen35_replace_and_alias", [&] {
      StatePolicy state(mixed_specs(), cpu);
      state.reset(7);
      const auto before = state.tensors();
      auto outputs = valid_outputs(state);
      // A returned KV view may be a different Tensor object with identical storage/layout.
      outputs[3] = outputs[3].view(outputs[3].sizes());
      state.accept(outputs);
      require(state.tensors()[0].const_data_ptr() == outputs[1].const_data_ptr() &&
                  state.tensors()[1].const_data_ptr() == outputs[2].const_data_ptr(),
              "Replacement state was not adopted");
      for (size_t i = 2; i < before.size(); ++i)
        require(state.tensors()[i].const_data_ptr() == before[i].const_data_ptr(),
                "Returned KV changed caller-owned storage");
      state.reset(-4);
      for (size_t i = 0; i < before.size(); ++i) {
        const bool inplace = i >= 2;
        require(state.tensors()[i].eq(inplace ? -4 : 0).all().item<bool>(),
                "Reset did not poison only KV state");
        if (inplace)
          require(state.tensors()[i].const_data_ptr() == before[i].const_data_ptr(),
                  "Reset reallocated KV storage");
      }
      require(outputs[1].eq(2).all().item<bool>() && outputs[2].eq(3).all().item<bool>(),
              "Reset modified prior replacement outputs");
    });

    using Mutation = std::function<void(Tensors &)>;
    const std::vector<std::pair<const char *, Mutation>> bad_outputs{
        {"missing_output", [](auto &outputs) { outputs.pop_back(); }},
        {"extra_output", [](auto &outputs) { outputs.push_back(outputs.back()); }},
        {"wrong_shape", [](auto &outputs) { outputs[1] = outputs[1].flatten(); }},
        {"wrong_dtype", [](auto &outputs) { outputs[1] = outputs[1].to(at::kFloat); }},
        {"wrong_device",
         [](auto &outputs) {
           outputs[1] = at::empty(outputs[1].sizes(), outputs[1].options().device(c10::kMeta));
         }},
        {"noncontiguous_replace",
         [](auto &outputs) {
           outputs[1] = at::zeros({1, 3, 2}, outputs[1].options()).transpose(1, 2);
         }},
        {"nonalias_kv", [](auto &outputs) { outputs[4] = outputs[4].clone(); }},
        {"different_kv_strides", [](auto &outputs) {
           auto strides = outputs[4].strides().vec();
           strides[0] += 1; // Still contiguous because the first dimension has size one.
           outputs[4] = outputs[4].as_strided(outputs[4].sizes(), strides);
           require(outputs[4].is_contiguous(), "Test must isolate the exact alias stride check");
         }}};
    for (const auto &[name, mutate] : bad_outputs)
      check(name, [&] {
        StatePolicy state(mixed_specs(), cpu);
        state.reset();
        const auto before = state.tensors();
        auto outputs = valid_outputs(state);
        mutate(outputs);
        reject([&] { state.accept(outputs); });
        // The later KV failures exercise validation before adopting earlier replacements.
        unchanged(state.tensors(), before);
      });

    using DescriptorMutation = std::function<void(std::vector<StateSpec> &)>;
    const std::vector<std::pair<const char *, DescriptorMutation>> bad_descriptors{
        {"missing_replace_output", [](auto &specs) { specs[0].output_index = std::nullopt; }},
        {"logits_output_as_state", [](auto &specs) { specs[0].output_index = 0; }},
        {"gapped_outputs", [](auto &specs) { specs.back().output_index = 5; }},
        {"duplicate_outputs", [](auto &specs) { specs.back().output_index = 3; }},
        {"reordered_outputs",
         [](auto &specs) { std::swap(specs[0].output_index, specs[1].output_index); }}};
    for (const auto &[name, mutate] : bad_descriptors)
      check(name, [&] {
        auto specs = mixed_specs();
        mutate(specs);
        reject([&] { StatePolicy state(specs, cpu); });
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
