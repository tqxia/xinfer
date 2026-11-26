// CPU-only checks of caller-owned in-place state storage and logits-only ABI validation.
#include "common/state.h"

#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {
using xinfer::require;
using xinfer::StatePolicy;
using xinfer::StateSpec;
using Tensors = std::vector<at::Tensor>;

StateSpec descriptor(const std::string &kind, int64_t batch = 1) {
  const auto shape =
      kind == "conv" ? std::vector<int64_t>{batch, 2, 3} : std::vector<int64_t>{batch, 2, 3, 4};
  std::vector<int64_t> stride(shape.size());
  int64_t step = 1;
  for (size_t i = shape.size(); i-- > 0;) {
    stride[i] = step;
    step *= shape[i];
  }

  return {
      "layer_0." + kind, kind, 0, shape, stride, kind == "recurrent" ? at::kFloat : at::kBFloat16};
}

std::vector<StateSpec> kv_specs(int64_t batch = 1) {
  return {descriptor("key", batch), descriptor("value", batch)};
}

std::vector<StateSpec> mixed_specs(int64_t batch = 1) {
  return {descriptor("conv", batch), descriptor("recurrent", batch), descriptor("key", batch),
          descriptor("value", batch)};
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
            "State storage or values changed unexpectedly");
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
      reject([&] { state.accept({at::zeros({1})}, 1); });

      state.reset(7);
      const auto before = state.tensors();
      for (const auto &tensor : before)
        require((reinterpret_cast<uintptr_t>(tensor.data_ptr()) & 15) == 0,
                "KV state buffer is not 16-byte aligned");
      state.accept({at::zeros({1})}, 1);
      unchanged(state.tensors(), before);

      state.reset(-4);
      for (size_t i = 0; i < before.size(); ++i)
        require(state.tensors()[i].const_data_ptr() == before[i].const_data_ptr() &&
                    state.tensors()[i].eq(-4).all().item<bool>(),
                "KV reset did not preserve storage or fill value");

      reject([&] { state.accept({}, 1); });
      reject([&] { state.accept({at::zeros({1}), state.tensors()[0]}, 1); });
    });

    check("qwen35_inplace_and_reset", [&] {
      StatePolicy state(mixed_specs(), cpu);
      state.reset(7);
      const auto before = state.tensors();
      for (size_t i = 0; i < before.size(); ++i) {
        const bool kv = i >= 2;
        require((reinterpret_cast<uintptr_t>(before[i].data_ptr()) & 15) == 0,
                "Mixed state buffer is not 16-byte aligned");
        require(before[i].eq(kv ? 7 : 0).all().item<bool>(),
                "Initial reset did not poison only KV state");
      }

      // Mutating caller-owned DeltaNet buffers in place must retain pointers and values.
      state.tensors()[0].fill_(2);
      state.tensors()[1].fill_(3);
      state.accept({at::zeros({1})}, 1);
      for (size_t i = 0; i < before.size(); ++i)
        require(state.tensors()[i].const_data_ptr() == before[i].const_data_ptr(),
                "In-place state changed caller-owned storage");
      require(state.tensors()[0].eq(2).all().item<bool>() &&
                  state.tensors()[1].eq(3).all().item<bool>(),
              "In-place DeltaNet state update was lost");

      state.reset(-4);
      for (size_t i = 0; i < before.size(); ++i) {
        const bool kv = i >= 2;
        require(state.tensors()[i].const_data_ptr() == before[i].const_data_ptr(),
                "Reset reallocated caller-owned state storage");
        require(state.tensors()[i].eq(kv ? -4 : 0).all().item<bool>(),
                "Reset did not poison only KV state");
      }

      reject([&] { state.accept({}, 1); });
      reject([&] { state.accept({at::zeros({1}), state.tensors()[0]}, 1); });
      reject([&] {
        state.accept({at::zeros({1}), state.tensors()[0], state.tensors()[1], state.tensors()[2],
                      state.tensors()[3]},
                     1);
      });
    });

    check("qwen35_dynamic_batch_preserves_storage_and_tail", [&] {
      StatePolicy state(mixed_specs(8), cpu);
      state.reset();
      for (const auto batch : {8, 1, 3, 2, 8}) {
        const auto ptrs = state.tensors();
        Tensors before;
        for (const auto &tensor : ptrs)
          before.push_back(tensor.clone());

        for (size_t i = 0; i < ptrs.size(); ++i)
          state.tensors()[i].narrow(0, 0, batch).add_(1);
        state.accept({at::zeros({batch})}, batch);

        for (size_t i = 0; i < ptrs.size(); ++i) {
          const auto &after = state.tensors()[i];
          require(after.size(0) == 8, "State allocation shrank");
          require(after.const_data_ptr() == ptrs[i].const_data_ptr(),
                  "In-place state storage changed");
          require(
              at::equal(after.narrow(0, batch, 8 - batch), before[i].narrow(0, batch, 8 - batch)),
              "Unused state rows changed");
          require(at::equal(after.narrow(0, 0, batch), before[i].narrow(0, 0, batch) + 1),
                  "Active state rows did not advance");
        }
      }

      reject([&] { state.accept({at::zeros({1})}, 0); });
      reject([&] { state.accept({at::zeros({8})}, 9); });
    });

    check("allocated_batch_size_bounds_and_allocation", [&] {
      reject([&] { StatePolicy state({}, cpu); });
      reject([&] { StatePolicy state(mixed_specs(8), cpu, 0); });
      reject([&] { StatePolicy state(mixed_specs(8), cpu, -1); });
      reject([&] { StatePolicy state(mixed_specs(8), cpu, 9); });

      for (const int64_t allocated : {1, 3}) {
        StatePolicy state(mixed_specs(8), cpu, allocated);
        require(state.allocated_batch_size() == allocated, "Incorrect allocated batch size");
        state.reset(7);
        const auto before = state.tensors();
        for (size_t i = 0; i < before.size(); ++i) {
          require(before[i].size(0) == allocated, "State did not allocate requested batch size");
          require(before[i].eq(i >= 2 ? 7 : 0).all().item<bool>(),
                  "Reset did not initialize allocated state");
        }

        state.accept({at::zeros({allocated})}, allocated);
        unchanged(state.tensors(), before);
        reject([&] { state.accept({at::zeros({allocated})}, 0); });
        reject([&] { state.accept({at::zeros({allocated})}, allocated + 1); });
      }
    });

    using DescriptorMutation = std::function<void(std::vector<StateSpec> &)>;
    const std::vector<std::pair<const char *, DescriptorMutation>> bad_descriptors{
        {"inconsistent_batch_size", [](auto &specs) { specs[3].shape[0] = 2; }}};
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
