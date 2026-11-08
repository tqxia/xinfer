// Correctness-only probes of the same attention helper used by MegaQwen decode.
// The reference materializes scores on the CPU and uses FP64 dense softmax.
#include "common/io.h"
#include "common/settings.h"
#include "attention.h"

#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/version.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {
using xinfer::Json;
using xinfer::require;
namespace mega = xinfer::megaqwen;

constexpr int kQueryHeads = 16;
constexpr int kKVHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kDefaultCapacity = 1024;
constexpr int kGuardElements = 32;
constexpr float kGuardValue = 12345.25f;

bool bits_equal(const at::Tensor &a, const at::Tensor &b) {
  return a.sizes() == b.sizes() && a.scalar_type() == b.scalar_type() &&
         at::equal(a.contiguous().view(at::kByte), b.contiguous().view(at::kByte));
}

struct GuardedBuffer {
  at::Tensor storage;
  at::Tensor values;

  explicit GuardedBuffer(int64_t count) {
    const auto options = at::TensorOptions().device(at::kCUDA, 0).dtype(at::kFloat);
    storage = at::full({count + 2 * kGuardElements}, kGuardValue, options);
    values = storage.slice(0, kGuardElements, kGuardElements + count);
  }

  float *data() { return values.data_ptr<float>(); }

  void check(const std::string &name) const {
    require(
        storage.slice(0, 0, kGuardElements).eq(kGuardValue).all().item<bool>() &&
            storage.slice(0, storage.numel() - kGuardElements).eq(kGuardValue).all().item<bool>(),
        name + ": attention overwrote a buffer guard");
  }
};

struct Fixture {
  at::Tensor query;
  at::Tensor key;
  at::Tensor value;
};

Fixture fixture(const std::string &kind, int length, int capacity = kDefaultCapacity) {
  require(length > 0 && length <= capacity, "Invalid attention fixture length/capacity");
  const auto cpu = at::TensorOptions().device(at::kCPU);
  Fixture result{at::empty({kQueryHeads, kHeadDim}, cpu.dtype(at::kFloat)),
                 at::empty({kKVHeads, capacity, kHeadDim}, cpu.dtype(at::kBFloat16)),
                 at::empty({kKVHeads, capacity, kHeadDim}, cpu.dtype(at::kBFloat16))};
  auto *query = result.query.data_ptr<float>();
  auto *key = result.key.data_ptr<at::BFloat16>();
  auto *value = result.value.data_ptr<at::BFloat16>();
  std::mt19937 generator(1907 + length);
  std::uniform_real_distribution<float> random(-1, 1);
  for (int head = 0; head < kQueryHeads; ++head)
    for (int d = 0; d < kHeadDim; ++d) {
      float q = random(generator);
      if (kind == "uniform_scores")
        q = 0;
      if (kind == "peaked_scores")
        q = d == 0 ? (head % 2 == 0 ? 256 : -256) : 0;
      query[head * kHeadDim + d] = q;
    }
  for (int head = 0; head < kKVHeads; ++head)
    for (int position = 0; position < capacity; ++position)
      for (int d = 0; d < kHeadDim; ++d) {
        const int index = (head * capacity + position) * kHeadDim + d;
        float k = random(generator), v = random(generator) + head * 0.125f;
        if (kind == "uniform_scores")
          v = (head + 1) * 0.0625f + (position % 17 - 8) * 0.03125f + (d % 11 - 5) * 0.015625f;
        if (kind == "peaked_scores")
          k = d == 0 ? (position == length / 2 ? 256 : -256) : 0;
        if (position >= length)
          k = v = std::numeric_limits<float>::quiet_NaN();
        key[index] = at::BFloat16(k);
        value[index] = at::BFloat16(v);
      }
  return result;
}

at::Tensor dense_reference(const Fixture &input, int length) {
  const auto capacity = input.key.size(1);
  const auto *query = input.query.const_data_ptr<float>();
  const auto *key = input.key.const_data_ptr<at::BFloat16>();
  const auto *value = input.value.const_data_ptr<at::BFloat16>();
  auto output = at::zeros({kQueryHeads, kHeadDim}, at::TensorOptions().dtype(at::kDouble));
  auto *result = output.data_ptr<double>();
  std::vector<double> scores(length);
  for (int head = 0; head < kQueryHeads; ++head) {
    const int kv_head = head / (kQueryHeads / kKVHeads);
    for (int position = 0; position < length; ++position) {
      double dot = 0;
      for (int d = 0; d < kHeadDim; ++d)
        dot += double(query[head * kHeadDim + d]) *
               float(key[(kv_head * capacity + position) * kHeadDim + d]);
      scores[position] = dot / std::sqrt(double(kHeadDim));
    }
    const double maximum = *std::max_element(scores.begin(), scores.end());
    double denominator = 0;
    for (double &score : scores) {
      score = std::exp(score - maximum);
      denominator += score;
    }
    for (int position = 0; position < length; ++position)
      for (int d = 0; d < kHeadDim; ++d)
        result[head * kHeadDim + d] += (scores[position] / denominator) *
                                       float(value[(kv_head * capacity + position) * kHeadDim + d]);
  }
  // Production writes FP32 values rounded through the final BF16 boundary.
  return output.to(at::kBFloat16).to(at::kFloat);
}

Json compare(const at::Tensor &actual, const at::Tensor &expected, const std::string &label) {
  require(actual.sizes() == expected.sizes() && actual.scalar_type() == at::kFloat,
          label + ": incorrect output shape/dtype");
  require(at::isfinite(actual).all().item<bool>(), label + ": nonfinite attention output");
  require(bits_equal(actual, actual.to(at::kBFloat16).to(at::kFloat)),
          label + ": output omitted the BF16 rounding boundary");
  const auto *a = actual.const_data_ptr<float>();
  const auto *e = expected.const_data_ptr<float>();
  double squared_error = 0, squared_reference = 0, maximum_error = 0;
  for (int64_t i = 0; i < actual.numel(); ++i) {
    const double error = std::abs(double(a[i]) - e[i]);
    // Permit one BF16 output ULP plus a small absolute margin near zero.
    // Split reductions change FP32 accumulation order, so bitwise equality to
    // a dense FP64 reference is not the numerical contract.
    const double ulp = e[i] == 0 ? 0 : std::ldexp(1.0, std::ilogb(std::abs(e[i])) - 7);
    require(error <= ulp + 5e-6,
            label + ": output exceeds one BF16 ULP at element " + std::to_string(i));
    squared_error += error * error;
    squared_reference += double(e[i]) * e[i];
    maximum_error = std::max(maximum_error, error);
  }
  const double relative_l2 = std::sqrt(squared_error / std::max(squared_reference, 1e-20));
  require(relative_l2 <= 0.005, label + ": relative L2 exceeds 0.5%");
  return {{"relative_l2", relative_l2}, {"max_abs_error", maximum_error}};
}

void policy_contract(Json &report) {
  for (int grid : {16, 31, 32, 47, 63, 64, 92})
    for (int length : {1, 128, 129, 256, 257, 1024, 1025, 8193, 16384, 40960}) {
      const int wanted = length <= 128 ? 1 : (length <= 256 ? 2 : 4);
      const int available = grid >= 64 ? 4 : (grid >= 32 ? 2 : 1);
      const int expected = std::min(wanted, available);
      require(mega::attention_partition_count(length, grid) == expected,
              "Attention partition policy exceeds available resident blocks or token threshold");
    }
  report["partition_policy"] = true;
}

void validate(Json &report) {
  const auto resources = mega::query_attention_launch_info();
  const int grid = resources.grid_blocks;
  require(grid >= kQueryHeads && resources.active_blocks_per_sm > 0,
          "Attention probe cannot launch a resident cooperative grid");
  report["launch"] = {{"grid_blocks", grid},
                      {"sm_count", resources.sm_count},
                      {"active_blocks_per_sm", resources.active_blocks_per_sm},
                      {"registers_per_thread", resources.registers_per_thread},
                      {"static_shared_bytes", resources.static_shared_bytes}};
  policy_contract(report);
  GuardedBuffer maximum(kQueryHeads * mega::kMaxAttentionPartitions);
  GuardedBuffer denominator(kQueryHeads * mega::kMaxAttentionPartitions);
  GuardedBuffer weighted(kQueryHeads * mega::kMaxAttentionPartitions * kHeadDim);
  GuardedBuffer output(kQueryHeads * kHeadDim);
  const mega::AttentionPartials partials{maximum.data(), denominator.data(), weighted.data()};
  const auto stream = c10::cuda::getCurrentCUDAStream(0);
  struct AttentionCase {
    std::string kind;
    int length;
    int capacity;
  };
  std::vector<AttentionCase> fixtures;
  for (const std::string kind : {"random", "uniform_scores", "peaked_scores"})
    for (int length : {1, 7, 8, 9, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024})
      fixtures.push_back({kind, length, kDefaultCapacity});
  // Stress capacity-dependent head strides independently of the prefix length.
  // Keep only one layer's KV live; the largest fixture is 160 MiB per device.
  const std::vector<std::pair<int, int>> long_cases{
      {512, 512},     {1537, 1025},   {8193, 8191},  {8193, 8192},   {8193, 8193},  {16384, 8193},
      {16384, 16383}, {16384, 16384}, {40960, 1025}, {40960, 40959}, {40960, 40960}};
  for (const auto &[capacity, length] : long_cases)
    for (const std::string kind : {"random", "uniform_scores"})
      fixtures.push_back({kind, length, capacity});
  for (const auto &[kind, length, capacity] : fixtures) {
    const auto input = fixture(kind, length, capacity);
    const auto expected = dense_reference(input, length);
    auto query = input.query.to(at::Device(at::kCUDA, 0));
    auto key = input.key.to(at::Device(at::kCUDA, 0));
    auto value = input.value.to(at::Device(at::kCUDA, 0));
    std::vector<int> modes{1, 2, 4};
    const int automatic = mega::attention_partition_count(length, grid);
    modes.push_back(automatic);
    for (size_t mode = 0; mode < modes.size(); ++mode) {
      const int partitions = modes[mode];
      if (partitions * kQueryHeads > grid)
        continue;
      const auto label = kind + "/capacity_" + std::to_string(capacity) + "/length_" +
                         std::to_string(length) + "/partitions_" + std::to_string(partitions) +
                         (mode == 3 ? "/automatic" : "/forced");
      auto launch = [&](float poison) {
        for (auto *buffer : {&maximum, &denominator, &weighted, &output})
          buffer->values.fill_(poison);
        mega::launch_attention_for_test(query.const_data_ptr<float>(), key.const_data_ptr(),
                                        value.const_data_ptr(), output.data(), partials, length,
                                        capacity, partitions, grid, stream.stream());
        C10_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
        for (auto *buffer : {&maximum, &denominator, &weighted, &output})
          buffer->check(label);
        return output.values.reshape({kQueryHeads, kHeadDim}).cpu().clone();
      };
      const auto actual = launch(std::numeric_limits<float>::quiet_NaN());
      auto row = compare(actual, expected, label);
      const auto replay = launch(-9876.5f);
      require(bits_equal(actual, replay), label + ": scratch contents changed replay output");
      require(bits_equal(query.cpu(), input.query) && bits_equal(key.cpu(), input.key) &&
                  bits_equal(value.cpu(), input.value),
              label + ": attention modified Q/K/V input");
      row.update({{"fixture", kind},
                  {"cache_len", length},
                  {"capacity", capacity},
                  {"partitions", partitions},
                  {"automatic", mode == 3},
                  {"passed", true},
                  {"nan_future_ignored", true},
                  {"scratch_poison_replay", true},
                  {"inputs_unchanged", true},
                  {"guards_intact", true}});
      report["cases"].push_back(std::move(row));
    }
  }
  // Exercise resource-limited policies with real launches, including grids
  // that have spare blocks not assigned to an attention partition.
  report["small_grid_cases"] = Json::array();
  const auto input = fixture("random", 257);
  const auto expected = dense_reference(input, 257);
  const auto query = input.query.to(at::Device(at::kCUDA, 0));
  const auto key = input.key.to(at::Device(at::kCUDA, 0));
  const auto value = input.value.to(at::Device(at::kCUDA, 0));
  for (int small_grid : {16, 31, 32, 63, 64}) {
    if (small_grid > grid)
      continue;
    const int partitions = mega::attention_partition_count(257, small_grid);
    for (auto *buffer : {&maximum, &denominator, &weighted, &output})
      buffer->values.fill_(std::numeric_limits<float>::quiet_NaN());
    mega::launch_attention_for_test(query.const_data_ptr<float>(), key.const_data_ptr(),
                                    value.const_data_ptr(), output.data(), partials, 257,
                                    kDefaultCapacity, partitions, small_grid, stream.stream());
    C10_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
    const auto label = "small_grid_" + std::to_string(small_grid);
    auto row = compare(output.values.reshape({kQueryHeads, kHeadDim}).cpu(), expected, label);
    for (auto *buffer : {&maximum, &denominator, &weighted, &output})
      buffer->check(label);
    row.update({{"grid_blocks", small_grid}, {"partitions", partitions}, {"passed", true}});
    report["small_grid_cases"].push_back(std::move(row));
  }
  report["case_count"] = report["cases"].size();
}
} // namespace

int main(int argc, char **argv) {
  Json report = {{"passed", false},
                 {"benchmark", false},
                 {"cases", Json::array()},
                 {"reference", "cpu_fp64_dense_softmax_then_bf16"},
                 {"relative_l2_limit", 0.005},
                 {"absolute_margin", 5e-6},
                 {"output_ulp_limit", 1}};
  try {
    require(argc == 2, "Usage: qwen_attention_contract OUTPUT_JSON");
    require(!xinfer::fs::exists(argv[1]), "Output already exists");
    c10::InferenceMode inference;
    c10::cuda::CUDAGuard guard(c10::Device(at::kCUDA, 0));
    xinfer::configure_runtime();
    report["torch"] = TORCH_VERSION;
    validate(report);
    report["passed"] = true;
  } catch (const std::exception &error) {
    report["error"] = error.what();
    std::cerr << error.what() << '\n';
  }
  if (argc == 2 && !xinfer::fs::exists(argv[1])) {
    try {
      auto file = xinfer::output_file(argv[1]);
      file << report.dump(2) << '\n';
      file.close();
      require(bool(file), "Failed to write attention contract report");
    } catch (const std::exception &error) {
      std::cerr << error.what() << '\n';
      return 2;
    }
  }
  return report["passed"].get<bool>() ? 0 : 1;
}
