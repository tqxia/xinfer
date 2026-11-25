#pragma once

#include "io.h"
#include "settings.h"
#include "weights.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/cuda.h>
#include <torch/version.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace xinfer {

inline constexpr const char *kMetadataFile = "metadata.json";
inline constexpr int64_t kMaxBatchSize = 8;
inline constexpr int64_t kMaxPrefillTokens = 512;
// Keep cumulative int32 KV lengths across all compiled batch slots below INT32_MAX.
inline constexpr int64_t kMaxCacheLength = std::numeric_limits<int32_t>::max() / kMaxBatchSize;

enum class ModelType { Qwen3, Qwen35, Qwen38 };

inline const char *model_name(ModelType model) {
  switch (model) {
  case ModelType::Qwen3:
    return "qwen3";
  case ModelType::Qwen35:
    return "qwen35";
  case ModelType::Qwen38:
    return "qwen38";
  }
  throw std::runtime_error("Unsupported model type");
}

inline ModelType parse_model_type(const std::string &type) {
  if (type == "qwen3")
    return ModelType::Qwen3;
  if (type == "qwen35")
    return ModelType::Qwen35;
  if (type == "qwen38")
    return ModelType::Qwen38;
  throw std::runtime_error("Unsupported model type");
}

inline ModelType model_from_identity(const Json &identity) {
  const auto model = identity.at("model").get<std::string>();
  if (model == "qwen3")
    return ModelType::Qwen3;
  if (model == "qwen3.5")
    return ModelType::Qwen35;
  if (model == "qwen3.8")
    return ModelType::Qwen38;
  throw std::runtime_error("Unsupported model identity: " + model);
}

inline void metadata_check(bool condition, const std::string &field) {
  require(condition, "Unsupported package metadata: " + field);
}

inline bool metadata_integer(const Json &value, int64_t minimum = 0) {
  if (!value.is_number_integer())
    return false;
  if (value.is_number_unsigned() &&
      value.get<uint64_t>() > uint64_t(std::numeric_limits<int64_t>::max()))
    return false;
  return value.get<int64_t>() >= minimum;
}

inline bool metadata_digest(const Json &value) {
  if (!value.is_string())
    return false;

  const auto text = value.get<std::string>();
  return text.size() == 64 && std::all_of(text.begin(), text.end(), [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}

inline const std::vector<std::string> &metadata_fields() {
  static const std::vector<std::string> fields{"model_type",   "torch_version",
                                               "cxx11_abi",    "cuda_compute_capability",
                                               "cache_length", "max_prefill_tokens",
                                               "vocab_size",   "eos_token_ids",
                                               "checkpoint",   "state",
                                               "entries",      "shared_weights"};
  return fields;
}

inline bool metadata_keys(const Json &value, const std::unordered_set<std::string> &keys) {
  if (!value.is_object() || value.size() != keys.size())
    return false;
  for (const auto &[key, _] : value.items())
    if (!keys.contains(key))
      return false;
  return true;
}

inline void validate_metadata(const Json &metadata) {
  require(metadata.is_object(), "Unsupported package metadata: expected an object");
  require(metadata.contains("model_type") && metadata.at("model_type").is_string() &&
              (metadata.at("model_type") == "qwen3" || metadata.at("model_type") == "qwen35" ||
               metadata.at("model_type") == "qwen38"),
          "Unsupported model type");

  const auto &fields = metadata_fields();
  for (const auto &field : fields)
    metadata_check(metadata.contains(field), field);
  for (const auto &[field, _] : metadata.items())
    metadata_check(std::find(fields.begin(), fields.end(), field) != fields.end(), field);

  const auto &m = metadata;
  metadata_check(metadata_integer(m.at("cache_length"), kMaxPrefillTokens) &&
                     m.at("cache_length").get<int64_t>() <= kMaxCacheLength,
                 "cache_length");
  metadata_check(metadata_integer(m.at("max_prefill_tokens"), 1) &&
                     m.at("max_prefill_tokens").get<int64_t>() == kMaxPrefillTokens &&
                     m.at("max_prefill_tokens").get<int64_t>() <=
                         m.at("cache_length").get<int64_t>(),
                 "max_prefill_tokens");
  metadata_check(metadata_integer(m.at("vocab_size"), 1), "vocab_size");

  metadata_check(m.at("torch_version").is_string() &&
                     !m.at("torch_version").get<std::string>().empty(),
                 "torch_version");
  metadata_check(m.at("cxx11_abi").is_boolean(), "cxx11_abi");
  const auto &sm = m.at("cuda_compute_capability");
  metadata_check(sm.is_array() && sm.size() == 2 && metadata_integer(sm[0], 1) &&
                     metadata_integer(sm[1]),
                 "cuda_compute_capability");

  const auto &eos = m.at("eos_token_ids");
  metadata_check(eos.is_array() && !eos.empty(), "eos_token_ids");
  for (const auto &token : eos)
    metadata_check(metadata_integer(token) &&
                       token.get<int64_t>() < m.at("vocab_size").get<int64_t>(),
                   "eos_token_ids");

  const auto &checkpoint = m.at("checkpoint");
  metadata_check(checkpoint.is_object(), "checkpoint");
  for (const auto *field : {"model", "revision"})
    metadata_check(checkpoint.contains(field) && checkpoint.at(field).is_string() &&
                       !checkpoint.at(field).get<std::string>().empty(),
                   "checkpoint");

  const auto &states = m.at("state");
  metadata_check(states.is_array() && !states.empty(), "state");
  std::unordered_set<std::string> names;
  for (const auto &item : states) {
    metadata_check(metadata_keys(item, {"layer", "kind", "shape", "dtype", "update"}), "state");
    metadata_check(item.at("kind").is_string() && metadata_integer(item.at("layer")), "state");
    const auto kind = item.at("kind").get<std::string>();
    const bool kv = kind == "key" || kind == "value";
    metadata_check(kv || kind == "conv" || kind == "recurrent", "state");
    const auto name = "layer_" + std::to_string(item.at("layer").get<int64_t>()) + "." + kind;
    metadata_check(names.insert(name).second, "state");

    const auto &shape = item.at("shape");
    metadata_check(shape.is_array() && shape.size() == (kind == "conv" ? 3 : 4), "state");
    int64_t elements = 1;
    for (const auto &dim : shape) {
      metadata_check(metadata_integer(dim, 1) &&
                         dim.get<int64_t>() <= std::numeric_limits<int64_t>::max() / 4 / elements,
                     "state");
      elements *= dim.get<int64_t>();
    }
    metadata_check(shape[0] == kMaxBatchSize, "state");
    metadata_check(item.at("dtype") == (kind == "recurrent" ? "float32" : "bfloat16"), "state");
    metadata_check(item.at("update") == (kv ? "inplace" : "replace"), "state");
    if (kv)
      metadata_check(shape[2] == m.at("cache_length"), "state");
  }

  const auto &entries = m.at("entries");
  metadata_check(metadata_keys(entries, {"prefill", "decode"}), "entries");
  for (const auto *kind : {"prefill", "decode"}) {
    const auto &entry = entries.at(kind);
    metadata_check(metadata_keys(entry, {"bytes", "sha256"}) &&
                       metadata_integer(entry.at("bytes"), 1) &&
                       metadata_digest(entry.at("sha256")),
                   "entries");
  }

  const auto &weights = m.at("shared_weights");
  metadata_check(metadata_keys(weights, {"identity", "bytes", "sha256"}) &&
                     metadata_integer(weights.at("bytes"), 1) &&
                     metadata_digest(weights.at("identity")) &&
                     metadata_digest(weights.at("sha256")),
                 "shared_weights");
}

struct StateSpec {
  std::string name, kind;
  int64_t layer;
  std::vector<int64_t> shape, stride;
  at::ScalarType dtype;
  bool inplace;
  // Qwen3 returns logits only; Qwen3.5 and Qwen3.8 also return every state.
  std::optional<size_t> output_index;
};

struct SpecBase {
  explicit SpecBase(const fs::path &directory) {
    metadata = read_json(directory / kMetadataFile);
    validate_metadata(metadata);

    model_type = parse_model_type(metadata.at("model_type").get<std::string>());
    capacity = metadata.at("cache_length");
    max_batch_size = metadata.at("state").at(0).at("shape").at(0);
    max_prefill = metadata.at("max_prefill_tokens");
    vocab = metadata.at("vocab_size");
    eos = metadata.at("eos_token_ids").get<std::vector<int64_t>>();
    compute_capability = metadata.at("cuda_compute_capability").get<std::vector<int64_t>>();
    shared_weights = metadata.at("shared_weights");

    const auto version = metadata.at("torch_version").get<std::string>();
    const auto plus = version.find('+');
    require(version.substr(0, plus) == TORCH_VERSION, "Torch version mismatch");
    if (plus != std::string::npos) {
      const auto suffix = version.substr(plus + 1);
      if (!suffix.starts_with("git")) {
        const long cudart = at::Context::versionCUDART();
        const std::string expected =
            "cu" + std::to_string(cudart / 1000) + std::to_string((cudart % 1000) / 10);
        require(suffix == expected, "Torch version mismatch");
      }
    }
    require(metadata.at("cxx11_abi") == bool(_GLIBCXX_USE_CXX11_ABI), "C++ ABI mismatch");

    for (const auto &item : metadata.at("state")) {
      const auto kind = item.at("kind").get<std::string>();
      const auto layer = item.at("layer").get<int64_t>();
      auto shape = item.at("shape").get<std::vector<int64_t>>();
      std::vector<int64_t> stride(shape.size());
      int64_t step = 1;
      for (size_t i = shape.size(); i-- > 0;) {
        stride[i] = step;
        step *= shape[i];
      }

      state.push_back({"layer_" + std::to_string(layer) + "." + kind, kind, layer, std::move(shape),
                       std::move(stride),
                       item.at("dtype") == "float32" ? at::kFloat : at::kBFloat16,
                       item.at("update") == "inplace",
                       (model_type == ModelType::Qwen35 || model_type == ModelType::Qwen38)
                           ? std::optional<size_t>(state.size() + 1)
                           : std::nullopt});
    }
  }

  ModelType model_type;
  int64_t capacity = 0, max_prefill = 0, vocab = 0, max_batch_size = 1;
  std::vector<int64_t> eos, compute_capability;
  Json metadata, shared_weights;
  std::vector<StateSpec> state;

  ValidatedBundle validate_files(const fs::path &directory) const {
    auto bundle = ValidatedBundle::verify(directory, metadata);
    require(model_from_identity(bundle.manifest().at("identity")) == model_type,
            "Package model identity mismatch");
    return bundle;
  }

  void check_token_ids(const std::vector<int64_t> &ids) const {
    for (auto id : ids)
      require(id >= 0 && id < vocab, "Token ID out of range");
  }

  void check_tokens(const std::vector<int64_t> &ids) const {
    require(!ids.empty() && static_cast<int64_t>(ids.size()) <= capacity,
            "Expected 1.." + std::to_string(capacity) + " input tokens");
    check_token_ids(ids);
  }

  int64_t check_batch_tokens(const std::vector<std::vector<int64_t>> &rows) const {
    require(!rows.empty() && static_cast<int64_t>(rows.size()) <= max_batch_size,
            "Batch size outside package range");
    size_t length = 0;
    for (const auto &row : rows) {
      require(row.size() <= static_cast<size_t>(max_prefill), "Too many tokens per compiled call");
      check_token_ids(row);
      length = std::max(length, row.size());
    }
    require(length > 0, "Expected at least one active row");
    return static_cast<int64_t>(length);
  }

  void check_request(const std::vector<int64_t> &ids, int64_t limit) const {
    check_tokens(ids);
    require(limit > 0 && limit <= capacity - static_cast<int64_t>(ids.size()),
            "Generation budget must be positive and prompt + budget must fit the context");
  }
};

inline c10::Device checked_device(const SpecBase &spec) {
  require(torch::cuda::is_available(), "CUDA device 0 is required");
  const auto *properties = at::cuda::getDeviceProperties(0);
  require(spec.compute_capability == std::vector<int64_t>{properties->major, properties->minor},
          "GPU architecture mismatch");
  return {c10::kCUDA, 0};
}

inline c10::Device configured_device(const SpecBase &spec) {
  configure_runtime();
  return checked_device(spec);
}

inline std::vector<int64_t> read_ids(const fs::path &path, const SpecBase &spec) {
  std::ifstream file(path);
  require(file.is_open(), "Cannot open token IDs: " + path.string());

  std::vector<int64_t> ids;
  for (std::string word; file >> word;) {
    ids.push_back(integer(word));
    require(static_cast<int64_t>(ids.size()) <= spec.capacity,
            "Expected 1.." + std::to_string(spec.capacity) + " input tokens");
  }

  require(file.eof(), "Failed to read token IDs: " + path.string());
  spec.check_tokens(ids);
  return ids;
}

} // namespace xinfer
