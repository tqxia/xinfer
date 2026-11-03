#pragma once
#include "attention.h"
#include "io.h"
#include "weights.h"

#include <ATen/Parallel.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/cuda.h>
#include <torch/version.h>

#include <algorithm>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

namespace xinfer {

enum class ModelType { Qwen3, Qwen35 };

inline const char *model_name(ModelType model) {
  return model == ModelType::Qwen3 ? "qwen3" : "qwen35";
}

inline ModelType parse_model_type(const std::string &type) {
  if (type == "qwen3")
    return ModelType::Qwen3;
  if (type == "qwen35")
    return ModelType::Qwen35;
  throw std::runtime_error("Unsupported model type");
}

inline ModelType model_from_identity(const Json &identity) {
  const auto model = identity.at("model").get<std::string>();
  if (model == "qwen3-0.6b")
    return ModelType::Qwen3;
  if (model == "qwen3.5")
    return ModelType::Qwen35;
  throw std::runtime_error("Unsupported model identity: " + model);
}

inline void metadata_check(bool condition, const std::string &field) {
  require(condition, "Unsupported package metadata: " + field);
}

inline bool metadata_equal(const Json &left, const Json &right) {
  // Preserve JSON types: numeric coercion must not equate an ABI integer and float.
  return left.dump() == right.dump();
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
  static const std::vector<std::string> fields{"schema_version",
                                               "model_type",
                                               "entry_point",
                                               "token_range",
                                               "torch_version",
                                               "torch_cuda_version",
                                               "cxx11_abi",
                                               "cuda_compute_capability",
                                               "batch_size",
                                               "dtype",
                                               "cache_length",
                                               "vocab_size",
                                               "eos_token_ids",
                                               "checkpoint",
                                               "state",
                                               "input_order",
                                               "output_order",
                                               "logits",
                                               "input_mutation",
                                               "attention_backend",
                                               "custom_attention",
                                               "cache_input",
                                               "kv_extent",
                                               "export_options",
                                               "tf32",
                                               "triton_enable_fp_fusion",
                                               "inductor_configs",
                                               "package_bytes",
                                               "package_sha256",
                                               "shared_weights",
                                               "model_specific",
                                               "provenance",
                                               "diagnostics"};
  return fields;
}

inline void check_metadata_schema(const Json &metadata) {
  require(metadata.is_object() && metadata.contains("schema_version") &&
              metadata_integer(metadata.at("schema_version"), 1) &&
              metadata.at("schema_version") == 1,
          "Unsupported package metadata schema version; convert retained metadata or re-export "
          "both packages");
}

inline bool metadata_keys(const Json &value, const std::unordered_set<std::string> &keys) {
  if (!value.is_object() || value.size() != keys.size())
    return false;
  for (const auto &[key, _] : value.items())
    if (!keys.contains(key))
      return false;
  return true;
}

inline void validate_metadata(const Json &metadata, const std::string &entry) {
  check_metadata_schema(metadata);
  require(metadata.contains("model_type") && metadata.at("model_type").is_string() &&
              (metadata.at("model_type") == "qwen3" || metadata.at("model_type") == "qwen35"),
          "Unsupported model type");
  const auto &fields = metadata_fields();
  for (const auto &field : fields)
    metadata_check(metadata.contains(field), field);
  for (const auto &[field, _] : metadata.items())
    metadata_check(std::find(fields.begin(), fields.end(), field) != fields.end(), field);
  const auto &m = metadata;
  metadata_check(m.at("entry_point") == entry, "entry_point");
  metadata_check(metadata_integer(m.at("batch_size"), 1) && m.at("batch_size") == 1, "batch_size");
  metadata_check(m.at("dtype") == "bfloat16", "dtype");
  metadata_check(metadata_integer(m.at("cache_length"), 1) && m.at("cache_length") == 1024,
                 "cache_length");
  metadata_check(metadata_integer(m.at("vocab_size"), 1), "vocab_size");
  const Json expected_tokens = entry == "prefill" ? Json::array({2, 512}) : Json::array({1, 1});
  const auto &tokens = m.at("token_range");
  metadata_check(tokens.is_array() && tokens.size() == 2 && metadata_integer(tokens[0], 1) &&
                     metadata_integer(tokens[1], 1) && tokens == expected_tokens,
                 "token_range");
  for (const auto *field : {"torch_version", "torch_cuda_version"})
    metadata_check(m.at(field).is_string() && !m.at(field).get<std::string>().empty(), field);
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
  metadata_check(metadata_keys(checkpoint, {"model_id", "revision", "manifest"}), "checkpoint");
  for (const auto *field : {"model_id", "revision"})
    metadata_check(checkpoint.at(field).is_string() &&
                       !checkpoint.at(field).get<std::string>().empty(),
                   "checkpoint");
  metadata_check(checkpoint.at("manifest").is_object(), "checkpoint");
  metadata_check(m.at("attention_backend") == "flash", "attention_backend");
  const Json attention = {
      {"name", "xinfer::flash_attention"}, {"abi_version", 1}, {"read_only", true}};
  const Json extent = {
      {"device", "cpu"}, {"dtype", "uint8"}, {"shape", "[valid_length + 1]"}, {"data_read", false}};
  metadata_check(metadata_equal(m.at("custom_attention"), attention), "custom_attention");
  metadata_check(m.at("cache_input") == "full_capacity_with_extent", "cache_input");
  metadata_check(m.at("input_mutation") == "kv_only", "input_mutation");
  metadata_check(metadata_equal(m.at("kv_extent"), extent), "kv_extent");
  metadata_check(metadata_equal(m.at("export_options"), Json({{"backed_size_oblivious", false}})),
                 "export_options");
  metadata_check(m.at("tf32").is_boolean() && m.at("tf32") == false, "tf32");
  metadata_check(m.at("triton_enable_fp_fusion").is_boolean(), "triton_enable_fp_fusion");
  metadata_check(m.at("inductor_configs").is_object(), "inductor_configs");
  const auto &states = m.at("state");
  metadata_check(states.is_array() && !states.empty(), "state");
  Json inputs = {"input_ids", "cache_position"}, outputs = {"last_logits"};
  std::unordered_set<std::string> names;
  for (const auto &item : states) {
    metadata_check(
        metadata_keys(item, {"name", "layer", "kind", "shape", "stride", "dtype", "update"}),
        "state");
    metadata_check(item.at("kind").is_string() && metadata_integer(item.at("layer")), "state");
    const auto kind = item.at("kind").get<std::string>();
    const bool kv = kind == "key" || kind == "value";
    metadata_check(kv || kind == "conv" || kind == "recurrent", "state");
    const auto name = "layer_" + std::to_string(item.at("layer").get<int64_t>()) + "." + kind;
    metadata_check(item.at("name") == name && names.insert(name).second, "state");
    const auto &shape = item.at("shape");
    metadata_check(shape.is_array() && shape.size() == (kind == "conv" ? 3 : 4), "state");
    int64_t elements = 1;
    for (const auto &dim : shape) {
      metadata_check(metadata_integer(dim, 1) &&
                         dim.get<int64_t>() <= std::numeric_limits<int64_t>::max() / 4 / elements,
                     "state");
      elements *= dim.get<int64_t>();
    }
    metadata_check(shape[0] == 1, "state");
    const auto &stride = item.at("stride");
    metadata_check(stride.is_array() && stride.size() == shape.size(), "state");
    int64_t step = 1;
    for (size_t i = shape.size(); i-- > 0;) {
      metadata_check(metadata_integer(stride[i], 1) && stride[i].get<int64_t>() == step, "state");
      step *= shape[i].get<int64_t>();
    }
    metadata_check(item.at("dtype") == (kind == "recurrent" ? "float32" : "bfloat16"), "state");
    metadata_check(item.at("update") == (kv ? "inplace" : "replace"), "state");
    if (kv)
      metadata_check(shape[2] == m.at("cache_length"), "state");
    inputs.push_back(name);
    outputs.push_back(name);
  }
  inputs.push_back("kv_extent");
  metadata_check(metadata_equal(m.at("input_order"), inputs), "input_order");
  metadata_check(metadata_equal(m.at("output_order"), Json::array({"last_logits"})) ||
                     metadata_equal(m.at("output_order"), outputs),
                 "output_order");
  metadata_check(metadata_equal(m.at("logits"),
                                Json({{"shape", {1, m.at("vocab_size")}}, {"dtype", "bfloat16"}})),
                 "logits");
  metadata_check(metadata_integer(m.at("package_bytes"), 1), "package_bytes");
  metadata_check(metadata_digest(m.at("package_sha256")), "package_sha256");
  const auto &weights = m.at("shared_weights");
  metadata_check(
      metadata_keys(weights, {"format", "file", "identity", "bytes", "sha256"}) &&
          metadata_integer(weights.at("format"), 1) && weights.at("format") == 2 &&
          weights.at("file") == "weights.pt" && metadata_integer(weights.at("bytes"), 1) &&
          metadata_digest(weights.at("identity")) && metadata_digest(weights.at("sha256")),
      "shared_weights");
  for (const auto *field : {"model_specific", "provenance", "diagnostics"})
    metadata_check(m.at(field).is_object(), field);
}

inline void validate_metadata_pair(const Json &prefill, const Json &decode) {
  check_metadata_schema(prefill);
  check_metadata_schema(decode);
  require(prefill.value("model_type", Json()) == decode.value("model_type", Json()),
          "Package model types differ");
  for (const auto *m : {&prefill, &decode})
    require(m->contains("model_type") && m->at("model_type").is_string() &&
                (m->at("model_type") == "qwen3" || m->at("model_type") == "qwen35"),
            "Unsupported model type");
  static const std::unordered_set<std::string> entry_fields{
      "entry_point", "token_range", "package_bytes", "package_sha256", "diagnostics"};
  for (const auto &field : metadata_fields())
    if (!entry_fields.contains(field))
      require(prefill.contains(field) == decode.contains(field) &&
                  (!prefill.contains(field) || metadata_equal(prefill.at(field), decode.at(field))),
              "Package metadata differs: " + field);
  validate_metadata(prefill, "prefill");
  validate_metadata(decode, "decode");
}

struct StateSpec {
  std::string name, kind;
  int64_t layer;
  std::vector<int64_t> shape, stride;
  at::ScalarType dtype;
  bool inplace;
};

struct SpecBase {
  explicit SpecBase(const fs::path &directory) {
    prefill_metadata = read_json(directory / "prefill.json");
    metadata = read_json(directory / "decode.json");
    validate_metadata_pair(prefill_metadata, metadata);
    model_type = parse_model_type(metadata.at("model_type").get<std::string>());
    capacity = metadata.at("cache_length");
    max_prompt = prefill_metadata.at("token_range").at(1);
    vocab = metadata.at("vocab_size");
    eos = metadata.at("eos_token_ids").get<std::vector<int64_t>>();
    compute_capability = metadata.at("cuda_compute_capability").get<std::vector<int64_t>>();
    shared_weights = metadata.at("shared_weights");
    const auto version = metadata.at("torch_version").get<std::string>();
    require(version.substr(0, version.find('+')) == TORCH_VERSION, "Torch version mismatch");
    require(metadata.at("cxx11_abi") == bool(_GLIBCXX_USE_CXX11_ABI), "C++ ABI mismatch");
    require(attention_ops_abi_version() == 1, "Unsupported native attention ABI");
    for (const auto &item : metadata.at("state"))
      state.push_back({item.at("name").get<std::string>(), item.at("kind").get<std::string>(),
                       item.at("layer").get<int64_t>(),
                       item.at("shape").get<std::vector<int64_t>>(),
                       item.at("stride").get<std::vector<int64_t>>(),
                       item.at("dtype") == "float32" ? at::kFloat : at::kBFloat16,
                       item.at("update") == "inplace"});
  }

  ModelType model_type;
  int64_t capacity = 0, max_prompt = 512, vocab = 0;
  std::vector<int64_t> eos, compute_capability;
  Json prefill_metadata, metadata, shared_weights;
  std::vector<StateSpec> state;

  void validate_files(const fs::path &directory) const {
    for (const auto *entry : {&prefill_metadata, &metadata}) {
      const auto kind = entry->at("entry_point").get<std::string>();
      const auto path = directory / (kind + ".pt2");
      require(fs::is_regular_file(path), "Package file is missing: " + kind);
      require(fs::file_size(path) == entry->at("package_bytes").get<uintmax_t>(),
              "Package file size mismatch");
      require(sha256_file(path) == entry->at("package_sha256").get<std::string>(),
              "Package checksum mismatch: " + kind);
    }
    const auto archive = validate_shared_weights(directory, prefill_metadata, metadata);
    require(model_from_identity(archive.at("identity")) == model_type,
            "Package model identity mismatch");
  }

  void check_tokens(const std::vector<int64_t> &ids) const {
    require(!ids.empty() && static_cast<int64_t>(ids.size()) <= max_prompt,
            "Expected 1..512 input tokens");
    for (auto id : ids)
      require(id >= 0 && id < vocab, "Token ID out of range");
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

inline void configure(const SpecBase &spec) {
  checked_device(spec);
  at::set_num_threads(1);
  at::globalContext().setAllowTF32CuBLAS(false);
  at::globalContext().setAllowTF32CuDNN(false);
}

inline std::vector<int64_t> read_ids(const fs::path &path, const SpecBase &spec) {
  std::ifstream file(path);
  require(file.is_open(), "Cannot open token IDs: " + path.string());
  std::vector<int64_t> ids;
  for (std::string word; file >> word;) {
    ids.push_back(integer(word));
    require(static_cast<int64_t>(ids.size()) <= spec.max_prompt, "Expected 1..512 input tokens");
  }
  require(file.eof(), "Failed to read token IDs: " + path.string());
  spec.check_tokens(ids);
  return ids;
}

} // namespace xinfer
