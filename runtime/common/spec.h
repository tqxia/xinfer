#pragma once
#include "io.h"
#include "weights.h"

#include <ATen/Parallel.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/cuda.h>
#include <torch/version.h>

#include <algorithm>
#include <initializer_list>
#include <string>
#include <vector>

namespace xinfer {

enum class ModelType { Qwen3, Qwen35 };

inline const char* model_name(ModelType model) {
  return model == ModelType::Qwen3 ? "qwen3" : "qwen35";
}

inline ModelType model_from_identity(const Json& identity) {
  const auto model = identity.at("model").get<std::string>();
  if (model == "qwen3-0.6b") return ModelType::Qwen3;
  if (model == "qwen3.5") return ModelType::Qwen35;
  throw std::runtime_error("Unsupported model identity: " + model);
}

struct StateSpec {
  std::string name, kind;
  std::vector<int64_t> shape;
  at::ScalarType dtype;
  bool inplace;
};

struct SpecBase {
  explicit SpecBase(ModelType model) : model_type(model) {}

  ModelType model_type;
  int64_t capacity = 0, max_prompt = 512, vocab = 0;
  std::vector<int64_t> eos, compute_capability;
  Json shared_weights;
  std::vector<StateSpec> state;

  void check_tokens(const std::vector<int64_t>& ids) const {
    require(!ids.empty() && static_cast<int64_t>(ids.size()) <= max_prompt,
            "Expected 1..512 input tokens");
    for (auto id : ids) require(id >= 0 && id < vocab, "Token ID out of range");
  }

  void check_request(const std::vector<int64_t>& ids, int64_t limit) const {
    check_tokens(ids);
    require(limit > 0 && limit <= capacity - static_cast<int64_t>(ids.size()),
            "Generation budget must be positive and prompt + budget must fit the context");
  }

  void check_eos() const {
    require(!eos.empty(), "Missing EOS tokens");
    for (auto id : eos) require(id >= 0 && id < vocab, "Invalid EOS token");
  }
};

inline void check_pair(const Json& prefill, const Json& decode,
                       std::initializer_list<const char*> keys,
                       const std::string& prefix = "Package metadata differs: ") {
  for (const auto* key : keys)
    require(prefill.at(key) == decode.at(key), prefix + key);
}

inline void check_optional_pair(const Json& prefill, const Json& decode,
                                std::initializer_list<const char*> keys,
                                const std::string& prefix = "Package metadata differs: ") {
  for (const auto* key : keys)
    require(prefill.value(key, Json()) == decode.value(key, Json()), prefix + key);
}

inline void check_model_type(const Json& prefill, const Json& decode, ModelType expected) {
  require(prefill.contains("model_type") == decode.contains("model_type") &&
              prefill.value("model_type", Json()) == decode.value("model_type", Json()),
          "Package model types differ");
  if (decode.contains("model_type"))
    require(decode.at("model_type") == model_name(expected), "Unsupported model type");
}

inline void check_toolchain(const Json& metadata, const char* version_key,
                            const char* version_error = "Torch version mismatch",
                            const char* abi_error = "C++ ABI mismatch") {
  const auto version = metadata.at(version_key).get<std::string>();
  require(version.substr(0, version.find('+')) == TORCH_VERSION, version_error);
  require(metadata.at("cxx11_abi") == bool(_GLIBCXX_USE_CXX11_ABI), abi_error);
}

inline void check_package_files(const fs::path& directory, const Json& prefill,
                                const Json& decode, bool require_sizes, ModelType expected) {
  require(prefill.contains("package_bytes") == decode.contains("package_bytes"),
          "Package size metadata differs");
  require(!require_sizes || decode.contains("package_bytes"), "Package size metadata is required");
  // Older Qwen3 bundles omit sizes; their mandatory archive SHA checks still apply.
  if (decode.contains("package_bytes")) {
    for (const auto* kind : {"prefill", "decode"}) {
      const auto& metadata = std::string(kind) == "prefill" ? prefill : decode;
      require(fs::file_size(directory / (std::string(kind) + ".pt2")) ==
                  metadata.at("package_bytes").get<uintmax_t>(),
              "Package file size mismatch");
    }
  }
  const auto archive = validate_shared_weights(directory, prefill, decode);
  require(model_from_identity(archive.at("identity")) == expected, "Package model identity mismatch");
}

inline c10::Device checked_device(const SpecBase& spec) {
  require(torch::cuda::is_available(), "CUDA device 0 is required");
  const auto* properties = at::cuda::getDeviceProperties(0);
  require(spec.compute_capability == std::vector<int64_t>{properties->major, properties->minor},
          "GPU architecture mismatch");
  return {c10::kCUDA, 0};
}

inline void configure(const SpecBase& spec) {
  checked_device(spec);
  at::set_num_threads(1);
  at::globalContext().setAllowTF32CuBLAS(false);
  at::globalContext().setAllowTF32CuDNN(false);
}

inline std::vector<int64_t> read_ids(const fs::path& path, const SpecBase& spec) {
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

}  // namespace xinfer
