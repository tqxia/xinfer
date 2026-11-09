#pragma once
#include "io.h"
#include "sha256.h"

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/script.h>
#include <caffe2/serialize/inline_container.h>

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace xinfer {

// Optional synchronous load diagnostics; callbacks are never retained by owners.
using LoadObserver = std::function<void(const char *)>;

// A load-scoped proof: only verification can bind a manifest to package paths.
class ValidatedBundle {
public:
  ValidatedBundle(const ValidatedBundle &) = delete;
  ValidatedBundle &operator=(const ValidatedBundle &) = delete;
  ValidatedBundle(ValidatedBundle &&) = default;
  ValidatedBundle &operator=(ValidatedBundle &&) = default;

  static ValidatedBundle verify(const fs::path &directory, const Json &prefill,
                                const Json &decode) {
    const auto root = fs::canonical(directory);
    std::unordered_map<std::string, std::string> package_digests;
    for (const auto *entry : {&prefill, &decode}) {
      const auto kind = entry->at("entry_point").get<std::string>();
      const auto path = root / (kind + ".pt2");
      require(fs::is_regular_file(path), "Package file is missing: " + kind);
      require(fs::file_size(path) == entry->at("package_bytes").get<uintmax_t>(),
              "Package file size mismatch");
      const auto digest = sha256_file(path);
      require(digest == entry->at("package_sha256").get<std::string>(),
              "Package checksum mismatch: " + kind);
      package_digests.emplace(kind, digest);
    }
    require(prefill.contains("shared_weights") && decode.contains("shared_weights"),
            "Shared weights metadata is required; re-export both packages");
    require(prefill.at("shared_weights") == decode.at("shared_weights"),
            "Shared weights contracts differ");
    for (const auto *metadata : {&prefill, &decode}) {
      const auto &options = metadata->at("inductor_configs");
      require(options.contains("aot_inductor.package_constants_in_so") &&
                  options.at("aot_inductor.package_constants_in_so") == false &&
                  options.contains("always_keep_tensor_constants") &&
                  options.at("always_keep_tensor_constants") == true,
              "Shared weights require weightless package compiler options");
    }
    const auto &contract = decode.at("shared_weights");
    require(contract.is_object() && contract.value("format", 0) == 2 &&
                contract.value("file", std::string()) == "weights.pt" &&
                contract.contains("identity") && contract.at("identity").is_string() &&
                contract.contains("bytes") && contract.at("bytes").is_number_integer() &&
                contract.contains("sha256") && contract.at("sha256").is_string(),
            "Shared weights metadata is required; re-export both packages");
    const auto path = root / "weights.pt";
    require(fs::is_regular_file(path), "Shared weights archive is missing");
    require(fs::file_size(path) == contract.at("bytes").get<uintmax_t>() &&
                sha256_file(path) == contract.at("sha256").get<std::string>(),
            "Shared weights archive checksum mismatch");
    Json data;
    try {
      caffe2::serialize::PyTorchStreamReader reader(path.string());
      const auto record = reader.getRecord("extra/weights.json");
      const auto *text = static_cast<const char *>(std::get<0>(record).get());
      data = Json::parse(text, text + std::get<1>(record));
    } catch (const std::exception &error) {
      throw std::runtime_error(std::string("Shared weights archive metadata is invalid: ") +
                               error.what());
    }
    require(data.is_object() && data.value("format", 0) == 2 &&
                data.at("identity_sha256") == contract.at("identity"),
            "Shared weights identity mismatch");
    const auto &entries = data.at("entries");
    require(entries.is_object() && entries.size() == 2 && entries.contains("prefill") &&
                entries.contains("decode"),
            "Shared weights manifest requires both package entries");
    for (const auto *kind : {"prefill", "decode"}) {
      require(package_digests.at(kind) == entries.at(kind).at("package_sha256").get<std::string>(),
              std::string("Shared weights package checksum mismatch: ") + kind);
      require(entries.at(kind).at("constants").is_object(),
              "Shared weights package mapping is invalid");
      for (const auto &name : entries.at(kind).at("constants"))
        require(data.at("tensors").contains(name.get<std::string>()),
                "Shared weights constant mapping is incomplete");
    }
    return ValidatedBundle(root, contract, std::move(data));
  }

  const Json &manifest() const { return manifest_; }

private:
  friend class SharedWeights;
  ValidatedBundle(fs::path directory, Json contract, Json manifest)
      : directory_(std::move(directory)), contract_(std::move(contract)),
        manifest_(std::move(manifest)) {}

  fs::path directory_;
  Json contract_, manifest_;
};

inline at::ScalarType shared_weights_dtype(const std::string &name) {
  if (name == "bfloat16")
    return at::kBFloat16;
  if (name == "float16")
    return at::kHalf;
  if (name == "float32")
    return at::kFloat;
  if (name == "float64")
    return at::kDouble;
  if (name == "float8_e4m3fn")
    return at::kFloat8_e4m3fn;
  if (name == "int64")
    return at::kLong;
  if (name == "int32")
    return at::kInt;
  if (name == "int16")
    return at::kShort;
  if (name == "int8")
    return at::kChar;
  if (name == "uint8")
    return at::kByte;
  if (name == "bool")
    return at::kBool;
  throw std::runtime_error("Shared weights dtype is unsupported: " + name);
}

class SharedWeights {
public:
  SharedWeights(ValidatedBundle bundle, const Json &contract, LoadObserver observer = {})
      : bundle_(std::move(bundle)) {
    require(bundle_.contract_ == contract, "Shared weights contracts differ");
    auto archive =
        torch::jit::load((bundle_.directory_ / "weights.pt").string(), c10::Device(at::kCPU));
    require(archive.attr("identity").toStringRef() == contract.at("identity").get<std::string>(),
            "Shared weights archive identity mismatch");
    std::unordered_map<std::string, at::Tensor> cpu;
    for (const auto &buffer : archive.named_buffers())
      cpu.emplace(buffer.name, buffer.value);
    require(cpu.size() == bundle_.manifest_.at("storages").size(),
            "Shared weights archive buffer set mismatch");
    // Validate every CPU buffer before allocating the canonical device pool.
    for (const auto &[name, descriptor] : bundle_.manifest_.at("storages").items()) {
      require(cpu.contains(name), "Shared weights archive buffer set mismatch");
      const auto device = descriptor.at("device").get<std::string>();
      require(device == "cpu" || device == "cuda",
              "Shared weights constant has unsupported device");
      const auto &tensor = cpu.at(name);
      require(tensor.device().is_cpu() && tensor.dim() == 1 && tensor.is_contiguous() &&
                  tensor.scalar_type() == shared_weights_dtype(descriptor.at("dtype")) &&
                  tensor.nbytes() == descriptor.at("bytes").get<size_t>(),
              "Shared weights archive tensor metadata mismatch");
    }
    for (const auto &[name, descriptor] : bundle_.manifest_.at("tensors").items()) {
      const auto storage = descriptor.at("storage").get<std::string>();
      require(
          cpu.contains(storage) &&
              descriptor.at("dtype") == bundle_.manifest_.at("storages").at(storage).at("dtype") &&
              descriptor.at("device") == bundle_.manifest_.at("storages").at(storage).at("device"),
          "Shared weights tensor storage metadata mismatch");
      // ATen checks the full strided view bounds while all storage is on CPU.
      cpu.at(storage).as_strided(descriptor.at("shape").get<std::vector<int64_t>>(),
                                 descriptor.at("stride").get<std::vector<int64_t>>(),
                                 descriptor.at("offset").get<int64_t>());
    }
    if (observer)
      observer("cpu_archive_validated");
    for (const auto &[name, descriptor] : bundle_.manifest_.at("storages").items()) {
      const auto type = descriptor.at("device").get<std::string>();
      storages_.emplace(name,
                        cpu.at(name).to(c10::Device(type == "cuda" ? at::kCUDA : at::kCPU, 0)));
    }
    for (const auto &[name, descriptor] : bundle_.manifest_.at("tensors").items()) {
      const auto shape = descriptor.at("shape").get<std::vector<int64_t>>();
      const auto stride = descriptor.at("stride").get<std::vector<int64_t>>();
      const auto offset = descriptor.at("offset").get<int64_t>();
      require(shape.size() == stride.size() && offset >= 0 &&
                  std::all_of(shape.begin(), shape.end(), [](auto n) { return n >= 0; }) &&
                  std::all_of(stride.begin(), stride.end(), [](auto n) { return n >= 0; }),
              "Shared weights tensor layout is invalid");
      const auto &storage = storages_.at(descriptor.at("storage").get<std::string>());
      tensors_.emplace(name, storage.as_strided(shape, stride, offset));
    }
    if (observer)
      observer("canonical_pool_allocated_cpu_archive_alive");
  }

  fs::path package_path(const std::string &kind) const {
    require(kind == "prefill" || kind == "decode", "Unknown package entry: " + kind);
    return bundle_.directory_ / (kind + ".pt2");
  }

  void load(torch::inductor::AOTIModelPackageLoader &loader, const std::string &kind) {
    const auto &mapping = bundle_.manifest_.at("entries").at(kind).at("constants");
    const auto names = loader.get_constant_fqns();
    require(names.size() == mapping.size() &&
                std::unordered_set<std::string>(names.begin(), names.end()).size() == names.size(),
            "Shared weights constant FQNs differ: " + kind);
    std::unordered_map<std::string, at::Tensor> constants;
    for (const auto &name : names) {
      require(mapping.contains(name), "Shared weights constant FQNs differ: " + kind);
      constants.emplace(name, tensors_.at(mapping.at(name).get<std::string>()));
    }
    loader.load_constants(constants, false, true, true);
    const auto actual = loader.get_runner()->extract_constants_map(false);
    require(actual.size() == constants.size(), "Shared weights native constant map is incomplete");
    for (const auto &[name, value] : actual) {
      const auto &expected = constants.at(name);
      require(value.data_ptr() == expected.data_ptr() && value.sizes() == expected.sizes() &&
                  value.strides() == expected.strides() &&
                  value.scalar_type() == expected.scalar_type() &&
                  value.device() == expected.device(),
              "Shared weights native constant does not alias the retained pool");
    }
  }

private:
  ValidatedBundle bundle_;
  std::unordered_map<std::string, at::Tensor> storages_, tensors_;
};

} // namespace xinfer
