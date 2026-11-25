#pragma once

#include "io.h"
#include "sha256.h"

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/cuda.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace xinfer {

constexpr const char *kSharedWeightsFile = "weights.safetensors";
constexpr const char *kSharedWeightsManifestKey = "xinfer_manifest";
constexpr uint64_t kMaxSafetensorsHeaderBytes = 100ULL << 20;

struct SafetensorsHeader {
  Json header;
  uint64_t data_offset = 0;
  uint64_t file_size = 0;
};

inline uint64_t non_negative_u64(const Json &value, const char *message) {
  require(value.is_number_integer(), message);
  if (value.is_number_unsigned())
    return value.get<uint64_t>();
  const auto signed_value = value.get<int64_t>();
  require(signed_value >= 0, message);
  return static_cast<uint64_t>(signed_value);
}

inline int64_t non_negative_i64(const Json &value, const char *message) {
  const auto raw = non_negative_u64(value, message);
  require(raw <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()), message);
  return static_cast<int64_t>(raw);
}

inline SafetensorsHeader read_safetensors_header(const fs::path &path) {
  const auto file_size = static_cast<uint64_t>(fs::file_size(path));
  require(file_size >= 8, "Truncated safetensors header");

  std::ifstream file(path, std::ios::binary);
  require(file.is_open(), "Cannot open safetensors archive");

  uint8_t prefix[8]{};
  file.read(reinterpret_cast<char *>(prefix), 8);
  require(file.gcount() == 8, "Truncated safetensors header");

  uint64_t header_length = 0;
  for (int i = 0; i < 8; ++i)
    header_length |= static_cast<uint64_t>(prefix[i]) << (8 * i);
  require(header_length > 0 && header_length <= file_size - 8 &&
              header_length <= kMaxSafetensorsHeaderBytes,
          "Invalid safetensors header length");

  std::string text(static_cast<size_t>(header_length), '\0');
  file.read(text.data(), static_cast<std::streamsize>(header_length));
  require(static_cast<uint64_t>(file.gcount()) == header_length, "Truncated safetensors header");

  Json header = Json::parse(text);
  require(header.is_object(), "Safetensors header must be an object");
  return {std::move(header), 8 + header_length, file_size};
}

class MappedFile {
public:
  MappedFile() = default;

  MappedFile(const fs::path &path, uint64_t expected_size) {
    if (expected_size == 0)
      return;

    const int fd = ::open(path.c_str(), O_RDONLY);
    require(fd >= 0, "Cannot open shared weights archive: " + path.string());
    struct stat info{};
    if (::fstat(fd, &info) != 0 || info.st_size < 0 ||
        static_cast<uint64_t>(info.st_size) != expected_size) {
      ::close(fd);
      throw std::runtime_error("Shared weights archive size changed during load");
    }

    void *mapped =
        ::mmap(nullptr, static_cast<size_t>(expected_size), PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    require(mapped != MAP_FAILED, "Cannot map shared weights archive: " + path.string());
    data_ = static_cast<const uint8_t *>(mapped);
    size_ = static_cast<size_t>(expected_size);
  }

  MappedFile(const MappedFile &) = delete;
  MappedFile &operator=(const MappedFile &) = delete;

  ~MappedFile() {
    if (data_)
      ::munmap(const_cast<uint8_t *>(data_), size_);
  }

  const uint8_t *data() const { return data_; }

private:
  const uint8_t *data_ = nullptr;
  size_t size_ = 0;
};

// A load-scoped proof: only verification can bind a manifest to package paths.
class ValidatedBundle {
public:
  ValidatedBundle(const ValidatedBundle &) = delete;
  ValidatedBundle &operator=(const ValidatedBundle &) = delete;
  ValidatedBundle(ValidatedBundle &&) = default;
  ValidatedBundle &operator=(ValidatedBundle &&) = default;

  static ValidatedBundle verify(const fs::path &directory, const Json &metadata) {
    const auto root = fs::canonical(directory);
    require(metadata.is_object() && metadata.contains("entries") &&
                metadata.at("entries").is_object(),
            "Unsupported package metadata: entries");
    const auto &declared_entries = metadata.at("entries");
    std::unordered_map<std::string, std::string> package_digests;
    for (const auto *kind : {"prefill", "decode"}) {
      require(declared_entries.contains(kind) && declared_entries.at(kind).is_object(),
              "Unsupported package metadata: entries");
      const auto &entry = declared_entries.at(kind);
      const auto path = root / (std::string(kind) + ".pt2");
      require(fs::is_regular_file(path), std::string("Package file is missing: ") + kind);
      require(fs::file_size(path) == entry.at("bytes").get<uintmax_t>(),
              "Package file size mismatch");
      const auto digest = sha256_file(path);
      require(digest == entry.at("sha256").get<std::string>(),
              std::string("Package checksum mismatch: ") + kind);
      package_digests.emplace(kind, digest);
    }

    require(metadata.contains("shared_weights"),
            "Shared weights metadata is required; re-export both packages");
    const auto &contract = metadata.at("shared_weights");
    require(contract.is_object() && contract.size() == 3 && contract.contains("identity") &&
                contract.at("identity").is_string() && contract.contains("bytes") &&
                contract.at("bytes").is_number_integer() &&
                contract.at("bytes").get<int64_t>() >= 0 && contract.contains("sha256") &&
                contract.at("sha256").is_string(),
            "Shared weights metadata is required; re-export both packages");

    const auto path = root / kSharedWeightsFile;
    require(fs::is_regular_file(path), "Shared weights archive is missing");
    require(fs::file_size(path) == contract.at("bytes").get<uintmax_t>() &&
                sha256_file(path) == contract.at("sha256").get<std::string>(),
            "Shared weights archive checksum mismatch");

    SafetensorsHeader safetensors;
    Json data;
    try {
      safetensors = read_safetensors_header(path);
      require(safetensors.header.contains("__metadata__") &&
                  safetensors.header.at("__metadata__").is_object(),
              "Shared weights archive metadata is missing");
      const auto &meta = safetensors.header.at("__metadata__");
      require(meta.contains(kSharedWeightsManifestKey) &&
                  meta.at(kSharedWeightsManifestKey).is_string(),
              "Shared weights archive metadata is missing");
      data = Json::parse(meta.at(kSharedWeightsManifestKey).get<std::string>());
      require(data.is_object() && data.size() == 6 && data.contains("identity") &&
                  data.contains("identity_sha256") && data.contains("entries") &&
                  data.contains("sources") && data.contains("storages") && data.contains("tensors"),
              "Shared weights archive metadata has invalid fields");
    } catch (const std::exception &error) {
      throw std::runtime_error(std::string("Shared weights archive metadata is invalid: ") +
                               error.what());
    }
    require(data.at("identity_sha256") == contract.at("identity"),
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

    return ValidatedBundle(root, contract, std::move(data), std::move(safetensors));
  }

  const Json &manifest() const { return manifest_; }

private:
  friend class SharedWeights;

  ValidatedBundle(fs::path directory, Json contract, Json manifest, SafetensorsHeader safetensors)
      : directory_(std::move(directory)), contract_(std::move(contract)),
        manifest_(std::move(manifest)), safetensors_(std::move(safetensors)) {}

  fs::path directory_;
  Json contract_, manifest_;
  SafetensorsHeader safetensors_;
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

inline const char *safetensors_dtype_name(at::ScalarType dtype) {
  switch (dtype) {
  case at::kBFloat16:
    return "BF16";
  case at::kHalf:
    return "F16";
  case at::kFloat:
    return "F32";
  case at::kDouble:
    return "F64";
  case at::kLong:
    return "I64";
  case at::kInt:
    return "I32";
  case at::kShort:
    return "I16";
  case at::kChar:
    return "I8";
  case at::kByte:
    return "U8";
  case at::kBool:
    return "BOOL";
  default:
    throw std::runtime_error("Shared weights dtype is unsupported");
  }
}

inline void check_view_bounds(const std::vector<int64_t> &shape, const std::vector<int64_t> &stride,
                              int64_t offset, int64_t numel) {
  require(shape.size() == stride.size() && offset >= 0 &&
              std::all_of(shape.begin(), shape.end(), [](auto n) { return n >= 0; }) &&
              std::all_of(stride.begin(), stride.end(), [](auto n) { return n >= 0; }),
          "Shared weights tensor layout is invalid");
  if (std::any_of(shape.begin(), shape.end(), [](auto n) { return n == 0; })) {
    require(offset <= numel, "Shared weights tensor layout is invalid");
    return;
  }

  int64_t max_index = offset;
  for (size_t i = 0; i < shape.size(); ++i) {
    const int64_t span = shape[i] - 1;
    if (span == 0 || stride[i] == 0)
      continue;
    require(stride[i] <= (std::numeric_limits<int64_t>::max() - max_index) / span,
            "Shared weights tensor layout is invalid");
    max_index += span * stride[i];
  }
  require(max_index < numel, "Shared weights tensor layout is invalid");
}

class SharedWeights {
public:
  SharedWeights(ValidatedBundle bundle, const Json &contract) : bundle_(std::move(bundle)) {
    require(bundle_.contract_ == contract, "Shared weights contracts differ");

    const auto &header = bundle_.safetensors_.header;
    const auto &manifest_storages = bundle_.manifest_.at("storages");
    const auto &manifest_tensors = bundle_.manifest_.at("tensors");
    require(manifest_storages.is_object() && manifest_tensors.is_object(),
            "Shared weights archive metadata is invalid");

    size_t header_tensors = 0;
    for (const auto &[key, _] : header.items())
      if (key != "__metadata__")
        ++header_tensors;
    require(header_tensors == manifest_storages.size(),
            "Shared weights archive buffer set mismatch");

    struct StorageEntry {
      at::ScalarType dtype;
      bool cuda;
      int64_t numel;
      uint64_t begin, end;
    };
    const uint64_t payload_size = bundle_.safetensors_.file_size - bundle_.safetensors_.data_offset;
    std::unordered_map<std::string, StorageEntry> entries;
    entries.reserve(manifest_storages.size());

    // Validate every storage entry and tensor view before mapping or allocating device memory.
    for (const auto &[name, descriptor] : manifest_storages.items()) {
      require(header.contains(name) && header.at(name).is_object(),
              "Shared weights archive buffer set mismatch");
      const auto device = descriptor.at("device").get<std::string>();
      require(device == "cpu" || device == "cuda",
              "Shared weights constant has unsupported device");
      const auto dtype = shared_weights_dtype(descriptor.at("dtype").get<std::string>());
      const auto expected_bytes = non_negative_u64(
          descriptor.at("bytes"), "Shared weights archive tensor metadata mismatch");

      const auto &item = header.at(name);
      require(item.contains("dtype") && item.at("dtype").is_string() &&
                  item.at("dtype").get<std::string>() == safetensors_dtype_name(dtype),
              "Shared weights archive tensor metadata mismatch");
      require(item.contains("shape") && item.at("shape").is_array() && item.at("shape").size() == 1,
              "Shared weights archive tensor metadata mismatch");
      const auto numel = non_negative_i64(item.at("shape").at(0),
                                          "Shared weights archive tensor metadata mismatch");
      const auto elem_size = static_cast<uint64_t>(c10::elementSize(dtype));
      require(expected_bytes <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) &&
                  static_cast<uint64_t>(numel) <=
                      std::numeric_limits<uint64_t>::max() / elem_size &&
                  static_cast<uint64_t>(numel) * elem_size == expected_bytes,
              "Shared weights archive tensor metadata mismatch");

      require(item.contains("data_offsets") && item.at("data_offsets").is_array() &&
                  item.at("data_offsets").size() == 2,
              "Shared weights archive tensor metadata mismatch");
      const auto begin = non_negative_u64(item.at("data_offsets").at(0),
                                          "Shared weights archive tensor metadata mismatch");
      const auto end = non_negative_u64(item.at("data_offsets").at(1),
                                        "Shared weights archive tensor metadata mismatch");
      require(begin <= end && end <= payload_size && end - begin == expected_bytes,
              "Shared weights archive tensor metadata mismatch");

      entries.emplace(name, StorageEntry{dtype, device == "cuda", numel, begin, end});
    }

    for (const auto &[name, descriptor] : manifest_tensors.items()) {
      const auto storage = descriptor.at("storage").get<std::string>();
      require(entries.contains(storage) &&
                  descriptor.at("dtype") == manifest_storages.at(storage).at("dtype") &&
                  descriptor.at("device") == manifest_storages.at(storage).at("device"),
              "Shared weights tensor storage metadata mismatch");
      check_view_bounds(descriptor.at("shape").get<std::vector<int64_t>>(),
                        descriptor.at("stride").get<std::vector<int64_t>>(),
                        descriptor.at("offset").get<int64_t>(), entries.at(storage).numel);
    }

    const bool need_map = payload_size > 0 && !entries.empty();
    MappedFile mapped(bundle_.directory_ / kSharedWeightsFile,
                      need_map ? bundle_.safetensors_.file_size : 0);
    bool copied_cuda = false;
    for (const auto &[name, entry] : entries) {
      const auto cpu_options = at::TensorOptions().device(at::kCPU).dtype(entry.dtype);
      if (entry.numel == 0) {
        storages_.emplace(name,
                          at::empty({0}, entry.cuda ? cpu_options.device(c10::Device(at::kCUDA, 0))
                                                    : cpu_options));
        continue;
      }

      auto *base =
          const_cast<uint8_t *>(mapped.data() + bundle_.safetensors_.data_offset + entry.begin);
      const auto nbytes = static_cast<int64_t>(entry.end - entry.begin);
      if (entry.cuda) {
        auto dst = at::empty({entry.numel}, cpu_options.device(c10::Device(at::kCUDA, 0)));
        const auto byte_cpu = at::TensorOptions().device(at::kCPU).dtype(at::kByte);
        const auto byte_cuda = byte_cpu.device(dst.device());
        at::from_blob(dst.data_ptr(), {nbytes}, byte_cuda)
            .copy_(at::from_blob(base, {nbytes}, byte_cpu));
        storages_.emplace(name, std::move(dst));
        copied_cuda = true;
      } else {
        auto dst = at::empty({entry.numel}, cpu_options);
        std::memcpy(dst.data_ptr(), base, static_cast<size_t>(nbytes));
        storages_.emplace(name, std::move(dst));
      }
    }
    // Complete all H2D copies before unmapping the backing file pages.
    if (copied_cuda)
      torch::cuda::synchronize();

    for (const auto &[name, descriptor] : manifest_tensors.items()) {
      const auto shape = descriptor.at("shape").get<std::vector<int64_t>>();
      const auto stride = descriptor.at("stride").get<std::vector<int64_t>>();
      const auto offset = descriptor.at("offset").get<int64_t>();
      const auto &storage = storages_.at(descriptor.at("storage").get<std::string>());
      tensors_.emplace(name, storage.as_strided(shape, stride, offset));
    }
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
