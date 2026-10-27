#pragma once
#include <ATen/ATen.h>
#include <nlohmann/json.hpp>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace xinfer {
namespace fs = std::filesystem;
using Json = nlohmann::json;

inline void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

inline int64_t integer(const std::string& text) {
  int64_t value = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  require(result.ec == std::errc{} && result.ptr == text.data() + text.size(),
          "Invalid integer: " + text);
  return value;
}

inline Json read_json(const fs::path& path, const std::string& prefix = "Cannot open metadata: ") {
  std::ifstream file(path);
  require(file.is_open(), prefix + path.string());
  return Json::parse(file);
}

inline std::ofstream output_file(const fs::path& path, bool binary = false) {
  if (path.has_parent_path()) fs::create_directories(path.parent_path());
  std::ofstream file(path, std::ios::out | (binary ? std::ios::binary : std::ios::openmode{}));
  require(file.is_open(), "Cannot open output: " + path.string());
  return file;
}

inline void write_logits(std::ostream& stream, const at::Tensor& logits) {
  const auto row = logits.to(at::kCPU, at::kFloat).contiguous();
  stream.write(reinterpret_cast<const char*>(row.const_data_ptr<float>()), row.numel() * sizeof(float));
  require(bool(stream), "Failed to write logits trace");
}

}  // namespace xinfer
