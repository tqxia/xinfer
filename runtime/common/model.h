#pragma once
#include "../qwen3/runtime.h"
#include "../qwen35/runtime.h"

#include <caffe2/serialize/inline_container.h>
#include <memory>

namespace xinfer {

inline ModelType parse_model_type(const std::string& type) {
  if (type == "qwen3") return ModelType::Qwen3;
  if (type == "qwen35") return ModelType::Qwen35;
  throw std::runtime_error("Unsupported model type: " + type);
}

inline ModelType select_model(const fs::path& packages) {
  const auto prefill = read_json(packages / "prefill.json");
  const auto decode = read_json(packages / "decode.json");
  require(prefill.contains("model_type") == decode.contains("model_type"), "Package model types differ");
  if (decode.contains("model_type")) {
    require(prefill.at("model_type") == decode.at("model_type"), "Package model types differ");
    return parse_model_type(decode.at("model_type").get<std::string>());
  }
  // Format-2 bundles already carry an explicit model identity. This peek only
  // selects the adapter; Spec validates archive/package checksums before CUDA.
  const auto archive = packages / "weights.pt";
  require(fs::is_regular_file(archive), "Shared weights archive is missing");
  Json identity;
  try {
    caffe2::serialize::PyTorchStreamReader reader(archive.string());
    const auto record = reader.getRecord("extra/weights.json");
    const auto* data = static_cast<const char*>(std::get<0>(record).get());
    identity = Json::parse(data, data + std::get<1>(record)).at("identity");
  } catch (const std::exception& error) {
    throw std::runtime_error(std::string("Shared weights archive metadata is invalid: ") + error.what());
  }
  return model_from_identity(identity);
}

// Dispatch once around the whole operation, keeping model policy out of callers.
template <class Function>
decltype(auto) with_model(const fs::path& packages, Function&& function) {
  switch (select_model(packages)) {
    case ModelType::Qwen3: {
      const qwen::Spec spec(packages);
      return function(spec, [&] { return std::make_unique<qwen::Runtime>(packages, spec); });
    }
    case ModelType::Qwen35: {
      const qwen35::Spec spec(packages);
      return function(spec, [&] { return std::make_unique<qwen35::Runtime>(packages, spec); });
    }
  }
  throw std::runtime_error("Unsupported model type");
}

}  // namespace xinfer
