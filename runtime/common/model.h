#pragma once

#include "../qwen3/runtime.h"
#include "../qwen35/runtime.h"

#include <memory>

namespace xinfer {

// Parse once, then dispatch the shared descriptor to its model-specific ABI checks.
template <class Function> decltype(auto) with_model(const fs::path &packages, Function &&function) {
  const SpecBase metadata(packages);
  switch (metadata.model_type) {
  case ModelType::Qwen3: {
    const qwen::Spec spec(metadata);
    return function(spec, [&](std::optional<int64_t> allocated_batch_size = std::nullopt) {
      return std::make_unique<qwen::Runtime>(packages, spec, allocated_batch_size);
    });
  }
  case ModelType::Qwen35:
  case ModelType::Qwen38: {
    const qwen35::Spec spec(metadata);
    return function(spec, [&](std::optional<int64_t> allocated_batch_size = std::nullopt) {
      return std::make_unique<qwen35::Runtime>(packages, spec, allocated_batch_size);
    });
  }
  }
  throw std::runtime_error("Unsupported model type");
}

} // namespace xinfer
