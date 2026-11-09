#pragma once
#include "../qwen3/runtime.h"
#include "../qwen35/runtime.h"
#include "../qwen36/runtime.h"

#include <memory>

namespace xinfer {

// Parse once, then dispatch the shared descriptor to its model-specific ABI checks.
template <class Function> decltype(auto) with_model(const fs::path &packages, Function &&function) {
  const SpecBase metadata(packages);
  switch (metadata.model_type) {
  case ModelType::Qwen3: {
    const qwen::Spec spec(metadata);
    return function(spec, [&] { return std::make_unique<qwen::Runtime>(packages, spec); });
  }
  case ModelType::Qwen35: {
    const qwen35::Spec spec(metadata);
    return function(spec, [&] { return std::make_unique<qwen35::Runtime>(packages, spec); });
  }
  case ModelType::Qwen36: {
    const qwen36::Spec spec(metadata);
    return function(spec, [&] { return std::make_unique<qwen36::Runtime>(packages, spec); });
  }
  }
  throw std::runtime_error("Unsupported model type");
}

} // namespace xinfer
