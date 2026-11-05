#pragma once
#include "../qwen3/runtime.h"
#include "../qwen35/runtime.h"

#include <memory>

namespace xinfer {

// Parse once, then dispatch the shared descriptor to its model-specific ABI checks.
template <class Function>
decltype(auto) with_model(const fs::path &packages, Function &&function,
                          DecodeBackendKind backend = DecodeBackendKind::Aoti) {
  const SpecBase metadata(packages);
  switch (metadata.model_type) {
  case ModelType::Qwen3: {
    const qwen::Spec spec(metadata);
    return function(spec, [&] { return std::make_unique<qwen::Runtime>(packages, spec, backend); });
  }
  case ModelType::Qwen35: {
    require(backend == DecodeBackendKind::Aoti, "MegaQwen decode requires Qwen3-0.6B");
    const qwen35::Spec spec(metadata);
    return function(spec, [&] { return std::make_unique<qwen35::Runtime>(packages, spec); });
  }
  }
  throw std::runtime_error("Unsupported model type");
}

} // namespace xinfer
