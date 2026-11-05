#pragma once
#include "io.h"

#include <memory>
#include <vector>

namespace xinfer {

enum class DecodeBackendKind { Aoti, MegaQwen };

inline const char *decode_backend_name(DecodeBackendKind kind) {
  return kind == DecodeBackendKind::Aoti ? "aoti" : "megaqwen";
}

inline DecodeBackendKind parse_decode_backend(const std::string &name) {
  if (name == "aoti")
    return DecodeBackendKind::Aoti;
  if (name == "megaqwen")
    return DecodeBackendKind::MegaQwen;
  throw std::runtime_error("Unknown decode backend: " + name);
}

// A native decode implementation borrows the runner's verified weights and KV
// tensors. The runner remains the sole owner of token position and reset rules.
class NativeDecode {
public:
  virtual ~NativeDecode() = default;
  virtual at::Tensor run(int64_t token, int64_t position) = 0;
  virtual Json info() const = 0;
};

class SharedWeights;
bool megaqwen_decode_available();
std::unique_ptr<NativeDecode> make_megaqwen_decode(const SharedWeights &weights,
                                                   const std::vector<at::Tensor> &state,
                                                   int64_t capacity, int64_t vocab);

} // namespace xinfer
