#include "../../common/decode.h"

#include <stdexcept>

namespace xinfer {

bool megaqwen_decode_available() { return false; }

std::unique_ptr<NativeDecode>
make_megaqwen_decode(const SharedWeights &, const std::vector<at::Tensor> &, int64_t, int64_t) {
  throw std::runtime_error(
      "MegaQwen decode was not built; configure with XINFER_ENABLE_MEGAQWEN=ON");
}

} // namespace xinfer
