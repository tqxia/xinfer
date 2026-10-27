#pragma once
#include "io.h"
#include <algorithm>
#include <vector>

namespace xinfer {

// The model validates the request and owns all state inside reset()/step().
template <class Runtime>
std::vector<int64_t> generate(Runtime& runtime, const std::vector<int64_t>& prompt,
                              int64_t limit, const std::vector<int64_t>& eos,
                              std::ostream* trace = nullptr) {
  runtime.reset();
  auto logits = runtime.step(prompt);
  std::vector<int64_t> generated;
  for (int64_t i = 0; i < limit; ++i) {
    if (trace) write_logits(*trace, logits);
    const auto token = logits.argmax(-1).template item<int64_t>();
    generated.push_back(token);
    if (std::find(eos.begin(), eos.end(), token) != eos.end() || i + 1 == limit) break;
    logits = runtime.step({token});
  }
  return generated;
}

}  // namespace xinfer
