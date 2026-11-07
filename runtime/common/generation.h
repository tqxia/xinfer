#pragma once
#include "io.h"
#include <algorithm>
#include <vector>

namespace xinfer {

// Request length and compiled-call length are separate contracts. A final
// singleton goes through step()'s decode path, with all earlier state carried.
template <class Runtime>
auto prefill(Runtime &runtime, const std::vector<int64_t> &prompt, int64_t max_prefill) {
  require(!prompt.empty() && max_prefill >= 2, "Invalid prompt partition");
  decltype(runtime.step(prompt)) logits{};
  for (size_t start = 0; start < prompt.size();) {
    const auto count = std::min(prompt.size() - start, static_cast<size_t>(max_prefill));
    logits =
        runtime.step(std::vector<int64_t>(prompt.begin() + start, prompt.begin() + start + count));
    start += count;
  }
  return logits;
}

// The model validates the request and owns all state inside reset()/step().
template <class Runtime>
std::vector<int64_t> generate(Runtime &runtime, const std::vector<int64_t> &prompt, int64_t limit,
                              const std::vector<int64_t> &eos, std::ostream *trace = nullptr) {
  runtime.reset();
  auto logits = runtime.prefill(prompt);
  std::vector<int64_t> generated;
  for (int64_t i = 0; i < limit; ++i) {
    if (trace)
      write_logits(*trace, logits);
    const auto token = logits.argmax(-1).template item<int64_t>();
    generated.push_back(token);
    if (std::find(eos.begin(), eos.end(), token) != eos.end() || i + 1 == limit)
      break;
    logits = runtime.step({token});
  }
  return generated;
}

} // namespace xinfer
