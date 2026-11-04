#pragma once

#include <ATen/Context.h>
#include <ATen/Parallel.h>

namespace xinfer {

// CPU-side settings shared by runners and standalone native probes.
inline void configure_runtime() {
  at::set_num_threads(1);
  at::globalContext().setAllowTF32CuBLAS(false);
  at::globalContext().setAllowTF32CuDNN(false);
}

} // namespace xinfer
