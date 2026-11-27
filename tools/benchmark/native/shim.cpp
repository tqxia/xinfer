// Tool-only CUDA allocator stats and thread-count helpers for xinfer_benchmark
// (statically linked only into xinfer_benchmark; never exposed in xinfer_aoti).
#include <ATen/Parallel.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <cstdint>

struct XinferCudaAllocStats {
  uint64_t allocated_bytes;
  uint64_t reserved_bytes;
  uint64_t peak_allocated_bytes;
  uint64_t peak_reserved_bytes;
};

// Reads aggregate (small + large pool) allocator stats on CUDA device 0.
// Callers must initialize CUDA on device 0 before calling this function.
extern "C" void xinfer_bench_cuda_alloc_stats(XinferCudaAllocStats *out) noexcept {
  const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
  const auto all = static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
  out->allocated_bytes = static_cast<uint64_t>(stats.allocated_bytes[all].current);
  out->reserved_bytes = static_cast<uint64_t>(stats.reserved_bytes[all].current);
  out->peak_allocated_bytes = static_cast<uint64_t>(stats.allocated_bytes[all].peak);
  out->peak_reserved_bytes = static_cast<uint64_t>(stats.reserved_bytes[all].peak);
}

extern "C" void xinfer_bench_cuda_reset_peak_stats(void) noexcept {
  c10::cuda::CUDACachingAllocator::resetPeakStats(0);
}

extern "C" int xinfer_bench_get_num_threads(void) noexcept { return at::get_num_threads(); }
