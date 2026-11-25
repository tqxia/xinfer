# Kernel inner loop

Use this loop inside outer step 2 (**Implement**) when writing a new custom
kernel (e.g., for fusion) or rewriting an existing kernel. Local kernel wins
remain local evidence until verified in the outer model loop.

## Choosing an implementation path in xinfer

Pick the simplest path that matches the measured bottleneck:

- **Triton (`@torch.library.triton_op` / `wrap_triton`)** — best default for
  memory-bound fusions and custom reductions/GEMVs. Packaged automatically by
  AOTI with no extra shared library. See
  `compiler/common/kernels/gemv/projection_gemv.py`,
  `compiler/common/kernels/kv/kv_cache.py`,
  `compiler/common/kernels/gated_delta_net/`, and
  `compiler/common/kernels/gemm/finegrained_fp8.py`.
- **Native CUDA C++ / targeted PTX** — use when instruction-level scheduling,
  custom synchronization, or C++ dispatch beats Triton. Build into `lib*.so`
  and link via `native_libraries` + `custom_op_shims` in
  `compiler/common/compile.py` (following
  `compiler/common/kernels/flash_attention/`).
- For inline PTX or instruction-level claims, verify SM 12.1 toolchain support
  and inspect generated PTX/SASS plus register/shared-memory usage. References:
  [CUDA inline PTX](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html)
  and [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#introduction).

## Before iterating

1. Fix the target operation or complete fused region, its correctness reference,
   and representative inputs — including actual shapes, strides, dtypes, and
   `B=1..8` dynamic-batch / `backed_size_oblivious` contracts where applicable.
   For fusion, benchmark the **complete** original region as the baseline.
2. **Calibrate the local baseline** against the corresponding region in native
   `nsys` (see [profiling.md](profiling.md#baseline-calibration-kernel-inner-loop))
   so local rankings reflect native behavior.

## Inner loop

All revisions stay inside the outer candidate's budget; reserve time for export
and full-model measurement.

1. **Form a local hypothesis.** Use NCU/SASS/roofline evidence to pick one
   focused change to tiling, work partitioning, data/state residence, memory
   layout, scheduling, or instructions. State the predicted counter/timing
   effect and what would refute it.
   - When combining interacting changes, test each viable change alone and
     together (or ablate one from the bundle) on the same baseline.
2. **Edit and compile.** For a brand-new operator or AOTI packaging boundary,
   prove export + AOTI + C++ execution with the first *correct* implementation
   before investing in tuning revisions.
3. **Check correctness.** Verify all outputs against existing tolerances across
   real shapes, tails, strides, aliasing/mutation, and masking. For stateful
   ops, test nonzero carried state, reset, and continuations. Retain failures.
4. **Microbenchmark.** Time the **complete** replacement (including prep,
   copies, dispatch, and workspace). Keep fixture generation, reference checks,
   and state restoration outside timing; separate compilation warmup from
   measurement; match cache/memory conditions; run GPU benchmarks serially.
5. **Analyze and decide.** Compare unprofiled local timing and NCU/SASS findings
   against baseline and prediction, and log the revision in the notebook.
   - **Redesign rule:** if 2–3 local tweaks stall or counters point to a
     structural bottleneck (e.g., CTA count, shared-memory/register ceiling,
     reduction decomposition, data reuse across warps), stop micro-tuning and
     change the algorithm/architecture before another edit.

## Integrating back into the model loop

Bring the local winner back to outer steps 3–6: export, verify the packaged
`.pt2` uses the exact expected kernel and launch settings (re-check NCU if AOTI
or Triton packaging changes launch configs or FP-fusion flags), validate, run
paired C++ benchmarks, and profile native execution.
