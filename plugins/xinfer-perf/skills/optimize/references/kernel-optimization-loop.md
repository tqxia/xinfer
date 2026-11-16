# Kernel optimization loop

Improve the implementation of a required operation through compiler/library
selection, tuning, layout/scheduling changes or rewriting. Use the observed cost
to choose a focused experiment, then iterate from its measured result.

## Implementation choices

Choose among Triton, CUDA C++, targeted PTX and existing library kernels using
the measured bottleneck, integration effort and target support. Consult the
[hardware guidance](hardware.md#feature-dependent-mechanisms) when a hypothesis
depends on architecture-specific instructions or resources.

Use CUDA intrinsics or inline PTX when the evidence identifies a specific
instruction-level opportunity. Check the target SM and toolchain, inspect
generated PTX/SASS and resource use, and validate numerical and memory behavior.
See NVIDIA's
[inline PTX guide](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html)
and
[PTX ISA introduction](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#introduction).

## Kernel inner loop

Use this inside outer step 2, **Implement the change**, when rewriting an
existing kernel or building a custom kernel for fusion. Ordinary compiler-option
comparisons can stay in the outer loop.

Define the operator or fused-region boundary, reference and baseline once. Reuse
existing component fixtures and declared tolerances. Cover actual shapes,
strides, dtypes, representative values and supported boundaries, including
nonzero/carried state for stateful operators. Use captured values for
data-dependent execution. A fusion baseline is the complete original region.
Keep the local baseline and reference intact while revising the implementation.

**Calibrate the baseline before choosing kernel revisions.** Establish
unprofiled local timings and a matching NCU baseline capture, then compare local
GPU time with the corresponding operation or complete region in native NSYS.
Follow [baseline calibration](profiling.md#baseline-calibration) to match the
comparison and explain material discrepancies before using local gains to rank
revisions. Reuse this calibration while the baseline and measurement conditions
remain unchanged.

1. **Form a local hypothesis.** Use the NCU evidence and generated code to
   choose a change in tiling, layout, scheduling, instruction use or
   implementation. Apply the brief
   [candidate comparison](experiment-loop.md#1-form-the-hypothesis) to plausible
   revisions within this region. Predict the affected counters and timing, and
   what would refute the mechanism.
2. **Edit and compile.** Make a focused revision using a suitable implementation
   language. For a new operator or packaging boundary, prove export/AOTI and C++
   execution with the first correct implementation before extensive tuning.
   Reuse that integration path for ordinary local revisions.
3. **Check correctness.** Compare all outputs using the existing acceptance
   limits. Exercise the operator's applicable aliasing, mutation, masking and
   boundary contracts; for stateful operators, include carried state, reset and
   continuations. Retain failures.
4. **Microbenchmark.** Time the complete replacement, including required
   preparation, copies, dispatch and workspace work. Keep fixture generation,
   input checking and reference comparisons outside timing. Separate compilation
   warmup from measurement. Restore equivalent inputs/state outside timing
   unless reset is part of the target, and match relevant memory/cache
   conditions. Run GPU measurements sequentially.
5. **Profile and analyze.** Collect targeted NCU evidence for the correct
   candidate configuration and compare it with the baseline and prediction.
   Inspect generated code, including SASS for instruction-level claims. Apply the
   [bounded follow-up](experiment-loop.md#measurement-disagreements) when
   measurements materially disagree. Keep, revise or reject the implementation
   using correctness, unprofiled timing and NCU findings; retain failed and
   inconclusive revisions. Update the expected native benefit and its evidence
   basis before further tuning or integration.

Bound the local search within the outer candidate's remaining budget and reserve
time for integration. Add each revision under the same outer candidate in the
[investigation record](records/format.md), linking its parent,
source/configuration, local prediction, correctness, unprofiled timing, NCU
capture and feedback decision. Stop when the evidence or remaining time does not
support another trial.

A local winner keeps its revision ID when returning to outer steps 3–7: export,
full validation, paired C++ measurement, native profiling and the next-target
decision. Verify the exported artifact uses the intended implementation; if
compilation changes the kernel or launch configuration, obtain matching NCU
evidence. Kernel-only improvements remain local evidence until the outer loop
establishes their model-level effect.
