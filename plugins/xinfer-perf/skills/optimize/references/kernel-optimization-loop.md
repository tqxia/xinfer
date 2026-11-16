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
comparisons can stay in the outer loop. The design loop chooses the kernel's
structure; the implementation loop realizes and tunes it, returning evidence
for the next design decision.

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

### Design loop

1. **Choose or retain a design.** Use the calibrated profile and generated code
   to compare plausible architectures using the
   [candidate comparison](experiment-loop.md#1-form-the-hypothesis). State the
   relevant decomposition, work partitioning, data/state residence and
   synchronization, with resource costs, dependent changes and expected effects.
   A small adjustment needs only a brief statement of the retained architecture.
2. **Implement and evaluate.** Run the implementation loop below within the
   chosen structure, testing whether its intended mechanisms are realized.
3. **Reassess from evidence.** Distinguish an unrealized mechanism needing repair,
   useful local tuning and a structural limit requiring a revised design. When
   repeated tuning brings no material gain or measurements suggest a structural
   limit, revisit the design before another sweep. Use a small probe when needed
   to distinguish the leading explanations, then revise, continue tuning or stop.

### Implementation loop

1. **Form a local hypothesis.** Within the chosen design, use NCU evidence and
   generated code to choose a focused revision to tiling, layout, scheduling or
   instruction use. Use the same candidate comparison for local revisions.
   Predict the affected counters and timing, and what would refute the mechanism.

   When changes plausibly interact, state the dependency to test. Compare the
   baseline, each viable change alone and the combination, or remove one change
   from a bundle. Choose only variants that answer that question, using the same
   baseline and conditions. A combined gain does not establish each member's
   contribution.
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
   basis. Continue local revisions while worthwhile; return to the design loop
   when the structure needs reassessment or the implementation is ready for
   integration.

### Records and model integration

Both loops share the outer candidate's remaining budget; reserve time for model
integration. Add each revision under the same outer candidate in the
[investigation record](records/format.md), linking its parent,
source/configuration, local prediction, correctness, unprofiled timing, NCU
capture and feedback decision. Put design choices in candidate `hypothesis` or
revision `change`, interaction questions in revision `prediction`, and feedback
from either loop in decision `assessment` / `next_action`. Stop when the evidence
or remaining time does not support another trial.

A local winner keeps its revision ID when returning to outer steps 3–7: export,
full validation, paired C++ measurement, native profiling and the next-target
decision. Verify the exported artifact uses the intended implementation; if
compilation changes the kernel or launch configuration, obtain matching NCU
evidence. Kernel-only improvements remain local evidence until the outer loop
establishes their model-level effect.
