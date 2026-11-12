# Kernel optimization

Improve the implementation of a required operation through compiler/library
selection, tuning, layout/scheduling changes or rewriting. Use the observed cost
to choose a focused experiment, then iterate from its measured result.

## Implementation choices

Consider Triton, CUDA C++ and targeted PTX according to the measured bottleneck,
required integration and target support. There is no required progression from
one language to another. Compiler options and existing library kernels remain
valid choices when they address the observation.

Use CUDA intrinsics or inline PTX when the evidence identifies a specific
instruction-level opportunity. Check the target SM and toolchain, inspect
generated PTX/SASS and resource use, and validate numerical and memory behavior.
PTX is a virtual ISA; source instruction counts do not establish the final
machine code or a speedup. See NVIDIA's [inline PTX guide](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html)
and [PTX ISA introduction](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#introduction).

## Kernel inner loop

Use this inside outer step 2, **Implement the change**, when rewriting an
existing kernel or building a custom kernel for fusion. Ordinary compiler-option
comparisons can stay in the outer loop.

Define the operator or fused-region boundary, reference and baseline once. Reuse
existing component fixtures and declared tolerances. Cover actual shapes,
strides, dtypes, representative values, nonzero/carried state and supported
boundaries; use captured values for data-dependent execution. A fusion baseline
is the complete original region. Keep the local baseline and reference intact
while revising the implementation.

Establish unprofiled local timings and a matching NCU baseline capture before
choosing kernel revisions. NCU is required for baseline and candidate kernel
analysis; matching captures may be reused. If collection is unavailable or
permission-blocked, retain the error and defer kernel analysis and further
kernel hypotheses until it is available. Preparation and correctness work can
continue, but timing alone does not complete the inner loop.

1. **Form a local hypothesis.** Use the NCU evidence and generated code to choose
   a change in tiling, layout, scheduling, instruction use or implementation.
   Predict the affected counters and timing, and what would refute the mechanism.
2. **Edit and compile.** Make a focused revision using a suitable implementation
   language. For a new operator or packaging boundary, prove export/AOTI and C++
   execution with the first correct implementation before extensive tuning.
   Reuse that integration path for ordinary local revisions.
3. **Check correctness.** Compare every output and carried state; exercise
   mutation/alias rules, masking, boundaries, reset and continuations. Keep the
   existing acceptance limits and failures visible. Operator registration checks
   alone do not establish numerical correctness.
4. **Microbenchmark.** Time the complete replacement, including required
   preparation, copies, dispatch and workspace work. Keep fixture generation,
   input checking and reference comparisons outside timing. Separate compilation
   warmup from measurement. Restore equivalent inputs/state outside timing
   unless reset is part of the target, and match relevant memory/cache conditions.
   Run GPU measurements sequentially.
5. **Profile and analyze.** Collect targeted NCU evidence for the correct
   candidate configuration and compare it with the baseline and prediction.
   Follow [profiling.md](../profiling.md) for metric selection and replay/cache
   caveats. Inspect generated code, including SASS for instruction-level claims.
   Keep, revise or reject the implementation using correctness, unprofiled timing
   and NCU findings; retain failed and inconclusive revisions.

Bound the local search within the outer candidate's remaining budget and reserve
time for integration. Record each revision's source/configuration, correctness,
unprofiled timing, NCU capture and predicted-versus-observed feedback in ordinary
notes. Stop when the evidence or remaining time does not support another trial.

A local winner returns to outer steps 3–7: export, full validation, paired C++
measurement, native profiling and the next-target decision. Verify the exported
artifact uses the intended implementation; if compilation changes the kernel or
launch configuration, obtain matching NCU evidence. Kernel-only improvements
remain local evidence until the outer loop establishes their model-level effect.

## Compiler and library choices

For a dominant matrix or reduction region, inspect the selected implementation
and its shapes/layouts. Compare a small set of supported compiler options,
autotuning choices or library/generated routes, using the installed version's
configuration and existing exporter interface. Keep unrelated settings fixed.

Predict which execution cost should change and verify the selected code actually
changes. Fewer library calls or a faster isolated kernel need not reduce native
decode latency once transformations, reductions and workspace are included.

A tuning flag can also change decomposition and arithmetic. In the
[GB10 experiment](https://github.com/tqxia/xinfer/blob/3e05a9d4db8c945dfcd12d05a31c50955775d8b6/docs/experiments/gb10-qwen35-perf-plugin-2026-11-10.md),
coordinate-descent tuning replaced singleton matmuls with reductions. Its broad
speedup added required token/state failures despite exact Python/native parity.
A narrower head experiment exposed another confound: fresh default autotuning
changed unrelated normalization schedules. Freeze unchanged per-kernel choices
when isolating a transformation, verify actual kernel bodies, launch settings
and binaries plus external-call operands, and compare state payloads directly.
Check output-logit errors separately; matching state does not imply matching
logits. A measured global gain is not numerical acceptance.

## Layout and scheduling

Use profiler evidence of inefficient accesses, spills, stalls or limited
parallelism to choose a focused mapping, tile, warp-count or scheduling change.
Preserve real input strides, dtypes, state and supported shape boundaries. A
rewritten kernel or library replacement must account for preparation, dispatch
and workspace costs as well as its main computation.

Predict the relevant counter or execution change, then compare it with latency
and memory. More occupancy is not itself the objective; extra parallel work can
add communication or reduction costs. Architecture-specific implementations
of these ideas belong in the [hardware note](hardware.md).

For concrete layout and tuning examples, consult the official
[Triton matrix-multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
for the installed Triton version. Its example configurations are starting
references, not settings to copy into a different kernel or device.
