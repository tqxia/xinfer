# Common optimization goals and mechanisms

These techniques apply broadly. Their speedups, compiler options and tuning
values remain specific to the measured workload and target. Start from a
profile observation and inspect the compiled implementation before choosing one.
Compiler settings, graph passes, inference adapters and custom operators are
ways to implement the goals below; a transformation can serve several goals.

## Work elimination

During the initial structural review, identify computations whose results the
requested inference path does not consume. Examples to check include vocabulary
projections for unused prompt rows, token-wise normalization before a last-row
slice, or attention over irrelevant cache capacity. Confirm the compiler has
not already omitted the work.

Preserve required outputs, all cache/state updates, masks and positions. A
discarded output can still depend on work needed to update state; trace those
dependencies before removing it. Predict the operations or rows omitted and
validate the resulting compiled path. Mathematically equivalent shapes can
still change floating-point rounding.

Review the relevant phase: an elimination for multi-token prefill can remove
nothing in single-token decode. Record already implemented or inapplicable
opportunities rather than forcing a candidate. Reuse the review until the graph,
workload or new evidence warrants another look.

## Fusion

Look for related launches and global-memory intermediates in the measured
region. Consider compiler fusion/combo options or a custom fused kernel;
independent projections sharing an input may suit packing. Check existing
compiler/pass coverage first; source-op count is not launch count.

Predict savings in launches, intermediates or repeated loads. The required
arithmetic may remain; describe any redundant computation removed separately.
Check whether added copies, workspace, register pressure or changed rounding
boundaries offset the saving. Measure the whole replacement, including its
preparation, then use the remaining profile costs to guide another iteration.

## Kernel optimization

Improve the implementation of a required operation through compiler/library
selection, tuning, layout/scheduling changes or rewriting. Use the observed cost
to choose a focused experiment, then iterate from its measured result.

### Compiler and library choices

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

### Layout and scheduling

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
