# Kernel optimization

Improve the implementation of a required operation through compiler/library
selection, tuning, layout/scheduling changes or rewriting. Use the observed cost
to choose a focused experiment, then iterate from its measured result.

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
