# Common optimization mechanisms

These mechanisms apply broadly. Their speedups, compiler options and tuning
values remain specific to the measured workload and target. Start from a
profile observation and inspect the compiled implementation before choosing one.

## Remove or combine work

Look for related launches and global-memory intermediates in the measured
region. Consider compiler fusion/combo options, an adapter or graph rewrite,
or a fused kernel. Independent projections sharing an input may suit packing.
Check existing compiler/pass coverage first; source-op count is not launch count.

Predict the work eliminated: launches, intermediates or repeated loads. Check
whether added copies, workspace, register pressure or changed rounding boundaries
offset that saving. Measure the whole replacement, including its preparation.

## Choose compiler or library execution

For a dominant matrix or reduction region, inspect the selected implementation
and its shapes/layouts. Compare a small set of supported compiler options,
autotuning choices or library/generated routes, using the installed version's
configuration and existing exporter interface. Keep unrelated settings fixed.

Predict which execution cost should change and verify the selected code actually
changes. Fewer library calls or a faster isolated kernel need not reduce native
decode latency once transformations, reductions and workspace are included.

A tuning flag can also change decomposition and arithmetic. In the
[GB10 experiment](../../../../../../docs/experiments/gb10-qwen35-perf-plugin-2026-11-10.md),
coordinate-descent tuning replaced singleton matmuls with reductions. Its broad
speedup added required token/state failures despite exact Python/native parity.
A narrower head experiment exposed another confound: fresh default autotuning
changed unrelated normalization schedules. Freeze unchanged per-kernel choices
when isolating a transformation, verify actual kernel bodies, launch settings
and binaries plus external-call operands, and compare state payloads directly.
Check output-logit errors separately; matching state does not imply matching
logits. A measured global gain is not numerical acceptance.

## Adjust layout and scheduling

Use profiler evidence of inefficient accesses, spills, stalls or limited
parallelism to choose a focused mapping, tile, warp-count or scheduling change.
Preserve real input strides, dtypes, state and supported shape boundaries.

Predict the relevant counter or execution change, then compare it with latency
and memory. More occupancy is not itself the objective; extra parallel work can
add communication or reduction costs. Architecture-specific implementations
of these ideas belong in the [hardware note](hardware.md).

For concrete layout and tuning examples, consult the official
[Triton matrix-multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
for the installed Triton version. Its example configurations are starting
references, not settings to copy into a different kernel or device.
