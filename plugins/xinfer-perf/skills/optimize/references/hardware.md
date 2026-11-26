# GB10 hardware guidance

Check gotchas before collecting measurements. Verify live SM count, memory
bandwidth, and clock behavior on the target host (`nvidia-smi`, NCU
`SpeedOfLight`) rather than assuming static specs.

## Target profile

- **Platform:** NVIDIA GB10, ARM64 (`aarch64`), Blackwell SM 12.1.
- **Stack:** CUDA 13.0, Torch `2.13.0+cu130` (`environments/gb10/requirements.txt`).
- **Decode characteristics observed in xinfer** (see
  `docs/experiments/qwen35-gemv-2026-11-13.md`):
  - Small-batch BF16 decode is overwhelmingly projection-dominated (~95% of
    decode kernel time in Qwen3.5-0.8B: ~69% decoder GEMVs, ~26% vocab head).
  - High register pressure and low CTA occupancy strongly penalized default
    Inductor decode GEMVs on GB10; lowering registers and increasing CTA
    parallelism (1–4 rows/program, 4 warps, 1 stage) cut native decode latency
    by ~19% on Qwen3.5-0.8B.
  - Gains are target- and geometry-dependent — verify each model size, context
    length, and batch regime directly on GB10.

## Gotchas

### Stuck clocks under load (611 MHz)

- Check GPU clocks under a representative load before measuring performance.
- If clocks are stuck at 611 MHz, stop and recover the host before continuing
  (full shutdown + PSU disconnect recovered the affected host; recheck clocks
  under load afterward).
- Repeat any measurements taken under the fault, and **re-export** packages
  autotuned while clocks were stuck — a healthy benchmark window does not fix
  autotuned kernel choices made at 611 MHz.

### Unrelated Inductor recompilation drift

- A focused graph rewrite changes generated kernel names, which can miss
  retained Inductor cache entries for *unchanged* operators.
- Copying a tuning cache does not prove isolation. Check unaffected kernel
  bodies and launch configs inside `.pt2` before attributing a gain or
  regression to your change.

### Triton option propagation through AOTI

- Eager Triton launch flags (such as `enable_fp_fusion`) may not propagate into
  AOTI's captured user-kernel metadata. Always verify compiled arithmetic and
  numerical behavior on the packaged `.pt2`, not just eager execution.

## Growing this guide

Keep new GB10 findings concise, self-contained, and grounded in measured
evidence (include inline code snippets, toolchain versions, and measurements;
never link to temporary `work/` reports). Only split out dedicated recipe or
case files once concrete GB10 entries exist.
