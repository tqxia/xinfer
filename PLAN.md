# Planned work

Current usage belongs in [README.md](README.md), the model guides and
[development tools](docs/development-tools.md). Recorded outcomes and open
numerical findings belong in [validation status](docs/validation-status.md).
This file lists unfinished work, not completed refactors or experiment logs.

## Numerical acceptance and scaling

- Resolve the Qwen3.8-27B-FP8 logit failures across normalization, attention,
  DeltaNet and packing before numerical acceptance. Retain the pinned FP8 HF
  reference, current bounds and separate native parity result; attribute the
  remaining production AOTI differences and expand the focused validation matrix.
- Investigate the Qwen3 focused 98-token logit failure and Qwen3.5-4B required
  logit errors. Preserve matched inputs, current tolerances and independent HF
  references; native parity alone does not close these issues.
- Recheck Q35-NUM-001 on the current toolchain and isolate any remaining cast-
  lowering issue in a small reproducer. Keep partition-sensitive token drift
  separate from compiler attribution.
- Validate current bundles for expanded contexts and 4B on GB10. Existing 0.8B
  1K-context GEMV evidence does not cover those profiles or Qwen3 packaging.
- Evaluate larger per-call prefill profiles against the existing chunked path.
  Long prompts already work through calls of at most 512 tokens; increasing that
  per-call bound needs new graph/kernel/workspace checks and idle-target timing.
- Preserve boundary coverage for 1/511/512/513/8191/8192-token prompts, singleton
  remainders, nonzero offsets, reset/reuse, full capacity and overflow. Require
  matched-partition accuracy checks and complete-prompt latency/memory evidence
  before selecting a larger default. Measure export/validation memory separately
  from native inference memory.

## Measured performance work

- Use the [performance plugin](plugins/xinfer-perf/README.md) to select fusion or
  kernel changes from native profiles. Decoder projections and the vocabulary
  head remain candidates, but gains and useful specializations are target-
  dependent; see the [GEMV case study](docs/experiments/qwen35-gemv-2026-11-13.md).
- Evaluate extending the fused zero-workspace `BF16x3` Tensor Core chunked
  CUDA prefill kernel (`xinfer::batched_deltanet`, per-row threshold 16 on SM80+)
  to larger per-call prefill profiles (> 512 tokens) and additional target GPUs
  as needed, preserving the BF16 (1%) and FP32 (0.01%) accuracy limits.
- Investigate CUDA Graphs, launch overhead and cache/state coordination as
  separate runtime work. Add a runtime-optimization skill only when that work
  starts; do not scaffold it now.
- Select independent prefill/decode attention implementations only for a concrete
  use case, with the linkage/registration changes described in the
  [attention guide](docs/flash-attention.md#build-and-deployment).

## Environment and plugin follow-up

- Add CodeBuddy packaging and verify discovery when that integration is taken up.
  The current plugin supports Codex and Claude Code. Grow GB10 hardware notes and
  kernel examples in `plugins/xinfer-perf/` from actual investigations.
- CI and small end-to-end GPU fixtures remain deferred. This cleanup does not
  introduce either.
