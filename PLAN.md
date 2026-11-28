# Planned work

Current usage belongs in [README.md](README.md), the model guides and
[development tools](docs/development-tools.md). Recorded outcomes and open
numerical findings belong in [validation status](docs/validation-status.md).
This file tracks unfinished work only.

## Numerical acceptance and scaling

- Resolve the Qwen3.8-27B-FP8 logit failures across normalization, attention,
  DeltaNet and packing before numerical acceptance, retaining the pinned FP8 HF
  reference, current bounds and separate native parity result.
- Investigate the Qwen3 focused 98-token logit failure and Qwen3.5-4B required
  logit errors against unmodified HF references.
- Recheck `Q35-NUM-001` on the current toolchain and isolate any remaining
  cast-lowering issue in a small reproducer, keeping partition-sensitive token
  drift separate from compiler attribution.
- Validate current bundles for expanded contexts and 4B on GB10.
- Evaluate larger per-call prefill profiles (> 512 tokens) against the existing
  chunked path with graph/kernel/workspace checks, matched-partition accuracy,
  boundary coverage (`1/511/512/513/8191/8192`, singleton remainders, nonzero
  offsets, reset/reuse, full capacity and overflow), and idle-target latency and
  memory measurements.

## Measured performance work

- Use the [performance plugin](plugins/xinfer-perf/README.md) to select fusion or
  kernel changes from native profiles; decoder projections and the vocabulary
  head remain target-dependent candidates.
- Evaluate extending `xinfer::batched_deltanet`'s fused zero-workspace `BF16x3`
  Tensor Core chunked CUDA prefill kernel (per-row threshold 16 on SM80+) to
  larger per-call prefill profiles (> 512 tokens) and additional target GPUs
  while preserving the 1% BF16 and 0.01% FP32 accuracy limits.
- Investigate CUDA Graphs, launch overhead and cache/state coordination as
  separate runtime work (add a runtime-optimization skill only when that work
  starts).
- Select independent prefill/decode attention implementations only for a concrete
  use case, with the linkage/registration changes described in the
  [attention guide](docs/flash-attention.md#build-and-deployment).

## Environment and plugin follow-up

- Add CodeBuddy packaging and verify discovery when needed (the current plugin
  supports Codex and Claude Code). Grow GB10 hardware notes and kernel examples
  in `plugins/xinfer-perf/` from measured investigations.
- CI and small end-to-end GPU fixtures remain deferred.
