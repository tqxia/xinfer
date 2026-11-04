# Qwen3 implementation and deferred work

Current Qwen3-0.6B scope: one fused BF16 AOTInductor compilation profile using
valid-prefix FlashAttention/native GQA, explicit caller-owned cache state and a
Python-free C++ generation loop. The eager-numerics profile and efficient
compiled backend are retired. Authoritative commands and contracts are in the
[Qwen3 guide](compiler/qwen3/README.md); historical numerical investigations are
in [Qwen3 numerical history](docs/experiments/qwen3-numerics.md).

## Supported Qwen3 pipeline

- Load the original checkpoint read-only; preserve independent HF eager/FP32
  reference execution and offline tokenization/text decoding.
- Install the existing Flash module adapter before export. Export bounded
  dynamic prefill (2–512 tokens) and decode (one token) with the same fixed ABI.
- Use batch-one BF16, caller-owned `[1,8,1024,128]` K/V tensors and an explicit
  shape-only CPU extent input. Capacity is 1,024 tokens; one-token prompts use
  decode at position zero.
- Keep BF16 cast emulation, TF32 disabled and normal compiler fusion. Record
  runtime compatibility and numerical settings in package metadata; keep full
  build details and diagnostics in export logs.
- Build/link the native attention operator for all Qwen3 targets. Deploy it with
  matching LibTorch/CUDA; reject old efficient packages with re-export guidance.
- Run C++ prefill/decode, cache management, greedy selection and EOS stopping
  without loading Python. No serving API, native tokenizer or concurrency is
  introduced by this cleanup.
- Share the package runner and generation/benchmark/validation executables
  across models. Use common versioned metadata, one state policy driven by the
  metadata and runtime ABI, and model-specific ABI validation.

## Validation and acceptance

Use independent eager inputs and teacher-forced identical tokens to compare
logits. Hard requirements are finite outputs, bitwise Python AOTI/native logits
and caches, correct write regions/masking, stable storage, reset/reuse, boundary
and overflow rejection, and native/Python AOTI generation agreement. HF logits
retain the 5% relative-L2 and peak-normalized error limits. HF cache differences
and differing top-token/greedy choices are diagnostic reports, with HF parity
separate from overall acceptance. Keep FP32 controls in reports.

The saved pre-refactor Flash matrix has 26 cases/185 rows, bitwise native/Python
AOTI parity, 14/56 HF cache tensors over the former limit and two top-token
mismatches. Independently tuned builds can differ. Re-export and validate both
entry points for every compiler/toolchain change; historical results are not a
claim that the new artifact passed. Current refactor verification is recorded in
[the validation record](docs/experiments/qwen3-refactor-validation.md).

## Measurement and next work

Both model benchmarks compare ordinary HF dynamic-cache eager and one compiled
profile through the shared cached-forward helper. Reports mark this baseline
with `eager_baseline: hf_dynamic_cache`; earlier masked-cache speedups need fresh
runs. Paired baseline/candidate runs remain available for optimizations and
record workload, package/native identities, numerical settings, rotated process order,
warmups, memory and GPU telemetry. Shared-GPU smoke runs establish functionality;
controlled performance conclusions require idle-boundary checks and repeated
measurements.

Further work can investigate CUDA Graphs, launch overhead, cache-layout changes
and measured kernel bottlenecks. Preserve the readable single supported path,
freeze artifacts for each comparison and rerun functional/numerical checks.
Historical GB10 Flash measurements remain on `dev-gb10`; they do not establish
performance on L20 or another toolchain. Generate profiling scripts as needed
for each investigation rather than maintaining profiling infrastructure here.

## Supported Qwen3.5 pipeline and scaling work

The [Qwen3.5 text-only pipeline](compiler/qwen35/README.md) currently supports
0.8B. It exports paired dynamic prefill (2–512 tokens) and single-token decode
with packaged Triton DeltaNet kernels and shared native valid-prefix
FlashAttention. Caller-owned KV buffers update in place; conv/recurrent inputs
remain unchanged, and returned tensors become the next state. Both entries share
one weight pool and use metadata schema version 3 with runtime ABI 1, the common
runner and shared
native generation, benchmark and validation executables.

Native/Python AOTI parity is validated for 0.8B; HF numerical acceptance remains
open. Keep the existing HF/FP32 comparisons and Q35-NUM-001/Q35-NUM-002 diagnostics.
The [experiment record](docs/experiments/qwen35.md) preserves earlier block probes,
fixed-shape packages and export failures as historical evidence. Those results
do not establish support for 4B in the current full-model Triton path.

4B full-model support remains a later milestone. Before claiming support:

- Support its separate key/value head counts (16/32), 8,192 convolution channels
  and per-value-head recurrent state. Update Q/K grouping, channel and gate
  offsets, convolution write ownership and workspaces in both short and chunked
  DeltaNet paths.
- Add the model's attention geometry from its verified config to the shared
  operator, fake shape contract, adapter and ABI validation. Coordinate any
  larger context/token ranges across export, metadata and runtime contracts.
- Regenerate independent HF/FP32 fixtures and validate state preservation,
  partition/reset behavior, native/Python AOTI parity, generation and GPU memory
  with the complete shared-weight package pair.

Keep unsupported-shape checks until the new profile passes these gates. Preserve
tested tuning specializations where they still fit the target configuration.
