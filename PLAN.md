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
  actual toolchain/architecture/compiler settings in package metadata.
- Build/link the native attention operator for all Qwen3 targets. Deploy it with
  matching LibTorch/CUDA; reject old efficient packages with re-export guidance.
- Run C++ prefill/decode, cache management, greedy selection and EOS stopping
  without loading Python. No serving API, native tokenizer or concurrency is
  introduced by this cleanup.

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

The canonical benchmark compares HF eager and one compiled profile. Paired
baseline/candidate runs remain available for optimizations and record workload,
source/package hashes, actual compiler settings, rotated process order,
warmups, memory and GPU telemetry. Shared-GPU smoke runs establish functionality;
controlled performance conclusions require idle-boundary checks and repeated
measurements.

Further work can investigate CUDA Graphs, launch overhead, cache-layout changes
and measured kernel bottlenecks. Preserve the readable single supported path,
freeze artifacts for each comparison and rerun functional/numerical checks.
Historical GB10 Flash measurements remain on `dev-gb10`; they do not establish
performance on L20 or another toolchain. Profiling currently uses the existing
Qwen3 NVTX driver described in [development tools](docs/development-tools.md).

## Qwen3.5 status (unchanged by this refactor)

The separate [Qwen3.5 text-only extension](compiler/qwen35/README.md) now targets
0.8B for the first complete generation pipeline, with 4B retained for later
scaling checks. Stages 1-2 pass on 0.8B: a real DeltaNet block compiles for T=64/T=1
and handles inputs of 1-512 tokens through chunk/remainder dispatch, with bitwise
C++/Python AOTI parity and passing eager/FP32 diagnostics. Boundary, partition,
reset and 512-step decode tests pass. Bounded dynamic prefill export fails shape
constraints in the pinned toolchain. Shapes come from config/package metadata;
the existing 4B packages also pass regression through the same probe. Step 3 now
exports and compiles the full text model: the eager adapter matches HF bitwise,
and 50/50 teacher-forced AOTI next-token predictions agree. Numerical acceptance
is open for 850 conv and four recurrent-state tensors; the unchanged checks fail.
Focused probes identify a lost BF16 rounding boundary before DeltaNet gated
normalization, and the Qwen3 numerical controls do not resolve it. Issue
Q35-NUM-001 now has a retained reproducer. Step 4's Python-free C++ generation
runtime passes native/Python AOTI parity on 3,136 teacher-forced tensors, 28
greedy logit rows and 288 final state tensors, plus reset/context/stopping and
18 rejection checks. Step 5 now passes native/Python AOTI parity on 9,947
teacher-forced tensors across 203 steps. All three greedy fixtures match HF
with the same partitions; full-prompt HF differs on one tied punctuation
decision (Q35-NUM-002). Conv/recurrent-state acceptance still fails under the
unchanged limits. The three-profile benchmark is functionally checked, including
fixed-token prediction differences; final timings will run in the user's
separate environment. Native parity does not close numerical acceptance.
Results and evidence are recorded in the extension guide; the plan below
describes the original Qwen3-0.6B PoC. All 0.8B experiment steps are kept in one implementation commit.

## Deferred work

- [ ] After Qwen3.5-4B text-only inference is fully functional (AOTI export/compile,
  C++ prefill/decode generation and end-to-end validation), refactor the profiling
  infrastructure for reuse across models. Separate shared NVTX capture and trace
  analysis from model/runtime adapters and kernel classification. Use both
  Qwen3-0.6B and Qwen3.5 to guide the abstractions; defer this refactor until that
  milestone.
