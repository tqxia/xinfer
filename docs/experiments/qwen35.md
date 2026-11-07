# Qwen3.5 experiment record

The current full-model path uses dynamic prefill, packaged Triton GDN and native
prefix FlashAttention with caller-owned KV buffers. See
[current commands](../../compiler/qwen35/README.md) and
[the Flash integration record](#prefix-flash-and-in-place-kv-2026-11-02).
Current checkpoint download, verification and token commands use
`python -m tools.prepare`; acquisition has moved out of the model implementation.
Current benchmark commands use `python -m tools.benchmark` for both models;
benchmark reports contain performance results independently of validation.
The retained rounding diagnostic now runs with
`python -m tools.diagnostics.qwen35.probe_rounding`.
Historical results retain their original artifact hashes. Historical commands
may reference retired source files; use their recorded source snapshots to
reproduce those experiments.

Current numerical validation uses `python -m tools.validation.validate` and
compares HF → eager → Python AOTI → C++ AOTI. Eager includes adapters and
package-selected linear packing. Qwen3.5 runtime-contract checks remain under `tools/validation/qwen35/`. The unified HF
reference uses matched input partitions; historical `hf_full_prompt` report
labels are retained only in the experiment records below.
Fused/conditional numerical and behavior checks are unit tests under
`compiler/qwen35/tests/`; `test_kernels_aoti` covers kernel export and Python AOTI
execution. The standalone kernel C++ runner has been removed. Historical commands
below retain their original module paths, binaries and report formats; use their
recorded checkout, or the current commands in the model guide for a new run.

The fixed-shape block exporter (`diagnostics/export_block.py`) and original HF
dynamic-export probe (`diagnostics/probe_dynamic.py`) are archived in commit
`0984678`. That commit also contains the old `diagnostics/block.py` path; its
standalone block preparation and loading are retired. The commands below retain
their historical paths and use the original checkout or recorded source snapshot.

The legacy `qwen35_block` runner, `tests/validate_block.py` and fixed-shape
sequence helpers were subsequently retired; their last containing commit is
`a63eb21`. The current synthetic unit tests and rounding diagnostic use only
the explicit-state wrappers `TensorState` and `BlockStep` from
`compiler/qwen35/tests/delta.py`. The diagnostic loads the prepared full-model
text checkpoint rather than a separate block checkpoint.

## Full 4B L20 validation: 2026-11-04

The complete text checkpoint at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` was verified and exported in the
`xinfer-l20` container. The environment uses Torch `2.13.0+cu126`,
Triton `3.7.1` and Transformers `5.20.2`, with a full L20 exposed as CUDA device
zero. The checkpoint contains 426 text tensors and 8,411,502,592 weight bytes.
Ordinary strict export and independent linear packing are restored; the
standard validators run in process with their original cases and numerical
limits. Prefill remains 2–512 tokens and total capacity remains 1,024.

| Stage | Time | Sampled device peak | Result |
| --- | ---: | ---: | --- |
| Paired export | 163.2 s | 23,632 MiB | Passed |
| Model validation | 217.5 s | 34,630 MiB | Numerical gate failed |
| Native runtime validation | 186.2 s | 17,703 MiB | Passed |
| End-to-end validation | 236.9 s | 25,022 MiB | Numerical gate failed; native parity passed |

These are one-second whole-device samples, not exact allocation peaks or
performance benchmarks. No stage ran out of memory. CPU checks passed: 96 common
tests and 32 Qwen3.5 tests, with 14 CUDA-dependent tests skipped in those CPU runs.

Native validation matched Python AOTI bitwise for 6,955 teacher tensors, 35
greedy logit rows and 384 greedy state tensors; all 58 rejection cases passed.
End-to-end validation matched 301 teacher steps and 19,565 tensors bitwise,
preserved cache mutation/masking/reset contracts and matched all three HF greedy
continuations. All 301 teacher top-token choices also agreed with HF.

The existing 5% numerical gates still fail. Model validation records 16,503
required failures across adapter/HF, AOTI/HF, packing/adapter, AOTI/adapter and
AOTI/packing: 75 logit rows and 16,428 state tensors. The AOTI/HF logit maximum
relative L2 is 36.36%. End-to-end records 60 failing logit rows and 11,851 state
tensors, with a maximum logit relative L2 of 41.49%. Every shared comparison
metric and required failure matches the earlier dev-box run: 444 model rows and
301 end-to-end rows. This establishes equality of recorded metrics, not a
cross-run bitwise comparison of complete traces.

Full FP32 controls now cover all 87 model steps and the original six end-to-end
cases (100 steps). They are diagnostic: BF16 HF and AOTI both exceed the FP32
comparison bounds, despite agreeing on top tokens. They do not excuse the
required failures or establish that one implementation is uniformly more
accurate. Full-model numerical acceptance remains open.

The shared archive is 8,411,833,455 bytes. A fresh-process `cudaMemGetInfo` probe
measured 7.9844 GiB growth after loading both entries and 8.0273 GiB growth after
an eight-token prefill plus one decode; absolute device use was 8.3927 GiB after
decode. This is a short native inference measurement, not a full-context peak.
The two entries use one externally managed weight pool. Device memory remained
reserved after runtime destruction in this probe; process exit released it.

The validation run retains the packages, complete validator reports, export
logs, stage status, device samples, verified checkpoint-transfer record and
native memory report. Run-specific paths and deployment details belong in
local validation records.

## Prefix Flash and in-place KV: 2026-11-02

The six full-attention layers now reuse the shared native prefix-Flash operator
with eight query heads, two KV heads and head dimension 256. The adapter retains
HF projections, Q/K normalization, partial RoPE and the sigmoid output gate.
The 18 DeltaNet layers retain the same packaged Triton kernels and conditional
dispatch. KV updates write into caller-owned buffers; returned KV tensors alias
those buffers. Conv/recurrent inputs remain unchanged. The CPU `kv_extent`
input supplies the populated length through its shape, without reading CUDA
tensor data. Lower-right causality covers decode and appended prefill chunks.

Both native model families were rebuilt. Qwen3.5 full-model executables now
require `libxinfer_attention_ops.so`; standalone GDN targets remain independent
of it. Rebuild and re-export the earlier `efficient_sdpa` / functional-state
packages. Current runtimes reject their metadata before package loading. The
five shared-weight deployment files and externally managed weight ownership
remain unchanged.

| Entry | Graph nodes | Conditional blocks | Short / long branch ops | In-place KV updates | Flash calls |
| --- | ---: | ---: | --- | ---: | ---: |
| Prefill | 2,300 | 18 | 18 / 18 | 12 | 6 |
| Decode | 2,262 | 0 | 18 / 0 | 12 | 6 |

Strict export retains dynamic prefill T=2..512 and a dynamic extent of 2..1025
elements. Generated-code inspection confirms that the old twelve full-capacity
1 MiB KV copies and twelve expanded-KV temporaries are absent in both entries.
Decode writes only twelve rows of 512 BF16 elements, totaling 12 KiB of KV
stores per token. This removes 12 MiB of copied cache payload (24 MiB of
read/write traffic) and 48 MiB of explicit head-expansion output writes per
token. These counts exclude the native Flash backend's internal traffic and
workspaces, and all other model operations.

| Validation | Result |
| --- | --- |
| Shared operator, conditional-cache and gated-adapter regressions | 4 operator tests, 1 CPU cache test and 2 adapter tests pass |
| Python runtime contract | 6 tests / 43 subcases pass both normally and with `python -O`, with CUDA hidden |
| Benchmark GPU-selection regressions | 19 tests pass |
| Native runtime and CLI | 35 cases; 5,243 teacher tensors, 288 greedy final-state tensors and 30 greedy logit rows match Python AOTI bitwise; 49 rejection cases pass |
| Full end-to-end native parity | 32 cases, 294 teacher steps and 14,406 teacher tensors match Python AOTI bitwise |
| Qwen3 compatibility | Existing paired package needs no re-export; all 79 native output files (454,738,432 bytes) match the previous baseline hashes |

HF numerical acceptance remains **open**. The end-to-end report retains 4,693
convolution-state and 15 recurrent-state failure entries, compared with 4,693
and 29 in the prior shared-weight baseline. Both reports have four HF teacher
top-1 differences, and the capital greedy continuation first differs from HF at
generated-token index 1. The maximum AOTI/HF logit relative-L2 error is
0.03144832328, compared with 0.03322604671 previously. The existing 5% numerical
limit is unchanged; native parity and these comparisons do not establish full
HF equivalence.

A separate fresh process measured each bundle after a 128-token prefill and
two warm decode calls. Five measured decode calls ending at positions 131..135 had the
same pre-call live allocation, 1,479.0659 MiB, in both bundles. The transient
PyTorch allocator peak above that live allocation fell from 38.8798828 MiB to
19.3203125 MiB; all five samples agreed. KV input/output addresses were stable
for 0/12 buffers previously and 12/12 buffers with the new contract. This is an
allocator measurement, not reserved memory, NVML device usage or end-to-end
device-memory savings. The shared development GPU reported 100% utilization,
so this run makes no latency or throughput claim.

Packages and reports are under `artifacts/l20/qwen35-flash-20261102/`, including
the `shared/` bundle and `memory-before.json` / `memory-after.json`. The memory
probe is `work/l20/qwen35-flash-20261102/memory_probe.py`. The previous numerical
baseline is
`artifacts/l20/shared-weights-archive-20261102/qwen35/shared-end-to-end/summary.json`;
the compiled-code comparison uses
`artifacts/l20/qwen35-default-20261102/shared/decode.pt2` as its old package.
Retain each report's package, source and attention-library hashes when
reproducing the comparison. Large native state traces were hashed and removed
after bitwise parity checks; their digests and reports remain available.

## Conditional full-model GDN integration: 2026-10-30

Measured source is `ae402e8` plus the conditional-dispatch working tree.
`kernels/recurrent.py` now owns the checked functional dispatcher and both
`triton_op` branches. T=1 and prefill through T=97 use unchanged fused
recurrence. Longer prefill uses C64 chunking with B3's 32-column U/W transform,
now the default implementation in `kernels/chunked.py`. The diagnostic copy
and temporary transform override are removed. The eight-input/three-output
tensor interface, BF16 boundaries and FP32 IEEE arithmetic remain the same.
Threshold 97 remains a provisional isolated-kernel crossover.

The full model needs no additional flag. Both entry points were compiled with
`--enable-inductor-optimization`, retaining combo kernels, 48 packing groups and
187->97 linear calls. The metadata records both algorithms/operators, threshold,
chunk size, transform tile and dot precision. Python and C++ reject mismatched
policy fields across the pair. Older package pairs with fields missing from
both entries remain compatible.

| Entry | Top-level graph nodes | Conditional blocks | Short / long branch ops | Token range |
| --- | ---: | ---: | --- | --- |
| Prefill | 2,203 | 18 | 18 / 18 | 2..512 |
| Decode | 2,165 | 0 | 18 / 0 | 1 |

Each prefill conditional remains symbolic through strict export. The actual
packaged C++ contains all 18 `if (s0 <= 97L)` statements and both kernel families;
there are no proxy-executor calls. A CUDA launch trace of the complete model
observes 18 fused GDN launches at T=97, and 18 launches of each of the five
chunk-path kernels at T=98, with no fused GDN launches in that long request.
These are dispatch checks, not latency measurements.

| Validation | Result |
| --- | --- |
| Promoted standalone kernels | 101 cases / 566 calls; all 1,698 JIT/Python AOTI/C++ tensors bitwise equal; `dispatch_passed=True` |
| Full-model state/masking suite | 87 steps; finite shapes/dtypes, input preservation, reset/repeat and future-KV checks pass, including T=96/97/98 |
| Native runtime | 35 cases; 5,243 teacher tensors, 30 greedy logit rows and 288 final-state tensors match Python AOTI bitwise |
| CLI and metadata rejection | 27 cases pass, including mixed/missing threshold, algorithm and transform-tile policy |
| End-to-end native teacher trace | 295 steps / 14,455 tensors match Python AOTI bitwise; 32 total teacher/greedy cases |
| Greedy generation | Capital ` Paris.\n`, arithmetic `42\n` and Chinese response all match HF in these three fixtures |

Both model suites cover threshold boundaries, mixed short/long/decode sequences,
repeat/reset, poisoned future KV and context overflow. Native processes load no
Python library. Full HF/FP32 numerical acceptance remains open. The expanded
model report retains 6,785 numerical failure entries across adapter/packing/AOTI
comparisons. The matched end-to-end report retains 4,867 convolution and 14
recurrent-state failures; its sole AOTI-vs-HF teacher top-1 difference is
`length_33`, step 0, with a zero HF top-two margin. Both HF and AOTI also differ
from FP32 at `capital_teacher`, step 1, with FP32 margin about 0.09289.
Matching selected greedy responses does not establish general accuracy.
These totals include the added cases; no numerical limit was enlarged.

The first expanded end-to-end harness hit the container's 64 GB memory limit
after its native worker finished. It retained HF, Python AOTI and FP32 CPU
traces while native output files also occupied tmpfs. The validator now proves
native/Python bitwise parity first, replaces the duplicate Python trace with
file-backed views of the verified native files, then runs teacher-forced HF/FP32 controls.
Selected final states are cloned before serialization. A separate 441-tensor
check verifies the mixed-dtype mapping. The complete rerun finished without
another OOM event, with all cases and acceptance rules retained.

Reproduce with fresh output directories after exporting both entries and
rebuilding the native targets using the commands in the compiler README:

```bash
work/qwen35/venv/bin/python -m compiler.qwen35.export prefill \
  --enable-inductor-optimization --output work/qwen35/cond-text
work/qwen35/venv/bin/python -m compiler.qwen35.export decode \
  --enable-inductor-optimization --output work/qwen35/cond-text
work/qwen35/venv/bin/python -m compiler.qwen35.tests.validate_cond \
  --output artifacts/qwen35/0.8b/cond-core-new
work/qwen35/venv/bin/python -m compiler.qwen35.tests.validate_model \
  --packages work/qwen35/cond-text --output artifacts/qwen35/0.8b/cond-model-new
work/qwen35/venv/bin/python -m compiler.qwen35.tests.validate_runtime \
  --packages work/qwen35/cond-text --output artifacts/qwen35/0.8b/cond-runtime-new
work/qwen35/venv/bin/python -m compiler.qwen35.tests.validate_end_to_end \
  --packages work/qwen35/cond-text --output artifacts/qwen35/0.8b/cond-end-to-end-new
```

Numerical suites return exit 1 while the stated accuracy issues remain open;
inspect structural/native checks separately. Compact reports, source snapshots,
context-end tensors, package metadata and launch evidence are retained under
`artifacts/qwen35/0.8b/cond-integration/`. The full compiled pair is temporarily
at `/dev/shm/xinfer-cond-model-20261030/text`. Verified raw native outputs were
discarded after their hashes and comparisons were recorded to fit development
storage; inputs and packages permit regeneration. Temporary files can disappear
after restart.

Validation uses development Torch `2.7.0+cu126`, Triton `3.3.0`, SM89 L20 and
C++11 ABI enabled. The GPU remains busy; no performance result is claimed.
The separate alpha Torch/Triton 3.2 stack still needs its own functional run.

## Standalone conditional GDN dispatch: 2026-10-30

This earlier standalone implementation has since been promoted from
`diagnostics/cond_dispatch.py` into the production kernels described above.
The following records its original isolated validation.

`compiler/qwen35/diagnostics/cond_dispatch.py` tests a graph-level
`torch.cond(qkv.shape[1] <= 97, short, long, operands)`. Short requests reuse
the existing fused preparation/recurrence kernel; long requests use C64
chunking with the B3 32-column U/W transform. The copied transform is
AST-identical to the original B3 source apart from its function name. Existing
chunk kernels are unchanged; their launcher gains a static transform/grid
override whose defaults preserve the full-model path. The full model does not
call this diagnostic dispatcher.

Both branches are functional `triton_op`s with the existing eight tensor
inputs and fresh `(core, next_conv, next_recurrent)` outputs. Chunk workspaces
are allocated inside the long branch. The predicate uses symbolic shape
metadata directly, so native dispatch needs no GPU scalar extraction.
Threshold 97 is the earlier isolated-kernel crossover, not a tuned full-model
threshold. C32, autotuning and precision changes are outside this probe.

Strict export and AOTI compilation succeed for BF16 and FP32. Each dtype has
one T=1 decode package and one dynamic T=2..512 prefill package. Export retains
a symbolic `higher_order.cond` and both branch operators. Explicit
`run_decompositions(..., decompose_custom_triton_ops=True)` retains the
condition and exposes both kernel families. This public option defaults to
false in Torch 2.7; ordinary export/decomposition intentionally preserves
high-level `triton_op` nodes.

The actual prefill packages contain six cubins and a generated C++
`if (s0 <= 97L)`. Decode packages contain one fused cubin and no conditional.
A separate CUDA launch trace of the final BF16 prefill package observes one
`fused_kernel` launch at T=97 and only the five chunk-path launches at T=98.
The unselected pipeline is not enqueued as gated no-op kernels.

The validator reuses the original 83 cases/400 calls without changing seeded
fixtures. Eighteen derived cases add T=96/97/98 with zero/nonzero state, mixed
short/long/decode sequences and repeats, and short prefill followed by 64 decode
continuations. The original long-prefill continuation, checkpoint projection,
tail/extreme-gate and numerical partition controls remain present.

| Check | Result |
| --- | --- |
| Expanded suite | 101 cases, 566 calls: 148 prefill and 418 decode |
| JIT dispatcher vs explicitly selected branch vs Python AOTI vs C++ | All 1,698 output/state tensors match bitwise |
| Strict dynamic range and both packaged branches | Pass for BF16 and FP32 |
| HF recurrent/chunk-64, FP32-input, partition and B3 controls | Existing precision-specific regression limits pass; `dispatch_passed=True` |
| Input preservation, output ownership, shapes/dtypes/finites, reset/repeat | Pass |
| Nondefault streams | JIT/Python AOTI, both dtypes and both branches pass |
| Noncontiguous inputs | Python dispatcher passes; package ABI remains contiguous |
| Native process | Four packages loaded; no Python or separate custom-op library |

Full-FP32 acceptance remains open. The original subset retains exactly the same
127 failure keys in each of dispatcher-vs-FP32 and HF-vs-FP32, compared with the
earlier development chunked report. The expanded suite has 182 failures per
comparison, 364 total; all failures are in these two BF16-to-FP32 comparisons.
No error limit is enlarged. Overall `passed=False` and exit 1 remain visible
alongside `dispatch_passed=True`. This operator result does not establish
full-model numerical acceptance or resolve its earlier non-tie diagnostics.

Run from the project root with the matching Python/LibTorch build:

```bash
work/qwen35/venv/bin/python -m compiler.qwen35.tests.validate_cond \
  --binary work/qwen35/build/qwen35_fused \
  --output artifacts/qwen35/0.8b/cond-validation-new
```

Output directories must be fresh. Compact reports, all four packages, graph
dumps, generated wrappers/cubin hashes and launch evidence are retained in
`artifacts/qwen35/0.8b/cond-validation/`. Large raw input/output fixtures remain
temporarily at `/dev/shm/xinfer-cond-20261030/validation-v4`; shared-memory files
can disappear after restart. The existing C++ runner needs no source change.

This run uses the development L20 with Torch `2.7.0+cu126`, Triton `3.3.0` and
C++11 ABI enabled. The GPU is fully loaded; no timing claim is made. The pinned
separate alpha Torch/Triton 3.2 environment still needs its own functional run
before using this dispatcher there. Full-model integration is a later step.

## Chunked GDN prefill PoC: 2026-10-30

The working tree based on `c25d154` replaces token-sequential prefill with a
forward-only chunked GDN implementation. Batch one, 16 heads, K=V=128 and
T=2..512 remain the prefill contract. The T=1 fused decode kernel and its
helpers are AST-identical to the committed implementation. Both paths keep
the same `xinfer::delta_fused` eight-input/three-output interface, functional
state ownership and existing C++ runner; no new backend flag or C++ source
change is required.

### Algorithm and precision

`kernels/recurrent.py` prepares convolution/SiLU output, normalized/scaled Q/K
and gates in parallel across tokens and heads. `kernels/chunked.py` then runs
four kernels:

1. Compute chunk-local cumulative decay and the unit-lower-triangular transform.
2. Compute U/W transforms using FP32 matrix products.
3. Propagate FP32 memory between chunks, storing incoming states and corrected
   values for each chunk.
4. Compute outputs in parallel across chunks, heads and value tiles.

This is five launches per DeltaNet prefill operator, rather than one larger
token loop. Chunk size is 64; the runtime chunk count is `ceil(T/64)`, so one
dynamic package covers all supported lengths. Prepared Q/K/V, beta and g are
masked to neutral tail values (`beta=0`, `g=0`); padding does not advance the
returned memory, and convolution history ends at the last real token.
Intermediate buffers and matrix products are FP32 with explicit IEEE dot
precision. Convolution/SiLU and beta retain the original input-dtype rounding;
the initial PoC introduces no BF16 matrix operands or TF32 arithmetic.

Metadata records `delta_algorithm={prefill: chunked_fp32, decode: fused_recurrent}`,
`delta_chunk_size=64` and `delta_dot_precision=ieee`. Source manifests include
`kernels/chunked.py`. The operator/state ABI and dynamic prefill range remain
unchanged. This is an algorithm/correctness PoC, with no performance claim.

### Focused validation

The 83-case suite executes 400 calls covering BF16/FP32, zero/nonzero incoming
state, lengths through 512, chunk and convolution boundaries, mixed prefill and
decode, repeated/reset requests, cancellation, checkpoint projections, and
decode/prefill gate extremes. HF recurrence and HF's chunked algorithm are
separate controls, alongside the independent full-FP32 control.

| Check | Result |
| --- | --- |
| Fused/chunked implementation vs HF recurrent and chunked controls | Existing numerical limits pass; `fusion_passed=True` |
| Python JIT vs Python AOTI vs standalone C++ | All 1,200 output/state tensors across 400 calls bitwise equal |
| Convolution history | Exact, including tails and sequence partitions |
| Core/recurrent state across sequence partitions | Pass existing precision-specific numerical limits |
| Input preservation, aliasing, shapes/dtypes/finites, streams, repeated requests and invalid-input rejection | Pass |

Chunking changes floating-point operation order, so partition equivalence now
uses the existing core/recurrent numerical limits rather than bitwise state
requirements. Same-input JIT/AOTI/C++ and repeated requests remain exact, as
does convolution history. No numerical threshold was enlarged.

Full-FP32 acceptance remains open: fused-vs-FP32 and HF-vs-FP32 each record
127 failures. All 254 comparisons remain visible, overall `passed=False` and
exit 1 are retained. Equal counts do not establish identical error metrics.

### Full-model export and numerical checks

Strict export and AOTI compilation pass for both complete model entry points.
Prefill has 2,166 graph nodes; decode has 2,165. Both retain 18
`xinfer.delta_fused` calls, 48 packing groups and 187->97 linear calls.
Generated wrappers each contain 97 external matrix calls and zero external
convolution or proxy-executor calls.

All 68 model-suite steps pass shape/dtype/finite, input preservation,
reset/repeat, future-KV/write-region and masking checks, including dynamic
lengths 2,22,65,511,512. The numerical report retains 5,219 failures versus
the previous unified implementation's 5,241, with 10 added and 32 removed
failure keys. Reported top-1-difference rows increase from 10 to 14, including
a new `aoti_vs_hf` difference at `dynamic_32`, step 1, with a control top-two
margin of 0.125. These rows can cover repeated comparisons; they are not a
count of distinct prompts. Accuracy acceptance remains open, and the lower
failure total does not establish an accuracy improvement.

Full-model native validation passes 27 cases: 4,018 teacher tensors, 28 greedy
logit rows and 288 final-state tensors match Python AOTI bitwise. All 23
rejection cases and input/reset/repeat, CLI and dispatch checks pass. The native
process loads no Python library. Relative to the previous fused package pair,
the capital response removes the period (`Paris` instead of `Paris.`), with the
same change in its repeated request. Arithmetic (`42`) and Chinese responses
remain unchanged. Native parity establishes agreement with the current AOTI
packages, while the HF/FP32 numerical issues remain open.

### Evidence and performance scope

Compact reports belong under
`artifacts/qwen35/0.8b/chunked-validation/summary.json`,
`artifacts/qwen35/0.8b/chunked-model/summary.json` and
`artifacts/qwen35/0.8b/chunked-runtime/summary.json`. The new model package pair
is temporarily in `/dev/shm/xinfer-chunked-20261030/text`; large fixtures and
compiler scratch also use temporary storage because development disk space is
limited. These files can disappear after restart; re-export on the target
host with its matching Torch/LibTorch and GPU architecture.

The shared development GPU is at 100% utilization, so no latency or speedup is
claimed. Further profiling should compare the committed token-sequential
baseline and this chunked candidate with the same Inductor optimization flags,
precision settings, workload and native build. Existing full-model numerical
differences must remain explicit during that comparison.

# Qwen3.5: text-only AOTI / C++ experiment

Original milestone summary (2026-10-27): the primary target is **Qwen/Qwen3.5-0.8B**. Steps 1-2 pass:
single DeltaNet block export/compile, Python-free C++ execution and extended
state/sequence tests. The existing T=64/T=1 packages handle inputs of 1-512 tokens
through chunk/remainder dispatch. Shapes come from config/package metadata;
the same runner still passes with the existing 4B packages. Keep 4B for later
scaling and additional-head-layout checks. Step 3 now exports, compiles and runs
the full text model in Python. Its state semantics checks pass, but numerical
acceptance remains open: 854 output-state comparisons against HF exceed the
unchanged limits. Step 4 passes native/Python AOTI parity, with a Python-free
C++ generation loop and offline text helpers. Step 5's expanded validation and
three-profile benchmark are implemented and functionally checked. HF numerical
acceptance remains open; final timing runs belong in the user's separate idle
environment. The 0.8B experiment is kept together in one implementation commit.

## Scope and approach

Use one GPU, batch size 1, BF16 weights/activations and FP32 DeltaNet recurrent
state. Start with the upstream PyTorch implementation and small inference/state
adapters, reusing HF modules and weights. Replace selected child modules before
export when needed; avoid a full model fork or a general adapter framework.

Python handles checkpoint loading, offline tokenization/decoding, export and
compilation under `compiler/qwen35/`. C++ under `runtime/qwen35/` loads compiled
AOTI packages and performs all prefill, decode, state management and greedy
token generation. Keep Torch/LibTorch versions matched and preserve the separate
Qwen3-0.6B environment. Initial full-model limits: prompts of 1-512 tokens and a
total context budget of 1,024 tokens. Vision, quantization, serving, batching and
performance kernels are outside the initial correctness milestone.

## Implementation sequence

1. **Retarget the single DeltaNet block probe to 0.8B (complete).** Pin the checkpoint revision
   and verify its weights. Derive dimensions from its config and pass them through
   package metadata to C++; remove the current 4B constants from both sides.
   Keep one implementation usable by both sizes. Start with prefill T=64 and
   decode T=1. Repeat HF eager, adapter, Python AOTI, native C++ and FP32-control
   comparisons on outputs and both state tensors. Preserve the completed 4B
   evidence and use separate output directories for 0.8B.
2. **Establish state and sequence-length handling (complete).** Test short inputs, convolution
   boundaries (3/4/5), chunk boundaries (63/64/65 and 127/128/129), longer decode
   chains, prefill/decode handoff and request reset. Compare different valid chunk
   partitions of the same sequence. Try bounded dynamic prefill; if the HF chunk
   implementation cannot export that cleanly, use fixed 64-token chunks with
   T=1 remainder calls first. Do not advance recurrent state through padding.
3. **Export the full 0.8B text model (implemented; numerical acceptance open).** Combine embedding, all 24 decoder layers,
   final norm and LM head through a thin tensor-only adapter. Expose 18 pairs of
   convolution/recurrent states and six KV-cache pairs. Document state ownership,
   positions, causal masking and update semantics. Compile prefill and decode,
   validate them in Python and record model/shape/dtype/toolchain metadata.
4. **Add the minimal C++ generation loop (complete: native/AOTI parity).** Load both packages, allocate state
   from metadata, execute prompt prefill and repeated decode, apply greedy token
   selection, and stop at configured EOS or limits. Add the offline Python text
   helper and reject invalid inputs/package combinations. Verify a fresh native
   process runs without Python libraries.
5. **Validate end to end and measure the baseline (functional work complete; numerical acceptance and final timings open).** Compare HF eager, Python AOTI
   and C++ with identical teacher-forced tokens and greedy-generation fixtures.
   Check logits, KV/conv/recurrent state, context boundaries, resets and repeated
   requests. Require native/Python AOTI bitwise parity; retain BF16 error metrics
   and focused FP32 controls, investigating failures without silently relaxing
   limits. On an idle GPU, measure compile/load cost, package sizes, prefill
   latency, decode latency/throughput and memory against the same eager workload.
6. **Profile and optimize the working pipeline.** Once full text generation and
   correctness checks pass, refactor the profiling tools for Qwen3 and Qwen3.5.
   Profile prefill and decode separately, then evaluate Inductor options, native
   CUDA Graphs and suitable existing DeltaNet/convolution/attention kernels.
   Prefer small module adapters for custom-op substitution. Validate and measure
   each change independently against the preserved baseline. Finally rerun 4B
   to test scaling and its different DeltaNet head layout.

## Step 5: end-to-end validation and benchmark

Functional work is complete on the shared L20 (2026-10-27). Numerical acceptance
is **not passed**, and no performance baseline is claimed. The user will run
the benchmark in a separate environment. Production model packages and their
precision settings are unchanged; no optimization was introduced in this step.

### Validation result

`validate_end_to_end.py` compares original HF eager with both natural full-prompt
calls and the compiled path's 64/1 partitions, Python AOTI, and a fresh native
process. `reference.py` keeps the independent HF cache/mask handling in one small
helper shared by validation and benchmarking. The native validation driver also
accepts poisoned future KV slots to check that masking isolates them.

The 21 native cases include teacher-forced prompt lengths 1/2/63/64/65/127/128/129/
511/512, eight subsequent decode steps (64 for length 512), the last context
slot, poisoned future KV, reset/repeat, and three ordinary chat prompts. A
separate original-weight FP32 model covers 73 steps: length 512, the context
boundary and the France teacher-forced sequence.

| Check | Result |
| --- | --- |
| C++ vs Python AOTI teacher-forced outputs/state | All 9,947 tensors across 203 steps bitwise equal |
| C++ vs Python AOTI greedy generation | IDs, 22 logit rows and 144 final state tensors bitwise equal |
| AOTI vs HF with matching partitions | 203/203 teacher-forced top-1 predictions; 3/3 greedy sequences equal |
| AOTI vs HF with full-prompt calls | 202/203 top-1 predictions; 2/3 greedy sequences equal |
| Masked future KV, reset/repeat, input preservation, consumed positions, context overflow | Passed |
| Native process Python-library check | Passed |
| Unchanged 5% numerical limits | Failed for conv/recurrent states; logits and K/V pass |

There are 3,466 failing state comparisons against matched-partition HF (3,451
conv and 15 recurrent), and 3,500 against full-prompt HF (3,451 conv and 49
recurrent). These are comparison entries, including repeated fixtures, not
6,966 distinct tensors. The validator retains all metrics and exits **1**.

| Worst error over checked groups | AOTI vs HF chunked | AOTI vs HF full prompt |
| --- | ---: | ---: |
| Logit relative L2 | 3.1356% | 3.2270% |
| Recurrent-state relative L2 | 4.6169% | 5.7811% |
| Recurrent-state peak-relative | 9.1719% | 9.2499% |

The large conv relative errors remain dominated by near-zero reference channels;
the limits have not been relaxed. On the 73 FP32-control steps, worst logit
relative L2 is 2.3753% for AOTI and 2.6459% for HF. These diagnostics neither
close Q35-NUM-001 nor establish general accuracy superiority.

Evidence: `artifacts/qwen35/0.8b/end-to-end/` contains `summary.json`, fixtures,
native dumps, source snapshots/hashes and final HF/AOTI/FP32 context-state dumps.
Step-4 binaries and sources are preserved in `work/qwen35/0.8b/step4-source/`,
because the native validator was rebuilt to support the poison checks.

### Open issue Q35-NUM-002: partition-sensitive greedy decision

For the France prompt, AOTI and matched-partition HF generate `Paris`, while
full-prompt HF generates `Paris.` (surrounding whitespace/special tokens omitted).
At generated token index 1, AOTI chooses `248046` (end of message), and full-prompt
HF chooses `13` (period). HF's top-two margin and its score gap to token 248046
are both **zero**. HF itself exhibits this partition difference, so it cannot
be attributed exclusively to compilation. Arithmetic and Chinese fixtures match
on every path. This remains an exact-greedy acceptance failure; no tie exception
or tolerance change was added. The retained report records token IDs and margins.

The separate fixed-token benchmark also exposes one AOTI/HF difference after
consuming `[248045, 57590]`: AOTI and FP32 choose `11`, HF BF16 chooses `248046`,
with HF score gap 0.0625. C++ and Python AOTI predictions agree. Logit relative
L2 is 2.0491% AOTI/HF, 1.4315% AOTI/FP32 and 1.3691% HF/FP32. This is a different
fixture from the partition tie above, and its precise cause is not isolated.
Evidence is `benchmark-prediction-check.json` under the same artifact root;
the focused reproducer is `work/qwen35/0.8b/check_predictions.py`. Matching one
FP32 token is not sufficient to dismiss a BF16 parity failure.

### Benchmark contract and functional checks

`benchmark.py` runs fresh processes for three profiles, rotating their order
over three rounds by default:

| Profile | Computation |
| --- | --- |
| `hf` | Original HF model with one full-prompt call, then single-token decode |
| `hf_chunked` | Original HF model with the same 64/1 dispatch as AOTI |
| `cpp_aoti` | `runtime/qwen35/benchmark.cpp`, using the generation CLI's `Runtime` |

All profiles use identical prompt and teacher-forced continuation IDs. EOS is
ordinary input data here, so every profile performs the same number of calls.
The timed prediction includes argmax and its host read. Tokenization, model
loading and report writing are outside steady-state request timing. One CPU
thread is used and TF32 is disabled. No CUDA Graphs or custom kernels are enabled.

Report fields separate reset, prefill, reset-plus-prefill (`first_prediction_ms`),
decode per token and total request time. HF lazily allocates state during prefill;
native state allocation is charged to reset. Compare `first_prediction_ms` when
including both costs. CUDA initialization, load and each case's first request
are retained separately from warmup/measured samples; only the first case starts
with a freshly loaded process. HF load includes checkpoint checks/construction
and transfer; native load includes both packages and initial state allocation.
Metadata retains package size and observed export/compile time; this is not a
controlled cold-cache compilation benchmark.

Memory reports distinguish allocator allocated/reserved peaks from sampled
device-wide usage. The latter includes other processes and can miss transient
peaks. Two native loaders retain independent package weight pools. Reports also
save the exact workload, source snapshots, binary/package hashes, environment,
per-worker GPU telemetry and the matching end-to-end acceptance report.
Prediction differences between profiles are recorded; within-profile repeats
must match. Functional success never changes failed numerical acceptance to pass.

The shared-host functional matrix passes all three profiles at lengths
1/22/65/128/512, 64 decode steps, one warmup and two measured repeats. A second
matrix covers lengths 64/511 and all three process orders, with physical GPU UUID
selection. Together these execute 132 requests. Of 65 predictions per case,
AOTI matches chunked HF on 64 at length 1 and all 65 at the other tested lengths.
Full-prompt HF also differs at length 22 (64/65 agreement). All results remain
`provisional_shared_gpu`; their latency values are not a speedup claim.

Evidence: `benchmark-functional/` and `benchmark-order-check/` under the artifact
root. The final script also passes a nine-request, three-profile rerun in
`benchmark-final-smoke/`. An idle-guard variable-shadowing bug found during validation was fixed;
`benchmark-idle-check-v2.json` records the intended busy-GPU rejection after 30
samples. By default the runner requires five consecutive utilization samples
at or below 10% before and after each worker. This checks run boundaries, not
exclusive GPU ownership. `--allow-busy` explicitly marks a functional-only run.

### Run in the benchmark environment

Use the sources from this experiment commit as well as the pinned prepared text directory
`work/qwen35/0.8b/text/` (weights, config, tokenizer and manifest). Alternatively,
run `compiler/qwen35/model.py` there to prepare it. The manifest is verified
when loading. Create an isolated environment with the pinned dependencies in
`requirements.txt` and a CUDA-enabled Torch installation; the development host's
`.pth` shortcut is specific to that host. Optional `fla` and `causal_conv1d` must
remain absent for this baseline. If the target GPU needs a newer Torch/toolchain,
treat that as a new configuration and rerun correctness before interpreting times.

Re-export on the target GPU/toolchain and rebuild against that same Torch's
LibTorch. The existing SM89 packages/binaries are not a portable deployment bundle.
Run these commands from the repository root; use fresh output paths on reruns:

```bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
QWEN_PY=work/qwen35/venv/bin/python
export TORCHINDUCTOR_CACHE_DIR="$PWD/work/qwen35/0.8b/inductor-target"
"$QWEN_PY" -m compiler.qwen35.export decode --output artifacts/qwen35/0.8b/text-target
"$QWEN_PY" -m compiler.qwen35.export prefill --output artifacts/qwen35/0.8b/text-target

QWEN_CMAKE_PREFIX=$("$QWEN_PY" -c 'import torch; print(torch.utils.cmake_prefix_path)')
QWEN_SM=$("$QWEN_PY" -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')
cmake -S . -B work/qwen35/build-target -G Ninja \
  -DCMAKE_PREFIX_PATH="$QWEN_CMAKE_PREFIX" -DTORCH_CUDA_ARCH_LIST="$QWEN_SM" \
  -DCMAKE_BUILD_TYPE=Release -DQWEN_BUILD_QWEN35=ON \
  -DQWEN_BUILD_VALIDATION=ON -DQWEN_BUILD_BENCHMARK=ON
cmake --build work/qwen35/build-target --target qwen35_aoti qwen35_validate qwen35_benchmark -j2

"$QWEN_PY" -m compiler.qwen35.tests.validate_end_to_end \
  --packages artifacts/qwen35/0.8b/text-target \
  --native work/qwen35/build-target/qwen35_validate \
  --output artifacts/qwen35/0.8b/end-to-end-target
```

Inspect the validation report even if it exits 1 for the known numerical issues.
The benchmark copies that failed status into its own report; do not interpret
functional timing execution as numerical acceptance. With an otherwise idle GPU:

```bash
"$QWEN_PY" -m compiler.qwen35.benchmark run --gpu 0 \
  --packages artifacts/qwen35/0.8b/text-target \
  --native work/qwen35/build-target/qwen35_benchmark \
  --validation artifacts/qwen35/0.8b/end-to-end-target/summary.json \
  --output artifacts/qwen35/0.8b/benchmark-target
```

Defaults: prompt lengths 22/64/65/128/511/512, 64 decode steps, two warmups, five
measured repeats and three rounds. `--gpu` selects a physical `nvidia-smi` index
or UUID; each worker exposes only that GPU as CUDA device 0, and telemetry uses
the same UUID. Keep export/validation on that GPU too. For a short harness-only
check, add `--lengths 22 65 --decode-steps 4 --warmup 1 --repeats 2 --rounds 1
--allow-busy` and choose another output directory. Final measurements must omit
`--allow-busy` and retain the whole result directory, not only `summary.json`.

## Step 3: full text model

`text_model.py` extracts the original text weights from the pinned 0.8B
checkpoint, checks the source shard against its Hub SHA256, and loads all 24 HF
decoder layers, embeddings, final norm and tied LM head. There are 752,393,024
text parameters. Vision and MTP weights are excluded. Weights and activations
use BF16; RoPE frequency buffers and DeltaNet recurrent states remain FP32.
The original HF modules perform all model arithmetic. The adapter only supplies
tensor state, positions, a causal mask and the last-token logits interface.

```text
step(input_ids, cache_position, state) -> last_logits, *next_state
```

- Batch size is one. `input_ids` is contiguous CUDA int64 `[1,T]` and
  `cache_position` is contiguous CUDA int64 `[T]`. T is exactly 64 for prefill
  or one for decode. Positions are consecutive append positions starting at zero
  for each request; the caller must enforce the 1,024-token capacity.
- `state` is a list of 48 contiguous CUDA tensors, two per decoder layer in
  layer order. Each of 18 DeltaNet layers receives BF16 conv `[1,6144,4]` and
  FP32 recurrent `[1,16,128,128]` tensors. Each of six full-attention layers
  receives BF16 K/V `[1,2,1024,256]` tensors. State occupies 30.84375 MiB.
- Zero state starts a request. The interface preserves its inputs; returned
  state becomes the next call's input. Conv storage is cloned and K/V updates
  use out-of-place `index_copy`. This deliberately permits allocation and cache
  copies for the initial correctness baseline.
- Full attention uses the original HF SDPA path with the memory-efficient CUDA
  backend and an explicit additive causal mask over all 1,024 cache slots.
  Query position `p` can read only slots `0..p`. DeltaNet receives no padding
  mask. Text positions use identical coordinates on all three RoPE axes.
- Output zero is BF16 last-token logits `[1,248320]`, followed by the 48 next
  states. The flattened AOTI/C++ ABI therefore has 50 inputs and 49 outputs.
  Package metadata records the order, dimensions, dtypes and model/toolchain
  provenance. No custom operator library is required.

`export_text.py` strictly exports and AOT-compiles each fixed entry point into
a `.pt2` package. `validate_text.py` compares the original HF model with its
own StaticCache, the eager adapter, loaded AOTI packages and an independent
FP32 reference. A separate HF DynamicCache run checks fixed-capacity attention
against attention over the valid prefix. Six teacher-forced cases cover 50
steps: decode from zero, 64-token prefill plus decode, a 65-token remainder,
two prefills, a 512-token prefix plus decode, and request reset/repeat.

```bash
# Once, after the existing step-1 environment setup:
work/qwen35/venv/bin/python -m compiler.qwen35.model
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/work/qwen35/0.8b/inductor-text"
work/qwen35/venv/bin/python -m compiler.qwen35.export decode
work/qwen35/venv/bin/python -m compiler.qwen35.export prefill
work/qwen35/venv/bin/python -m compiler.qwen35.tests.validate_model
```

Preparation refuses an existing model directory; export refuses an existing
package; validation refuses an existing result directory. Use fresh `--output`
directories for reruns and `--packages` to select another compiled pair.
The validator exits nonzero when numerical acceptance fails and retains its
report. Full-model native execution, greedy generation and context-limit tests
are covered by the step-4 result below; step 3 alone does not establish those
results or GPU performance.

### Observed result — 2026-10-27

Strict export and AOTI compilation succeed without graph breaks or graph-signature
input/buffer mutations. An initial attempt to construct `SimpleNamespace` inside
the traced adapter was rejected; using an ordinary small `LayerState` class
resolved it without changing any model arithmetic.

| Entry point | Exported nodes | Export time | Compile time | Package bytes |
| --- | ---: | ---: | ---: | ---: |
| Decode T=1 | 3,605 | 6.60 s | 59.87 s | 1,506,903,652 |
| Prefill T=64 | 19,211 | 27.31 s | 149.98 s | 1,509,608,510 |

Both files are compiled AOTI packages, about 1.4 GiB each, with separate weight
pools when both are loaded. Build times are observations on a shared host, not
controlled compiler or inference benchmarks.

| Check | Result |
| --- | --- |
| Eager adapter vs original HF StaticCache | All 2,450 logits/state tensors bitwise equal |
| HF DynamicCache vs StaticCache control | All 441 tensors bitwise equal |
| AOTI vs HF next-token argmax | 50/50 teacher-forced steps agree |
| AOTI logits and all full-attention K/V states | Pass unchanged 5% limits |
| AOTI conv/recurrent states | **Fail:** 850 conv and four recurrent tensors exceed limits |
| Poisoned future KV ignored; past/future KV write regions preserved | Passed |
| Inputs preserved; finite values, shapes/dtypes; request reset/repeat | Passed |

The validator compares AOTI against both HF and the identical eager adapter, so
its report contains 1,708 failure entries for those 854 distinct tensors. It
exits with status 1. Top-1 agreement under teacher forcing does not establish
greedy generation parity or close cache-state acceptance.

Worst AOTI/HF relative L2 is 1.3690% for logits, 1.8532% for keys, 2.4940% for
values and 4.4172% for recurrent states. Recurrent peak-relative error reaches
7.2500%. Conv comparisons use the existing per-channel, four-value metric;
near-zero channels can have very large relative errors. HF itself also has
large per-channel relative errors against FP32. This does not waive the test:
all raw errors and unchanged limits remain recorded. Worst logits relative L2
against the independent FP32 reference is 1.3995% for AOTI and 1.3042% for HF.

The historical diagnostic `--eager-numerics` option retained native
`aten.mean.dim` and disabled FP contraction, as in the Qwen3 experiment. A
complete second run still failed the same 854 tensors; this option was **not**
an accepted correctness fix. The current exporter has retired this option and
uses the ordinary Inductor path. The commands below record the historical run;
legacy eager-numerics packages require re-export with the current exporter.

```bash
export TORCHINDUCTOR_CACHE_DIR="$PWD/work/qwen35/0.8b/inductor-text-eager"
work/qwen35/venv/bin/python -m compiler.qwen35.export decode --eager-numerics \
  --output artifacts/qwen35/0.8b/text-eager-numerics
work/qwen35/venv/bin/python -m compiler.qwen35.export prefill --eager-numerics \
  --output artifacts/qwen35/0.8b/text-eager-numerics
work/qwen35/venv/bin/python -m compiler.qwen35.tests.validate_model \
  --packages artifacts/qwen35/0.8b/text-eager-numerics \
  --output artifacts/qwen35/0.8b/text-eager-validation
```

### Open issue Q35-NUM-001: lost BF16 rounding boundary

Status: **open**, recorded on 2026-10-27. Preserve this issue while implementing
the native runtime; C++/Python AOTI parity does not resolve HF numerical acceptance.
The affected environment is Torch 2.7.0+cu126, Triton 3.3.0, Transformers 5.20.2
and NVIDIA L20 SM89, using the pinned 0.8B checkpoint above.

Focused layer-0 decode probes locate the first BF16 activation difference at
the DeltaNet output: embeddings and input RMSNorm are exact; 588/1,024 DeltaNet
output elements differ (relative L2 0.2629%). Retaining native sum reductions
makes that probe's recurrent state exact but leaves its output difference.
An intermediate-output probe then identifies a lost rounding boundary before
gated RMSNorm: the BF16 core output agrees, but its FP32 conversion uses the
unrounded producer values. All 2,048 converted values differ; rounding the
compiled FP32 values through BF16 reproduces eager exactly. The discrepancy
persists with `emulate_precision_casts=True` and with pattern matching disabled.
Smaller FP32 differences also occur in exponential/softplus/SiLU operations.
This isolates a contributing cause, not every full-model numerical discrepancy.

Forcing native `aten._to_copy` in the focused probe compiles but its package
fails to load with `[json.exception.type_error.302] type must be string, but is
number` in this Torch 2.7 environment. That experimental workaround is not in
the implementation. The exact compiler pass responsible has not been isolated;
the failed pattern-matcher control is not evidence that a particular graph pass
caused it. No production model fork, graph rewrite or custom kernel has been
added to hide these failures.

The retained `probe_rounding.py` makes this observation reproducible. It uses
the original layer-0 DeltaNet with the normalized embedding of token `The`,
zero state, native mean/sum reductions and disabled FP contraction. It exposes
intermediate outputs only in a diagnostic graph and compares execution before
and after AOTI compilation with `emulate_precision_casts=True`.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/work/qwen35/0.8b/inductor-layer-probe"
work/qwen35/venv/bin/python -m compiler.qwen35.diagnostics.probe_rounding
```

The recorded run in `artifacts/qwen35/0.8b/rounding-probe/` confirms:

| Boundary check | Result |
| --- | --- |
| BF16 input to gated RMSNorm (`reshape_4`) | Exact |
| Conversion back to FP32 (`to_11`) | 2,048 / 2,048 elements differ |
| Maximum absolute difference at that conversion | 0.000015905126929283142 |
| Compiled FP32 values rounded through BF16 vs eager FP32 values | Exact |

`summary.json` records package, checkpoint, source and HF implementation hashes;
`boundary.pt` holds the four compared tensors; `graph.py` and `probe.pt2` retain
the diagnostic graph and compiled package. Exit zero means the known issue was
reproduced, **not** that numerical acceptance passed. Use a fresh `--output` on
reruns. This is evidence of a missing rounding boundary, not proof that every
full-model discrepancy shares one cause or that eager BF16 is closer to FP32.

Next investigation: isolate explicit dtype-conversion lowering/fusion in a small
model-independent reproducer, then test a targeted compiler fix or a pinned newer
toolchain. Any candidate must rerun the complete HF/AOTI/native checks without
relaxing the existing limits. This investigation remains separate from step 4.

Evidence is retained under `artifacts/qwen35/0.8b/text/`, `text-validation/`,
`text-eager-numerics/` and `text-eager-validation/`. The validation reports contain
all per-step metrics, package/checkpoint/source hashes and final HF/AOTI/FP32
state dumps. Focused diagnostic scripts/logs are in `work/qwen35/0.8b/probe_*`
and `probe-*.log`; their packages/value dumps are under the same artifact root.
The initial source snapshot is in `work/qwen35/0.8b/text-initial-source/`.

## Step 4: native text generation

`runtime/qwen35/runtime.h` contains the metadata contract and a small
stateful runtime. `text_main.cpp` supplies the token-ID command line. Both model
packages load once. Each request starts from zero state and absolute position
zero; complete 64-token chunks use prefill and the remaining tokens use decode.
The loop selects the largest logit, returns only newly generated token IDs,
and stops at the package's configured EOS or `--max-new-tokens`.

The caller owns the returned conv/recurrent/KV tensors and passes them into the
next package call. No hidden tokens are padded. The last predicted token is not
fed back after stopping, so consumed positions equal prompt length plus generated
length minus one. The conservative budget check requires prompt length plus the
requested generation limit to fit within 1,024 tokens. Prompts contain 1-512 IDs.

Before loading weights, the CLI validates the request and package metadata:
entry points, state order/shapes/dtypes, compatible checkpoint and compiler
settings, LibTorch version, C++ ABI and file sizes. GPU architecture is checked
before constructing loaders. Metadata and its corresponding `.pt2` files must
stay together; the Python validator additionally verifies package SHA256s.
Existing output/trace files are refused. Optional `--trace` writes raw FP32
logit rows, one vocabulary row per generated token.

`prepare_text.py` applies the checkpoint's official non-thinking chat template
offline and writes whitespace-separated IDs. It explicitly requests a list from
Transformers 5's chat-template API. Its decode command reads generated IDs and
prints text; neither helper performs model inference. The native executable
does not tokenize, import Python, embed an interpreter or call a Python process.

```bash
/data/workspace/abo/.venv/bin/cmake -S . -B work/qwen35/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/data/workspace/abo/.venv/lib/python3.12/site-packages/torch/share/cmake \
  -DQWEN_BUILD_QWEN35=ON -DQWEN_BUILD_VALIDATION=ON
/data/workspace/abo/.venv/bin/cmake --build work/qwen35/build \
  --target qwen35_aoti qwen35_validate -j2

work/qwen35/venv/bin/python -m compiler.qwen35.prepare encode \
  --prompt 'What is the capital of France? Answer briefly.' \
  --output work/qwen35/0.8b/demo/prompt.ids
work/qwen35/build/qwen35_aoti artifacts/qwen35/0.8b/text \
  work/qwen35/0.8b/demo/prompt.ids work/qwen35/0.8b/demo/generated.ids \
  --max-new-tokens 24
work/qwen35/venv/bin/python -m compiler.qwen35.prepare decode \
  --input work/qwen35/0.8b/demo/generated.ids

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 work/qwen35/venv/bin/python \
  -m compiler.qwen35.tests.validate_runtime
```

`runtime/qwen35/tests/text_validate.cpp` is a separate driver using the same
runtime. It tests multiple requests with one loader pair, captures tensors,
checks input preservation and inspects process mappings for Python libraries.
The Python validator compares native tensors bitwise with Python-loaded AOTI,
checks the CLI independently, and exercises request/package rejection paths.
Its report distinguishes native runtime acceptance from the still-open HF
numerical acceptance. Step 5 will extend HF end-to-end comparisons and measure
performance on an idle GPU; this runtime validation is not a benchmark.

### Step 4 result — 2026-10-27

The 18-case matrix uses one loader pair across requests. It includes all six
step-3 teacher-forced cases, native dispatch of 2/63/65/129/511-token requests,
the final legal position 1,023 followed by a rejected overflow, three chat
prompts, a repeated chat, a one-token generation limit, and a test-only EOS
override that stops on the first prediction. Actual model EOS is also reached
in all three ordinary chat fixtures.

| Check | Result |
| --- | --- |
| Teacher-forced C++ vs Python AOTI | All 3,136 tensors across 64 checked steps bitwise equal |
| Greedy C++ vs Python AOTI | All 28 logit rows, generated IDs and 288 final state tensors bitwise equal |
| Request reset/repeat, state handoff, consumed positions, input preservation | Passed |
| Last legal context slot and overflow rejection | Passed |
| EOS and generation-limit stopping | Passed |
| CLI invoked independently with Python environment removed | IDs and logits bitwise equal |
| `ldd` and native `/proc/self/maps` checks | No `libpython` or `libtorch_python` |
| Invalid request/package rejection checks | All 18 passed |

Rejection checks cover empty/oversized prompts, malformed/negative/out-of-range
IDs, invalid or overflowing generation budgets, output overwrite, mixed
checkpoint/state/numerical metadata, wrong entry-point length, Torch version,
state dtype, tensor ABI order and GPU architecture. Invalid requests produce
no output file, and the overwrite check preserves its existing input file.

The standalone CLI's France prompt uses 22 input tokens and returns four tokens,
including configured EOS 248044; it consumes 25 positions. Offline decoded
examples, excluding special tokens and surrounding whitespace:

| Prompt | Native output |
| --- | --- |
| What is the capital of France? Answer briefly. | Paris |
| What is 17 + 25? Answer with only the number. | 42 |
| 用一句话解释什么是编译器。 | 编译器是将源代码翻译成机器码的自动化过程。 |

Evidence is in `artifacts/qwen35/0.8b/runtime-validation-v2/`: `summary.json`,
fixture definitions, native state/logit dumps, CLI and rejection logs, linkage
reports, source/binary/package hashes and decoded results. A separate CLI smoke
run is in `artifacts/qwen35/0.8b/runtime-smoke/`. The initial helper attempt
exposed Transformers 5's dictionary return default; `return_dict=False` fixes
the helper, and the successful matrix uses the corrected integer token lists.
The original model packages and the step-3 failure reports are unchanged.

The report explicitly sets `native_runtime_passed: true` while keeping
`hf_numerical_acceptance: open`. This result does not close Q35-NUM-001 or the
854 step-3 HF state failures. The GPU remained busy; no performance baseline is
claimed. The 0.8B implementation and retained diagnostics belong to one experiment commit.

## Step 2 result: sequence and state handling

### Dynamic export probe and selected path

`probe_dynamic.py` tries strict export with a sample T=32 and bounded dynamic
ranges T=2..512 and T=2..64. Both fail with `UserError: Constraints violated
(tokens)` in Torch 2.7 / Transformers 5.20.2. Guards generated from HF's modulo
padding, reshapes and sequential chunk scan ultimately require `tokens = 32`.
The single-chunk range also fails; a smaller bound alone does not resolve it.
This is a dynamic-shape export limitation. Fixed T=64/T=1 still exports and
compiles as full graphs, and its existing packages are reused without changes.

The minimal baseline uses `sequence.py::run_sequence` and the equivalent C++
function: consume each complete 64-token chunk with prefill, then consume the
remaining 0-63 tokens through decode. For example, T=129 is 64+64+1 and T=63 is
63 decode calls. Each call passes its returned conv/recurrent state to the next;
all token outputs are concatenated. External hidden-token padding is never added.
This trades extra launches on short/tail inputs for a small, explicit contract.
No speedup is claimed; dynamic export or additional buckets can be revisited
during optimization if profiling justifies them.

### Validation

`validate_block.py --sequence-tests` covers 18 cases and 666 checked sequence
steps, dispatching 34 prefill and 918 decode package calls per compiled path:

- Prefix lengths 1/2/3/4/5, 63/64/65, 127/128/129 and 511/512, each followed by
  four decode steps. These cover convolution and chunk boundaries.
- The same 65-token prefix as one request versus 65 single-token requests;
  the same 129-token prefix as one request versus 1+64+64 and 3+61+65 requests.
  Compare all token outputs and both states at the prefix and final boundaries.
- A 512-token prefix followed by 512 decode steps; a repeated 65-token request
  after that long request verifies reset and reuse across different workloads.

HF runs twice: once with the supplied request lengths, and once partitioned
exactly like the compiled path. This separates partition-induced BF16 differences
from adapter and compiler differences. The original checkpoint also runs in FP32.

| Check | Result |
| --- | --- |
| C++ vs Python AOTI | All 1,998 output/state tensors bitwise equal |
| Adapter vs HF with matching partitions | All 1,998 tensors bitwise equal |
| AOTI vs HF supplied request lengths | All unchanged 5% relative limits pass |
| Alternate partitions | All BF16 comparisons pass; FP32 diagnostics retained |
| Input preservation, shapes/dtypes, finite values, reset/repeat | Passed |
| Native Python-free linkage/loading | Passed |
| Metadata and request rejection checks | Seven passed |

Worst errors across all 666 checked steps:

| Tensor | AOTI vs HF relative L2 | AOTI vs HF peak-relative | AOTI vs FP32 relative L2 | HF vs FP32 relative L2 |
| --- | ---: | ---: | ---: | ---: |
| Output | 0.9043% | 2.1834% | 1.1041% | 1.1535% |
| Convolution state | 0.5451% | 0.7519% | 0.3849% | 0.3849% |
| Recurrent state | 0.6470% | 0.6414% | 1.0847% | 1.0847% |

With matching partitions, compiled convolution states are bitwise equal to HF;
worst recurrent-state relative L2 is 0.0005064%. HF itself shows BF16 differences
when partitioning changes, so cross-partition equality is numerical rather than
bitwise. The unchanged per-token/per-channel/per-head limits apply. FP32 controls
are diagnostic, and all metrics remain in `summary.json`.

Native checks reject empty/oversized requests and multi-token decode requests,
in addition to mismatched hidden/state dimensions, wrong dtype and wrong strides.
The updated runner also passes the original 4B matrix (25 calls / 75 tensors and
seven rejections) with the preserved 4B packages.

```bash
# Diagnostic probe: inspect each result's exported field, not just process exit.
work/qwen35/venv/bin/python -m compiler.qwen35.diagnostics.probe_dynamic
/data/workspace/abo/.venv/bin/cmake --build work/qwen35/build --target qwen35_block -j2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 work/qwen35/venv/bin/python \
  -m compiler.qwen35.tests.validate_block --sequence-tests --binary work/qwen35/build/qwen35_block
```

Evidence is under `artifacts/qwen35/0.8b/dynamic-prefill/` (including complete
constraint errors), `artifacts/qwen35/0.8b/sequence-validation/` and
`artifacts/qwen35/4b-sequence-regression/`. Commands require the step-1 setup and
packages below. Use fresh `--output` directories for reruns. Step-1 packages and
results remain unchanged; its runner/source snapshot is in
`work/qwen35/0.8b/step1-source/`.

## Step 1 result: 0.8B single block

The probe uses the original layer-0 DeltaNet weights and HF forward, with a small
`TensorState` adapter. The independent eager reference uses HF `DynamicCache`.
The block includes projections, depthwise convolution, the gated delta rule,
gated normalization and output projection; surrounding MLP/residuals and all
other layers are outside this test.

```text
step(hidden, conv, recurrent) -> output, next_conv, next_recurrent
```

| Tensor | Shape | Dtype |
| --- | --- | --- |
| hidden / output | `[1, T, 1024]` | BF16 |
| conv / next_conv | `[1, 6144, 4]` | BF16 |
| recurrent / next_recurrent | `[1, 16, 128, 128]` | FP32 |

Prefill uses fixed T=64; decode uses T=1. Inputs are contiguous CUDA tensors.
Inputs are preserved and returned state tensors become the next call's inputs.
Zero state begins each request. The adapter clones convolution storage to isolate
HF's in-place update. This functional baseline permits allocations/copies.

- Pinned [checkpoint](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17):
  `2fc06364715b967f1860aea9cf38778875588b17`. The downloaded weight shard's SHA256
  matched the Hub record; the extracted block and config have recorded checksums.
- Torch `2.7.0+cu126`, Triton `3.3.0`, Transformers `5.20.2`, NVIDIA L20 SM89;
  same isolated environment and native toolchain as the original 4B probe.
- Strict export succeeds without graph breaks or graph-signature mutations:
  decode 127 nodes, prefill 994 nodes. Both `.pt2` files contain compiled AOTI
  code. No HF arithmetic changes, custom kernels or numerical overrides beyond
  `emulate_precision_casts=True` and disabled TF32 were needed.

| Entry point | Export time | Compilation time | Package size |
| --- | ---: | ---: | ---: |
| Decode | 0.51 s | 14.28 s | 20.67 MiB |
| Prefill | 1.68 s | 30.32 s | 22.22 MiB |

These are observed build times on a shared host, not inference benchmarks.
Four cases cover 25 calls / 75 output-and-state tensors: prefill plus eight
decode steps, four decode steps from zero state, consecutive prefills followed
by decode, and a repeated request. Inputs are seeded synthetic hidden states.

| Check | Result |
| --- | --- |
| Adapter vs HF DynamicCache | All 75 tensors bitwise equal |
| C++ vs Python AOTI | All 75 tensors bitwise equal |
| AOTI vs HF BF16, unchanged 5% relative limits | Passed |
| State reset/repeat, input preservation, shapes/dtypes, finite values | Passed |
| Python-free native loading (`libpython` and `libtorch_python`) | Passed |
| Native rejection of mismatched hidden/state shapes, wrong dtype/stride | All four passed |

Worst observed numerical errors, using the same per-token/per-channel/per-head
metrics described in the historical result below:

| Tensor | AOTI vs HF relative L2 | AOTI vs HF peak-relative | AOTI vs FP32 relative L2 | HF vs FP32 relative L2 |
| --- | ---: | ---: | ---: | ---: |
| Output | 0.5736% | 1.4881% | 1.0391% | 1.0801% |
| Convolution state | 0% | 0% | 0.3847% | 0.3847% |
| Recurrent state | 0.0000903% | 0.0000735% | 1.1241% | 1.1241% |

FP32 is a diagnostic control using checkpoint values loaded into FP32 and the
same BF16 input fixtures promoted to FP32. Its metrics are retained for both HF
and AOTI; eager parity is not a claim of improved FP32 accuracy.

Evidence: `artifacts/qwen35/0.8b/block/` and `artifacts/qwen35/0.8b/validation/`.
The latter contains `summary.json`, native traces/linkage and rejection logs.
`work/qwen35/0.8b/` holds the extracted weights, compilation cache and logs.
The 4B compatibility run in `artifacts/qwen35/4b-config-regression/` also passes
25 calls / 75 tensors and all four rejections, using the preserved 4B packages.
No full-model generation or inference-performance result is claimed here.

### Reproduce 0.8B

Run from `/data/workspace/develop/qwen3-aoti-poc`. The isolated environment uses
the existing Torch installation read-only through a `.pth` file; the original
`.venv` and ABO packages are not upgraded. Skip environment/build configuration
if already set up. Optional `fla` and `causal_conv1d` packages must be absent so
eager and compiled runs use the documented PyTorch baseline.

```bash
.venv/bin/python -m venv work/qwen35/venv
work/qwen35/venv/bin/python - <<'PY'
from pathlib import Path
site = Path('work/qwen35/venv/lib/python3.12/site-packages')
(site / 'torch-base.pth').write_text('/data/workspace/abo/.venv/lib/python3.12/site-packages\n')
PY
work/qwen35/venv/bin/python -m pip install -r compiler/qwen35/requirements.txt
work/qwen35/venv/bin/python -m pip check

work/qwen35/venv/bin/python -m compiler.qwen35.diagnostics.block
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/work/qwen35/0.8b/inductor"
work/qwen35/venv/bin/python -m compiler.qwen35.diagnostics.export_block decode
work/qwen35/venv/bin/python -m compiler.qwen35.diagnostics.export_block prefill

/data/workspace/abo/.venv/bin/cmake -S . -B work/qwen35/build -G Ninja \
  -DCMAKE_PREFIX_PATH=/data/workspace/abo/.venv/lib/python3.12/site-packages/torch/share/cmake \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DTORCH_CUDA_ARCH_LIST=8.9 -DCMAKE_BUILD_TYPE=Release -DQWEN_BUILD_DELTANET=ON
/data/workspace/abo/.venv/bin/cmake --build work/qwen35/build --target qwen35_block -j2
work/qwen35/venv/bin/python -m compiler.qwen35.tests.validate_block \
  --binary work/qwen35/build/qwen35_block
```

Preparation/export/validation refuse to overwrite their outputs. For another
run, choose fresh `--output` paths and use the matching `--model-dir` and
`--packages` arguments. `diagnostics/block.py --model Qwen/Qwen3.5-4B` selects the pinned 4B
weights and defaults extraction to `work/qwen35/4b/block`; pass that path explicitly
to export/validation and select separate package/result directories.

## Unified DeltaNet preparation and recurrence: 2026-10-29

**Source cleanup:** The standalone `recurrent_kernel`, its operator wrapper and
Python/C++ test runners have been removed. The fused validator now uses HF
convolution, gates and recurrence plus an independent FP32 control. The model
validator keeps its existing HF comparisons and removes the standalone-core
control. Numerical limits and native parity requirements are unchanged.
Cleanup revalidation recompiles four focused AOTI packages and passes all
396 native calls / 1,188 bitwise tensor checks. `fusion_passed=True`; the fused
and HF controls each retain 125 failures against full FP32, so `passed=False`
and exit 1 remain. The retained fused functions are AST-identical, and strict
prefill/decode exports produce identical graph code to the initial unified pair.
The retired operator and CMake target are absent. Evidence is in
`artifacts/qwen35/0.8b/remove-recurrent-validation/summary.json` and
`artifacts/qwen35/0.8b/remove-recurrent-audit.json`.

The results below were recorded before this cleanup and retain their original
comparison names and counts; they are not fresh results for the revised tests.

Both production entry points now use `xinfer::delta_fused`, exposed as
`fused(qkv, b, a, conv, state, weight, A_log, dt_bias)`. The kernel consumes
projected QKV/A/B, fuses width-four causal convolution/history handling, SiLU,
beta/log-decay preparation and recurrence, and returns fresh core output,
convolution state and FP32 recurrent state. T=1 uses the decode package;
T=2..512 uses one dynamic prefill package. QKV/B/A/Z projections, gated
normalization and output projection remain outside. The initial validation
retained a recurrent-core-only operator for reference checks; that control has
since been removed. Chunked prefill remains a TODO.

Strict full-model export and AOTI compilation pass with default compiler
settings (`emulate_precision_casts=True`, optional combo optimizations off):

| Entry point | Token contract | Graph nodes | `delta_fused` calls | Generated matrix calls | External convolution / proxy-executor calls |
| --- | --- | ---: | ---: | ---: | --- |
| Prefill | Dynamic T=2..512 | 2,166 | 18 | 97 | 0 / 0 |
| Decode | T=1 | 2,165 | 18 | 97 | 0 / 0 |

Packing remains 48 groups / 187->97 linear calls in both graphs. Packages are
in `artifacts/qwen35/0.8b/text-unified-fused`. No separate custom-op library or
Python runtime is needed by the C++ executable.

### Code-generation issue and source workaround

The earlier two-dimensional tap helper exposed a missing BF16 V conversion in
generated code under the runtime token loop on L20 / SM89 with Torch
2.7.0+cu126 and Triton 3.3.0. LLVM IR still requested BF16-to-FP32 widening,
but PTX loaded V's raw BF16 bits into FP32 registers. A zero-V control reproduced
the bad result to FP32 roundoff. The current helper uses one-dimensional static
four-tap accumulation with explicit
masked history/input loads. It preserves the convolution-result cast before
SiLU and the activation/beta casts before recurrence; no precision setting or
numerical tolerance was relaxed. The retained probe and generated-code evidence
are under `artifacts/qwen35/0.8b/unified-fused-codegen`.

### Focused operator validation

Eighty-one BF16/FP32 cases cover convolution-history boundaries T=1,2,3,4,5,
lengths through 512, zero/nonzero state, mixed prefill/decode carry, whole versus
partitioned input, repeated requests, gate extremes, cancellation, checkpoint
projections, strided inputs, streams and contract rejection. Python JIT,
Python-loaded AOTI and standalone C++ agree bitwise across 396 calls / 1,188
tensors. Fused-versus-unfused and HF comparisons satisfy the declared
precision-specific numerical limits.

Both carried states remain bitwise across partitions, as do BF16 core outputs.
The FP32 core partition comparison records a maximum absolute difference of
1.86e-9 across static/decode and dynamic/prefill specializations. It uses the
existing FP32 numerical limits; its bitwise flag and error metrics remain in
the report. All same-input JIT/AOTI/C++ and repeated-request checks remain exact.

`fusion_passed=True`, while full-FP32 acceptance remains open. Fused and unfused
BF16 paths each have 125 comparisons against full FP32 above the existing
limits, with identical BF16 error metrics. All 250 entries remain visible,
overall `passed=False` and exit 1 are retained. This distinguishes fusion
correctness from the BF16-to-FP32 gap without hiding either result.

### Full-model and native validation

The 68-step model suite's `fusion_vs_unfused` comparison is bitwise for all
3,332 tensors. Shape/dtype/finite, masks, future-cache/write regions, unchanged
inputs and reset/repeat checks pass, including dynamic lengths 2,22,65,511,512.
The broader numerical report retains 5,241 failure keys versus the earlier
decode-only run's 5,248: zero added and seven removed. Existing HF/compiler
numerical acceptance remains open.

Full-model native validation passes 27 cases: 4,018 teacher tensors, 30 greedy
logit rows and 288 final-state tensors match Python AOTI bitwise. All 23
rejection checks, CLI parity, reset/repeat and dispatch checks pass. The native
process maps no Python library. Compiled generation differs from the earlier
package pair: the capital answer adds token 13 (a period), producing `Paris.`
instead of `Paris`. Arithmetic and Chinese chat outputs remain unchanged.
Exact eager fusion and native/AOTI parity do not close the numerical issues or
establish identical outputs across separately compiled package versions.

Compact reports are retained at
`artifacts/qwen35/0.8b/unified-fused-validation/summary.json`,
`artifacts/qwen35/0.8b/unified-fused-model/summary.json` and
`artifacts/qwen35/0.8b/unified-fused-runtime/summary.json`. Raw fixtures remain
temporarily under `/dev/shm/xinfer-unified-fused-20261029/` because development
disk space is limited; that temporary storage can disappear after restart.
The shared GPU is busy, so no latency or speedup is claimed. Performance must
be measured on the test host with matching compiler settings and source.

## Fused DeltaNet decode preparation: 2026-10-29

**Historical scope:** This section records the earlier T=1-only fusion.
The current source uses `xinfer::delta_fused` for T=1..512;
the results below do not establish acceptance of that unified implementation.

Decode now uses `xinfer::delta_decode`, a `torch.library.triton_op` whose single
Triton launch fuses width-four causal convolution/history shift, SiLU,
beta/log-decay preparation and the recurrent core. QKV/B/A/Z projections and
z-gated normalization/output projection remain outside. The model adapter selects
this path only for T=1; dynamic prefill keeps native convolution and
`xinfer::delta_recurrent`. The two kernels share `_recurrent_step`.

The kernel is deliberately narrow: batch one, 16 heads, K=V=128, bias-free
width-four SiLU convolution, BF16 or FP32 inputs/weights and FP32 recurrent state.
Convolution accumulates in FP32 and rounds to the input dtype before SiLU;
activation and beta round again before recurrence. The installed HF reference
uses grouped `F.conv1d`, so there is no BF16 rounding of individual products.
Four value tiles share Q/K preparation; only tile zero writes Q/K convolution
history, while V tiles write disjoint channels. Core output, next convolution
state and next recurrent state are fresh tensors; input state is unchanged.

Focused validation retains the old HF convolution/gate preparation followed by
the original recurrent operator, HF recurrence and an independent full-FP32
control. Nineteen cases cover random zero/nonzero state, 64 carried steps and
reset/repeat, gate extremes, convolution cancellation, actual checkpoint
projections, noncontiguous inputs, non-default streams, and rejection checks.
All 158 BF16 steps match the unfused control bitwise for core and both states.
FP32 versus unfused maximum errors are 5.59e-9 for core and 1.79e-7 for recurrent
state; maximum normalized error is 4.57e-6, below the existing 1e-4 FP32 limit.
All convolution-state updates are exact. Two tiny AOTI packages execute in a
Python-free C++ process: 300 calls / 900 tensors match Python AOTI bitwise,
without a separate custom-op library.

The full-FP32 gate is deliberately reported separately. BF16 convolution and
gate rounding produce 77 comparisons above the existing 1% BF16 limits, both
for fused and unfused paths. Their per-row FP32 errors are exactly equal.
`fusion_passed=True`, `fp32_acceptance=False`, and
`bf16_fp32_errors_match_unfused=True`; all 154 entries remain visible, overall
`passed=False` and exit 1 are retained. No tolerance was relaxed to accept fusion.
See `artifacts/qwen35/0.8b/fused-decode-validation-v2/summary.json`.

The unchanged recurrent regression passes all 72 cases with no numerical
failures, including dynamic lengths and partitioned carry; standalone C++
executes 204 calls / 408 tensors. A direct probe against the saved original
kernel is bitwise for BF16 and FP32 at T=1,2,32,65,512. Evidence is
`fused-decode-recurrent-regression/summary.json` and
`fused-decode-original-kernel.json` under `artifacts/qwen35/0.8b/`.

Full-model strict export and AOTI compilation pass with the default compiler
settings (`emulate_precision_casts=True`, optional Inductor optimizations off).
Packing remains 48 groups / 187->97 linear calls in both entry points. Prefill
has 2,672 nodes with T=2..512, and its exported code is identical to the prior
packed graph. Decode decreases from 2,705 to 2,165 nodes with 18 fused operators.
Generated wrappers each retain 97 matrix calls; decode has zero external
convolution calls versus 18 in prefill. Both have zero proxy-executor calls.
New packages are in `artifacts/qwen35/0.8b/text-fused-decode`.

The 68-step full-model suite adds an eager `decode_fusion_vs_unfused` control;
all 3,332 tensor comparisons are bitwise equal. Shape/dtype/finite, masks,
future-cache/write regions, input preservation and reset/repeat checks pass.
Existing full-model HF/packing/compiler gates remain open: 5,248 failures,
with exactly the same failure keys as the prior packed run (zero added/removed).
See `fused-decode-model/summary.json`.

Full-model native runtime validation passes 27 cases: 4,018 teacher tensors,
28 greedy logit rows and 288 final-state tensors match Python AOTI bitwise.
All 23 rejection checks pass, including context and package contracts. Greedy
IDs match the prior packed native run for all common cases. This does not mean
the compiled packages are numerically identical: 25/27 raw state files differ,
and arithmetic/capital logit traces differ (maximum absolute logit differences
0.171875/0.328125; global relative L2 about 0.00479/0.01308). Chinese logits
remain bitwise. Existing AOTI/HF teacher top-token differences remain present.
See `fused-decode-runtime/summary.json` and `fused-decode-audit.json`.

The development GPU remains shared/busy. These are functional and numerical
checks, not evidence of a speedup. The next timing comparison should recompile
on the test host and compare the old combo-enabled control with fused decode
plus the same combo settings; launch/fusion gains must be remeasured together.
The optional `--enable-inductor-optimization` flag still only controls the
existing Inductor combo settings. Chunked prefill remains a TODO.

## Frozen linear weight packing: 2026-10-28

Implemented `compiler/common/passes/pack_linears.py` as a model-independent pass over
`ExportedProgram`, before Inductor decomposition. It groups bias-free
`aten.linear` calls sharing the exact input node and compatible frozen 2D
weights. It skips trainable/shared/tied weights, tensor subclasses, buffers,
and weights with other consumers. Real weights are concatenated once during
export into lifted frozen parameters; replaced parameters/placeholders are
removed. The original output node names, user/state ABI and range constraints
are preserved. Split outputs restore contiguous layout for existing views.

`compiler.qwen35.export` always applies the pass to both entry points and
writes to `artifacts/qwen35/0.8b/text` by default, with no packing switch.
The initial validation below used explicitly selected `text-packed` packages;
those artifact paths remain unchanged. Metadata records versioned logical
packing groups; separate
`.packing.json` files include FX names for profiling. C++ and Python reject
mixed packing settings. `export_step` shares graph creation with the eager
packing-only numerical control; no new model adapters are needed.

Both strict dynamic export and AOTI compilation pass on the development L20,
Torch 2.7.0+cu126 / Triton 3.3.0. There are 48 groups: 18 DeltaNet input groups,
24 MLP gate/up groups and six attention QKV groups. Linear calls decrease from
187 to 97 in both graphs. Prefill has 2,672 nodes with T=2..512; decode has
2,705 nodes at T=1. The extra split/layout nodes increase graph size despite
reducing projection calls.

The generated wrappers each contain exactly 97 `aoti_torch_cuda_mm_out` calls
and 48 packed constants. None of the 138 replaced source weight constants
remain. Their packed bytes are exactly 718,405,632, equal to the source weight
bytes. Packing is absent from inference: weights are already materialized
constants, not runtime concatenations. Package sizes are 1,506,707,734 bytes
(prefill) and 1,506,617,488 bytes (decode), close to the unpacked pair rather
than carrying another copy of the original weights. Evidence is
`artifacts/qwen35/0.8b/packing-package-audit.json`.

Six CPU tests cover dynamic shapes, uneven output widths, interface/state
preservation, parameter pruning, contiguous view consumers, BF16 output rounding,
independent input groups, serialization and idempotence. Exclusion subcases cover
single projections, bias, trainable weights, different inputs/dtypes, repeated
weights, distinct parameters sharing storage and other weight consumers. All pass.

Native-runtime validation passes 27 cases: 4,018 teacher tensors, 28 greedy
logit rows and 288 final-state tensors match Python AOTI bitwise. All 23
rejection checks pass, including a mismatched-packing package pair. The broader
suite matches all 249 teacher-forced steps / 12,201 tensors, plus greedy token
IDs/logits/final states, across C++ and Python AOTI. Input preservation, masks,
write regions, context overflow and reset/repeat checks pass.

Packing is not numerically identical to separate BF16 projections. The 68-step
model suite now includes an eager packed graph to distinguish that change from
Inductor effects, with unchanged 5% limits:

| Comparison | Maximum logits relative L2 / peak-relative | Failed tensor comparisons |
| --- | --- | --- |
| Packing eager vs unpacked adapter | 1.492277% / 1.313406% | 1,036 conv + 2 recurrent |
| AOTI vs packing eager | 1.427014% / 1.127820% | 1,011 conv + 1 recurrent |
| AOTI vs HF BF16 | 1.543383% / 1.086957% | 1,121 conv |

The eager packed graph has the same top token as the unpacked adapter at all
68 steps, despite the tensor differences. AOTI versus HF differs at
`dynamic_31` step 2 (HF tie) and `dynamic_32` step 1 (HF margin 0.125).
Do not attribute every difference to the previously recorded gated-norm issue:
changed GEMM shapes already alter numerical results before compilation.

Expanded HF acceptance remains open: 4,099 conv and 17 recurrent comparisons
exceed 5%. Maximum logits relative L2 / peak-relative are 3.924062% / 1.979167%
against HF, versus 2.684237% / 1.458333% in the earlier unpacked run with these
fixtures/toolchain. On the FP32 subset, maximum logit relative L2 is 3.177323%
for packed AOTI and 2.346991% for HF BF16. Arithmetic and Chinese greedy output
match HF; the capital answer is ` Paris\n` rather than ` Paris.\n`, where HF
has a tie between period and end-of-turn. Teacher-forced differences also
include `length_32` step 1 with a nonzero HF margin, so not all differences
are ties. Both numerical validators exit 1 and retain their full reports.

The HF/native benchmark smoke passes lengths 1, 22 and 512 with two decode
inputs, one warmup and one measured repeat; timings are provisional on the busy
GPU. No performance improvement is claimed. Packing is now the default at the
user's request; numerical acceptance and idle A/B measurements remain open.
The tuning guide uses the earlier unpacked control checkout for its original
compiler-option experiment. Keep packing fixed across tuning arms, then compare
packed and unpacked checkouts with the selected configuration held fixed.

Canonical evidence under `artifacts/qwen35/0.8b/`:
`packing-runtime/summary.json`, `packing-model/summary.json`,
`packing-end-to-end/summary.json`, `packing-benchmark-smoke/summary.json` and
`packing-package-audit.json`. Build and export logs are in `work/packing-*`.
The existing unpacked package pair is retained as the control. The rebuilt CLI
reproduces its earlier capital token IDs and full logit trace bitwise; evidence
is `packing-unpacked-compatibility/summary.json` under the same artifact root.

## Dynamic-only Triton operators: 2026-10-28

Dynamic prefill T=2..512 and decode T=1 are now the only full-model execution
contract. Both entry points always install `adapters/delta.py`, and the exporter
needs no dynamic flag, backend selection, native library or kernel-build path.
The runtime executes each multi-token input in one prefill call. Fixed 64/1
dispatch and old-package compatibility have been removed. HF eager references
and the benchmark now use full-prompt calls; the benchmark compares HF and C++
AOTI without the obsolete HF chunked profile.

`kernels/recurrent.py` registers `xinfer::delta_recurrent` with
`torch.library.triton_op` and launches the kernel via `wrap_triton`. Strict
export keeps the high-level operator node; AOTInductor decomposes it, compiles
the kernel, and bundles its cubin and generated native launcher inside `.pt2`.
The C++ process loads the model package without registering a custom operator.
The package still contains AOTI's generated model library; there is no separately
built or maintained custom-op `.so`.

Removed the ATen chunk-rule implementation, hand-written CUDA-driver launcher,
external cubin/header builder and their diagnostic-only native-op export/test
drivers. The older isolated HF block probe remains a diagnostic. Previous
uncommitted native sources are retained in `work/before-triton-op-source.tar`.
Older generated artifacts remain available as historical evidence, but the new
runtime requires packages with `delta_backend=triton_op`.

The Torch 2.7 probe established two details before the full-model change:

1. A convolution-like transpose/split/reshape producer gets contiguous kernel
   buffers `[2048*T,2048,128,1]` without private layout-registration hooks.
2. Inductor reconstructs user-kernel metadata. The manual-AOT implementation's
   no-specialization decorators and `enable_fp_fusion=False` launch option did
   not survive that reconstruction, causing small eager/AOTI roundoff differences.
   Plain `@triton.jit` and ordinary warp/stage options give bitwise eager/AOTI
   agreement for BF16/FP32 and fixed T1/dynamic T2..512. This uses the compiler's
   default contraction behavior; it does not claim that FP contraction is disabled.

The isolated operator suite passes all 72 cases / 204 calls, including 15
lengths, zero/nonzero state, whole-vs-partitioned state carry, repeats,
noncontiguous inputs and a nondefault CUDA stream. It retains the previous
1% BF16 / 0.01% FP32 normalized limits and the 1e-7 FP32 absolute floor.
There are no numerical failures. Standalone C++ loads four core packages,
runs 204 calls and matches all 408 output/state tensors bitwise. Process-map
checks reject Python and custom-op libraries. No external kernel binary/header
or library build is needed. Evidence: `artifacts/qwen35/0.8b/triton-op-core/summary.json`;
the smaller API/layout probe is under `work/triton-op-probe/`.

Both full-model entry points compile using the standard commands without
backend/library/dynamic flags. The default directory is
`artifacts/qwen35/0.8b/text`; its previous fixed packages were moved to
`text-fixed-retired`. No custom-op library exists beside the new packages.

| Entry point | Graph nodes | Triton operators | Token range | Compiled package |
| --- | ---: | ---: | --- | ---: |
| Prefill | 2,528 | 18 | 2..512, one symbol | 1,506,979,675 bytes |
| Decode | 2,561 | 18 | 1 | 1,506,860,654 bytes |

The archives contain 32 / 31 cubins respectively, including the user kernel.
Their generated C++ has no proxy-executor custom-op calls. There is no separate
kernel-build directory, embedded header or custom-op library in this deployment.
The standard C++ executables were rebuilt under `work/qwen35/build`.

Validation encountered a host storage problem: the first final export failed
linking because `/tmp` was full. Export succeeded after setting `TMPDIR` and
`TORCHINDUCTOR_CACHE_DIR` to project-owned directories on `/data`. Torch 2.7's
C++ package loader nevertheless hard-codes `/tmp/XXXXXX` for extraction, ignores
`TMPDIR`, and did not report the truncated extraction before a SIGBUS. The failed
extraction and this run's failed linker cache were moved to
`work/triton-op-disk-failure`, freeing about 4 GB on the system filesystem.
The original linker log is `work/triton-op-first-link-failure.log`. Packages
were intact; validation was restarted after freeing extraction space.

The full-model native runtime suite passes 27 cases, comparing 4,018
teacher-forced tensors, 30 greedy logit rows and 288 final-state tensors bitwise
against Python AOTI. Dynamic call counts, context overflow, CLI output,
reset/repeat and input preservation pass, along with all 22 input/package
rejection checks. The native process rejects mapped Python and custom-op
libraries. The canonical report is
`artifacts/qwen35/0.8b/triton-op-runtime/summary.json`.

The 68-step model suite passes all state-semantics checks, including poisoned
future KV, write regions, repeat/reset, finite shapes/dtypes and input
preservation. Numerical acceptance remains open under the original 5% limits:

| Comparison | Maximum logits relative L2 / peak-relative | Failed tensor comparisons |
| --- | --- | --- |
| Eager Triton adapter vs HF BF16 | 1.439449% / 1.398601% | 956 conv |
| AOTI vs HF BF16 | 1.754282% / 1.143791% | 1,034 conv |
| AOTI vs eager adapter | 1.612021% / 1.071429% | 1,012 conv |

Logits, recurrent state and KV comparisons pass the limits in this suite.
AOTI/HF top-token differences occur at `dynamic_31` step 2 and `dynamic_33`
step 0, both HF BF16 ties. Eager Triton also differs at `dynamic_32` step 1
with HF margin 0.125, so adapter drift still exists before compilation.
Maximum logit relative L2 against FP32 is 1.635470% for AOTI versus 1.304223%
for HF BF16. The complete report is
`artifacts/qwen35/0.8b/triton-op-model/summary.json`; its exit status is 1.

The broader end-to-end suite also passes native parity: all 249 teacher-forced
steps / 12,201 tensors, greedy token IDs, logit traces and final states match
C++ versus Python AOTI bitwise. All three greedy chat outputs match HF,
including ` Paris.\n`, `42\n` and the Chinese compiler explanation. The
previous manually packaged baseline's missing-period case no longer occurs.
Two teacher-forced top-token choices differ from HF at ties: `length_31`
step 2 and `length_33` step 0.

HF numerical acceptance remains a failure: 3,836 conv and nine recurrent
tensor comparisons exceed 5%. Maximum logits relative L2 / peak-relative
against HF BF16 are 2.684237% / 1.458333%. On the independent FP32 subset,
maximum logits relative L2 is 2.139068% for AOTI and 2.346991% for HF BF16.
Both BF16 paths differ from FP32 at `capital_teacher` step 1 (FP32 margin
0.0928917). These controls do not override the failed state thresholds.
The report is `artifacts/qwen35/0.8b/triton-op-end-to-end/summary.json` and exits 1.

The simplified benchmark harness passes HF and C++ AOTI profiles for prompt
lengths 1, 22 and 512 with two fixed decode tokens, one warmup, one repeat
and one smoke-test round. It preserves numerical-acceptance failures in its
report and marks shared-GPU timings provisional. The normal default is two
rounds, alternating which of the two profiles runs first. Evidence is
`artifacts/qwen35/0.8b/triton-op-benchmark-smoke/summary.json`.

No acceptance threshold was relaxed. Export, core, native-runtime and benchmark
functional checks pass; full-model numerical acceptance and idle performance
measurements remain open. Chunked prefill remains a TODO.

## Full-model recurrent Triton integration: 2026-10-28

New dynamic exports use `xinfer::delta_recurrent` in all 18 DeltaNet layers
for both prefill and decode. The adapter keeps the original projections,
convolution, gated normalization and output projection. Decode clones its
convolution state before calling HF's mutating update helper, preserving the
functional package ABI. Full-attention layers, the 48 state tensors and shared
generation/benchmark loops are unchanged.

Strict export and AOTI compilation on L20 / SM89 succeed:

| Entry point | Export | Recurrent custom ops | Compiled package |
| --- | --- | ---: | ---: |
| Prefill | 2,582 nodes, `{s0: VR[2,512]}` | 18 | 1,506,934,310 bytes |
| Decode | 2,561 nodes, T=1 | 18 | 1,506,835,066 bytes |

Neither exported graph contains the old `xinfer::delta_prefill` operator.
The cubins are embedded in `libxinfer_recurrent_ops.so`; export copies it
beside the packages under the established `libxinfer_delta_ops.so` sidecar
name. Metadata records `delta_backend=triton_recurrent`, the library checksum,
and the kernel build, including source/header/cubin checksums and SM. The native
loader rejects mixed backends/builds. Old packages without a backend field
retain the ATen chunk interpretation.

The first full-model run exposed an Inductor layout issue. Although the
adapter called `.contiguous()`, Torch 2.7 relaid out Q/K/V to strides
`[2048*T,1,128*T,T]`. The native op's contiguous guard rejected these tensors
before launching the cubin. `load_library` now declares an explicit
`require_contiguous` layout constraint for the recurrent op. Recompiled
wrappers pass `[2048*T,2048,128,1]` buffers. The kernel and its hard input
checks remain unchanged. This is an operator layout contract, not a graph
rewrite or a runtime copy workaround.

The native-runtime validator passes 27 cases: 4,018 teacher-forced tensors,
28 greedy logit rows and 288 final-state tensors agree bitwise with Python
AOTI. It verifies dynamic call counts, context-end state carry, input
preservation, repeat/reset behavior, CLI parity, and 26 rejection cases.
The fresh C++ process maps neither `libpython` nor `libtorch_python`.

The 68-step model validator keeps its original 5% row-relative policy:

| Comparison | Maximum logits relative L2 / peak-relative | Failed tensor comparisons |
| --- | --- | --- |
| Eager Triton adapter vs HF BF16 | 1.662634% / 1.048951% | 959 conv |
| AOTI vs HF BF16 | 1.720276% / 1.071429% | 1,032 conv |
| AOTI vs eager Triton adapter | 2.220087% / 1.223776% | 1,011 conv |

Logits, recurrent states and KV comparisons pass that policy in this suite.
All input-preservation, masked-future, KV-write-region and reset/repeat checks
pass, including mask lengths 2, 22, 65, 511 and 512. HF DynamicCache and
StaticCache controls remain bitwise equal.

Unlike the previous ATen adapter, the eager Triton adapter is not bitwise
equal to HF. Both eager Triton and AOTI choose a different top token for
`dynamic_32`, step 1, where HF's top-two margin is 0.125. AOTI also differs at
`dynamic_33`, step 0, where HF has a tie. The FP32 comparisons are retained:
maximum logit relative L2 is 1.659203% for AOTI and 1.304223% for HF BF16.
These results do not close numerical acceptance. In particular, the new
adapter differences already exist before Inductor, so they cannot all be
attributed to the previously recorded lost BF16 rounding boundary.

The broader end-to-end suite checks 15 prompt lengths from 1 to 512,
64 decode steps after a 512-token prompt, the 1,024-token context boundary,
masked future KV, repeats and three chat prompts. All 249 teacher-forced
steps / 12,201 tensors match between C++ and Python AOTI bitwise; greedy
token IDs, logits and final states also match across those two runtimes.
The HF comparison remains a failure under the unchanged policy:

- 3,830 conv and eight recurrent tensor comparisons exceed 5%.
- Maximum logits relative L2 / peak-relative: 3.376350% / 1.875000%.
- Three teacher-forced top-token differences: `length_32` step 1 (HF margin
  0.125), `length_33` step 0 (tie), and `capital_teacher` step 1 (tie).
- Arithmetic and Chinese greedy output match HF exactly. The capital prompt
  produces ` Paris\n` instead of ` Paris.\n`: at the first difference, HF BF16
  assigns equal scores to period (13) and end-of-turn (248046). AOTI chooses
  end-of-turn; the independent FP32 teacher-forced control chooses the same
  token, with a top-two margin of 0.0928917.
- On the FP32 control subset, maximum logit relative L2 is 2.155917% for AOTI
  and 2.346991% for HF BF16. These controls do not override the acceptance
  failures or prove that every differing state is harmless.

Both numerical validators therefore exit 1; the runtime validator exits 0.
Canonical reports are `artifacts/qwen35/0.8b/recurrent-model-validation/summary.json`,
`recurrent-runtime-validation/summary.json` and `recurrent-end-to-end/summary.json`
under the same model artifact directory. The end-to-end report retains the
actual recurrent kernel/launcher sources, rather than the older chunk-rule source.

The C++ benchmark smoke passes prompt lengths 1, 22 and 512 with two decode
tokens, one warmup and one measured repeat; repeated predictions and consumed
positions agree. Timings are provisional on the busy GPU. The rebuilt CLI also
loads the previous `text-dynamic` ATen packages and reproduces the earlier
capital token IDs and full logit trace bitwise. Both checks are recorded in
`artifacts/qwen35/0.8b/recurrent-integration-smoke/summary.json`.

Initial artifacts and logs are retained under `work/recurrent-full-layout-initial`,
`work/recurrent-full-runtime-initial`, `work/recurrent-layout-failure.log`
and `work/recurrent-full-initial-sources.tar`. The corrected packages are in
`artifacts/qwen35/0.8b/text-recurrent`; the native build is `work/recurrent-build`.
Reproduction commands are in
[the model README](../../compiler/qwen35/README.md#dynamic-full-model-prefill-and-triton-gdn).

TODO: chunked prefill. This sequential recurrence establishes a working model
integration; performance remains to be measured on an idle GPU.

## Recurrent Triton baseline and native execution: 2026-10-28

Implemented an isolated GDN core in `compiler/qwen35/kernels/recurrent.py`.
It supports T=1..512, B=1, H=16, K=V=128, contiguous BF16/FP32 Q/K/V/beta,
FP32 log decay and FP32 recurrent state. One Triton program owns a head and
32 value columns; it keeps its state tile through the sequential token loop.
Q/K normalization, decay, correction and output reduction are fused. Inputs
are never mutated; both returned tensors have fresh storage. Projection,
convolution and gated normalization remain outside this experiment.

`diagnostics/build_recurrent.py` compiles two kernels with Triton 3.3.0,
four warps, one stage and FP contraction disabled. Their cubins are embedded
in a generated header. `runtime/qwen35/recurrent_ops.cpp` registers
`xinfer::delta_recurrent` and launches them through the CUDA driver on
LibTorch's current stream. The native library contains no Python callback or
Triton JIT. This minimal launcher accepts only CUDA device 0 and the exact
compiled SM; it deliberately pins the Triton 3.3 ABI, including the trailing
scratch argument. The build rejects kernels requiring scratch storage.

Validated on L20 / SM89, Torch 2.7.0+cu126 and Triton 3.3.0:

- 72 cases, 204 calls per execution path, with lengths
  1, 2, 3, 22, 31, 32, 33, 63, 64, 65, 127, 128, 129, 511 and 512.
- Unit-scale Gaussian Q/K/V, zero/nonzero state, small negative decay, and
  beta=0/decay=0, strong forgetting and zero-Q/K edge cases.
- Whole T512 versus `[129, 1 × 32, 351]` state carry; reset/repeat tests.
  Outputs and final states are bitwise invariant to partitioning in the
  recurrent implementation.
- Schema, fake-tensor and dynamic-AOT opcheck all pass. Strict export and
  AOTI compilation produce four small core packages: BF16/FP32 × fixed T1 /
  dynamic T2..512. Each graph has one recurrent custom op.
- Python Triton JIT, Python native-op and Python AOTI agree bitwise across
  all 408 output/state tensors.
- Standalone C++ passes directly and through AOTI: 204 calls and 408 bitwise
  tensor comparisons in each mode. Both processes verify that neither
  `libpython` nor `libtorch_python` is mapped. Direct mode loads no AOTI
  package; AOTI mode loads four packages once.
- Nondefault CUDA stream, input preservation, fresh output storage, finite
  results, dtype/shape contracts and six invalid-input rejection checks pass.
- All 2,112 recorded numerical/partition comparisons pass the policy below.

Maximum errors against HF recurrent (BF16 fixtures) and the FP32 control:

| Comparison | Output relative L2 | Output peak-relative | State relative L2 | State peak-relative |
| --- | ---: | ---: | ---: | ---: |
| BF16 kernel vs HF BF16 | 0.143764% | 0.492611% | 0.000219% | 0.000247% |
| BF16 kernel vs FP32 | 0.264709% | 0.386952% | 0.000219% | 0.000247% |

FP32 uses the same input values as the BF16 fixtures, converted to FP32, to
isolate arithmetic error from input quantization. Native ATen chunk-rule
comparisons also pass: maximum BF16 output relative L2 / peak-relative
0.132540% / 0.632911%; state 0.000149% / 0.000169%. On FP32 fixtures, maximum
absolute output/state error against HF recurrent is 5.22e-8 / 7.16e-7.
These are core-operator results; they do not close the full-model numerical
issues or establish an end-to-end speedup.

Two validation details are retained rather than hidden:

1. Initial Triton JIT calls inferred pointer alignment, while the AOT build
   supplied no alignment attributes. The resulting PTX used different thread
   layouts/reduction orders and produced small numerical differences. The
   kernel now disables JIT pointer-alignment and token-count specialization
   to match the AOT signature; all JIT/native comparisons are bitwise.
2. The initial relative-only FP32 limit rejected one strong-forgetting case
   against each of HF recurrent, native chunk and FP32. Nearly orthogonal
   Q/K produce tiny output rows: absolute errors were at most 3.36e-8, while
   peak-relative error reached 0.051724%. The final policy retains normalized
   limits of 1% for BF16 and 0.01% for FP32 and additionally permits FP32
   absolute errors <=1e-7. Every raw metric is still recorded. Exact-path
   comparisons always require bitwise agreement, regardless of tolerances.

Reproduction commands are in `compiler/qwen35/README.md`. Canonical evidence:
`artifacts/qwen35/0.8b/recurrent-validation/summary.json`, its fixtures,
four compiled packages and `cpp-direct` / `cpp-aoti` traces. The tested build
is `work/recurrent-build`; embedded kernel inputs are in
`work/qwen35/recurrent-kernels`. The initial failing report, kernel artifacts
and source snapshot are retained under `work/recurrent-validation-initial`,
`work/qwen35/recurrent-kernels-initial` and `work/recurrent-original-sources.tar`.
Generated artifacts remain ignored by Git.

TODO: implement chunked prefill and compare it with this sequential kernel
and the native ATen baseline in an idle benchmark environment. This isolated
baseline initially left the full-model backend unchanged; the subsequent
integration is recorded above. No timing claim is made on the busy development GPU.

## Dynamic full-model prefill and C++ generation: 2026-10-27

The native chunk-rule adapter now integrates with all 18 DeltaNet layers in
`TextStep`. `NativeDeltaNet` wraps each original module, reuses its parameters,
and writes returned states into the existing explicit `TextCache`. For T=1 it
delegates to the original HF decode implementation. Full-attention layers and
the 48-tensor state ABI are unchanged.

The exporter accepts `--dynamic-prefill --library PATH` for both entry points.
It copies the native library into the package directory and records its checksum
and size in both metadata files. The prefill contract is `tokens: null` with
`token_range: [2,512]`; decode remains T=1. The two dynamic input dimensions
(token IDs and cache positions) share one exported symbol. C++ loads the native
library before loading either AOTI package and retains it through loader teardown.
Python AOTI loads the same packaged library. Neither runtime pads prompt tokens.

Package metadata selects dispatch: dynamic packages consume each 2–512-token
input in one prefill call, while T=1 uses decode. Older fixed packages retain
their 64/1 behavior. Thus 22, 511 and 512 input tokens each require one prefill
call with these packages. Appended chunks use the current position and returned
states. Generation still uses the shared greedy loop, context limit and EOS rules.

On the source L20 / Torch 2.7.0+cu126 / Transformers 5.20.2 toolchain:

| Entry point | Exported graph | Native DeltaNet calls | AOTI package |
| --- | --- | ---: | ---: |
| Dynamic prefill | 2,528 nodes, `{s0: VR[2,512]}` | 18 | 1,506,851,414 bytes |
| Decode | 3,605 nodes, fixed T=1 | 0 | 1,506,844,698 bytes |

Both strict exports and AOTI compilation passed, with no input/buffer mutations.
Compilation uses `emulate_precision_casts=True`, TF32 disabled, and no
`--eager-numerics` override. The native operation uses existing ATen kernels;
no fused CUDA kernel or performance claim is introduced here.

### Validation results

The functional model check covers 68 steps / 3,332 tensors. The native adapter
in eager mode is bitwise equal to original HF on all of them. HF DynamicCache
and StaticCache also match on all 441 control tensors. Input preservation,
finite values, shape/dtype checks and reset/repeat pass. Future KV poisoning and
write-region checks pass at dynamic lengths 2, 22, 65, 511 and 512, including the
subsequent decode write.

The native runtime validator passes 27 cases: 4,018 teacher-forced tensors,
30 greedy logit rows and 288 final-state tensors all match Python AOTI bitwise.
Actual call counters confirm one prefill per multi-token input. The context-end
case consumes `[512,448,63,1]`, makes three prefill calls plus one decode call,
then rejects overflow without modifying state. EOS, one-token generation limits,
repeat/reset, the production CLI, 22 input/package rejection cases and absence
of Python libraries all pass.

The broader end-to-end check covers prompt lengths
`1,2,3,22,31,32,33,63,64,65,127,128,129,511,512`, continuation, appended chunks,
poisoned future KV, reset/repeat and context capacity. Its 249 teacher-forced
steps / 12,201 tensors match C++ and Python AOTI bitwise. HF and FP32 references
use matching full-prompt partitions for dynamic packages. All three independent
greedy chat cases agree with HF, including every emitted token:

- Capital: `Paris.`
- Arithmetic: `42`
- Chinese: `编译器是将源代码翻译成机器码的自动化过程。`

**Numerical acceptance remains open.** The unchanged 5% checks still fail:

| Check | Result |
| --- | --- |
| Model AOTI vs HF (68 steps) | 1,126 conv + 2 recurrent tensor comparisons fail |
| Model AOTI vs eager adapter | Same failures; adapter vs HF itself is bitwise |
| End-to-end AOTI vs HF (249 steps) | 4,182 conv + 15 recurrent tensor comparisons fail |
| Teacher-forced top-1 vs HF | Two differences: length 31 / step 2, length 33 / step 0 |
| Independent greedy chat sequences vs HF | All three agree |

Both top-1 differences occur at an HF top-two margin of zero; they remain
recorded as differences. Worst end-to-end logit relative L2 is 3.9121%,
peak-relative 2.2917%. FP32 controls are retained in both validation reports;
they do not replace the BF16 parity requirements. The large relative conv errors
use the same per-channel metric as earlier experiments. No tolerance was relaxed
and neither numerical validator reports a pass. Eager adapter agreement excludes
a porting mismatch for these fixtures, but does not establish the cause of every
remaining compiled error or close Q35-NUM-001/Q35-NUM-002.

### Reproduction and evidence

See [dynamic full-model commands](../../compiler/qwen35/README.md#dynamic-full-model-prefill-and-triton-gdn).
The following generated directories are ignored by Git:

- `artifacts/qwen35/0.8b/text-dynamic/`: compiled packages, graph dumps, metadata
  and `libxinfer_delta_ops.so`.
- `artifacts/qwen35/0.8b/dynamic-model-validation/`: HF/adapter/AOTI/FP32 metrics
  and mask/state checks; exit status 1 records numerical failures.
- `artifacts/qwen35/0.8b/dynamic-runtime-validation/`: passing native/CLI parity,
  call counts, state traces, linkage, EOS and rejection checks.
- `artifacts/qwen35/0.8b/dynamic-end-to-end/`: source snapshots, full-prompt
  numerical comparisons, greedy results and native traces; exit status 1.

The primary tests used `work/reorg-20261027/build/`. A fresh build at
`work/dynamic-text-build/`, with only the full-model option enabled, produced
identical library, runtime and validation binaries. The fresh configure required
an explicit `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` on this host.
The rebuilt runtime also passed the original fixed-package validation:
3,136 teacher tensors, 28 greedy logit rows, 288 final-state tensors and all
18 rejection checks. Its report is under
`artifacts/qwen35/0.8b/dynamic-fixed-compatibility/`.
Logs and orchestration exit codes are under `work/dynamic-*.log` and
`work/dynamic-text-validation-status.json`. The shared GPU remains unsuitable
for performance conclusions.

## Dynamic single-block native operator: 2026-10-27

The original HF fallback's modulo padding, reshapes and Python chunk scan
specialized export to the example token length. The new single-block adapter
keeps the outer HF operations visible and moves only the chunk rule into
`xinfer::delta_prefill`. It reuses the original parameters without changing HF
globals, registering an attention backend, or rewriting the exported graph.

The native implementation uses existing ATen CUDA operations and HF eager's
unit-lower-triangular solve. Its internal zero padding uses beta=0 and log-decay=0,
preserving recurrent state through padded positions. No fake tokens pass through
the outer convolution. Inputs remain unchanged, the recurrent state stays FP32,
and output is materialized in the query dtype before returning to the graph.
This is a correctness implementation; the operator still launches many ATen
kernels and is not a fused DeltaNet kernel.

Source entry points:

- `compiler/qwen35/adapters/native_delta.py`: original-weight prefill adapter,
  native library loading and symbolic fake implementation.
- `runtime/qwen35/delta_ops.cpp`: native chunk rule and dispatcher registration.
- `compiler/qwen35/diagnostics/export_dynamic_block.py`: strict dynamic export
  and AOTI package compilation.
- `runtime/qwen35/tests/dynamic_block.cpp`: one package loader, varying-length
  calls, explicit state, input checks, and Python-free process verification.
- `compiler/qwen35/tests/validate_dynamic_block.py`: operator checks, HF/FP32
  controls, stateful fixtures, partition comparisons and native bitwise checks.

On the source L20 with Torch 2.7.0+cu126, Transformers 5.20.2, Triton 3.3.0 and
C++11 ABI enabled, strict export succeeded with **69 nodes, one custom operator,
and range `{s0: VR[2, 512]}`**, traced at T=32. AOTI compilation succeeded with
`emulate_precision_casts=True` and TF32 disabled. Inspection of the generated
C++ wrapper confirms a native proxy-executor call to the registered operator.
`prefill.pt2` is the compiled package; `prefill.exported.pt2` is the separate
saved export graph. This introduces no runtime compilation.

The validator uses real layer-0 weights and seeded unit-scale Gaussian hidden
inputs. The 14 tested lengths are 2, 3, 22, 31, 32, 33, 63, 64, 65, 127, 128,
129, 511 and 512, each with zero and nonzero initial states. Additional cases
process the same 129 tokens as `[129]`, `[64,65]` and `[2,63,64]`, then repeat
the last case from its original state. There are 37 calls / 111 output and state
tensors. The synthetic states use Gaussian scale 0.1.

| Check | Result |
| --- | --- |
| Direct native op vs HF chunk rule, BF16 and FP32, T=2/65/129 | All 12 tensors bitwise equal |
| `opcheck`: schema, fake tensor, dynamic AOT dispatch | Passed |
| Adapter eager vs original HF block | All 111 tensors bitwise equal |
| C++ vs Python AOTI | All 111 tensors bitwise equal |
| AOTI vs HF, unchanged 5% limits | Passed |
| Whole vs split sequences, unchanged 5% limits | Passed for all four engines |
| Reset/repeat, input preservation, shape/dtype, finite values | Passed |
| C++ loads one package; no Python libraries mapped | Passed |
| C++ rejects lengths 1 and 513 before package invocation | Passed |

Worst errors across calls, using the existing per-token/channel/head metrics:

| Tensor | AOTI vs HF relative L2 | AOTI vs HF peak-relative | AOTI vs FP32 relative L2 | AOTI vs FP32 peak-relative |
| --- | ---: | ---: | ---: | ---: |
| Output | 0.2117% | 0.6211% | 1.2469% | 3.3774% |
| Convolution state | 0% | 0% | 0.3821% | 0.3888% |
| Recurrent state | 0.0002883% | 0.0004943% | 1.0461% | 1.7148% |

HF BF16 has nearly identical aggregate errors against FP32. Partitioning still
changes BF16 results: maximum relative L2 0.6370%, peak-relative 1.4151%; the FP32
control differs by at most 0.0001781% / 0.0002122%, respectively. These checks do
not close Q35-NUM-001 or Q35-NUM-002 and do not establish full-model acceptance.

Final evidence is in `artifacts/qwen35/0.8b/dynamic-block/` and
`artifacts/qwen35/0.8b/dynamic-block-validation/`. An earlier run used smaller
hidden amplitudes and is retained under `work/delta-native-preliminary/`.
Builds used `work/reorg-20261027/build/`;
logs are `work/delta-native-*.log`. All generated data remains ignored by Git.
The [compiler README](../../compiler/qwen35/README.md#dynamic-single-block-experiment)
contains portable reproduction commands with fresh default output directories.

This resolved the demonstrated export blocker for the isolated native-op block.
The subsequent full-model integration is recorded above. No speedup is claimed
on the busy development GPU.

## Completed 4B single-block probe

The following contract, commands and results describe **4B only**. This probe
uses the real weights of `model.language_model.layers.0.linear_attn`, including its projections,
depthwise convolution, gated delta rule, gated normalization and output
projection. It does not include the surrounding decoder MLP/residuals, other
layers, tokenizer, vision encoder or text generation.

The HF module and its PyTorch kernels are reused without modifying their
arithmetic. `TensorState` implements the small cache interface this block needs.
The independent reference uses HF's `DynamicCache`, not that adapter.

### Contract

```text
step(hidden, conv, recurrent) -> output, next_conv, next_recurrent
```

| Tensor | Shape | Dtype |
| --- | --- | --- |
| hidden / output | `[1, T, 2560]` | BF16 |
| conv / next_conv | `[1, 8192, 4]` | BF16 |
| recurrent / next_recurrent | `[1, 32, 128, 128]` | FP32 |

Prefill has fixed `T=64`; decode has fixed `T=1`. Inputs are contiguous CUDA
tensors. Zero state starts a request; returned states become the next call's
inputs. Inputs are not mutated. The adapter clones convolution storage because
HF's decode convolution updates that storage in place. This deliberately simple
functional interface permits allocations/copies; it is not a CUDA Graph path.

Weights are loaded as BF16, including conversion of the checkpoint's FP32
`A_log` and gated-norm parameters. The FP32 control loads the original checkpoint
values into FP32 parameters and uses the same BF16 input fixtures promoted to
FP32. It measures both weight/activation rounding and compute differences.

### Reproduce 4B

Use the environment and native build above. Explicit arguments reproduce the
original 4B paths; existing results must be preserved by choosing fresh output
directories for another run.

```bash
# Downloads only the shard containing layer 0, verifies its Hub SHA256, and
# extracts the block. Only the block is loaded onto the GPU.
work/qwen35/venv/bin/python -m compiler.qwen35.diagnostics.block \
  --model Qwen/Qwen3.5-4B --output work/qwen35/block

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/work/qwen35/inductor"
work/qwen35/venv/bin/python -m compiler.qwen35.diagnostics.export_block decode \
  --model-dir work/qwen35/block --output artifacts/qwen35/block
work/qwen35/venv/bin/python -m compiler.qwen35.diagnostics.export_block prefill \
  --model-dir work/qwen35/block --output artifacts/qwen35/block

work/qwen35/venv/bin/python -m compiler.qwen35.tests.validate_block \
  --model-dir work/qwen35/block --packages artifacts/qwen35/block \
  --output artifacts/qwen35/validation \
  --binary work/qwen35/build/qwen35_block
```

Export and validation refuse to overwrite their outputs. For another run, use
fresh `--output` directories and pass the corresponding `--packages` directory
to validation. `--model-dir` selects an extracted block directory.

The native runner can also be invoked directly using already prepared fixtures:

```bash
./work/qwen35/build/qwen35_block artifacts/qwen35/block \
  artifacts/qwen35/validation work/qwen35/native-rerun
```

The last directory must not exist. The native runner reads raw BF16 hidden
inputs, loads both AOTI packages once, resets state between cases and saves raw
output/state tensors. It does not invoke or link Python. The validation script
launches it with a minimal environment and checks both `ldd` and loaded mappings.

### Recorded result: 2026-10-23

- Model revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Torch `2.7.0+cu126`, Triton `3.3.0`, Transformers `5.20.2`, C++11 ABI enabled.
- NVIDIA L20, SM89, 14,336 MiB visible allocation; native CUDA toolkit 12.2.
- Strict export succeeded for both entry points, with no graph breaks and no
  graph-signature input/buffer mutations: decode 129 nodes, prefill 996 nodes.
- Both `.pt2` files are compiled AOTInductor packages, approximately 81/83 MiB
  for decode/prefill. They are not merely saved `ExportedProgram` objects.
- `emulate_precision_casts=True`; TF32 disabled. No custom kernels, mean
  overrides, or FP-contraction override were required.

Four cases cover 25 block calls: prefill + eight decode steps, four decode steps
from zero state, two consecutive prefills + decode, and a repeated first request.
Inputs are seeded synthetic hidden states, not captured full-model activations.

| Comparison | Result |
| --- | --- |
| State adapter vs HF DynamicCache | All 75 output/state tensors bitwise equal |
| Native C++ vs Python AOTI | All 75 tensors bitwise equal |
| Reset/repeated request | Bitwise equal |
| Input preservation, shapes, dtypes, finite values | Passed |
| Python-free native loading | Passed |
| AOTI vs HF BF16, 5% limits | Passed |

Worst observed errors across the 25 calls:

| Tensor | AOTI vs HF relative L2 | AOTI vs HF peak-relative | AOTI vs FP32 relative L2 | AOTI vs FP32 peak-relative |
| --- | ---: | ---: | ---: | ---: |
| Output | 0.7270% | 1.5152% | 1.0932% | 2.4579% |
| Convolution state | 0% | 0% | 0.3886% | 0.3886% |
| Recurrent state | 0.0001901% | 0.0002847% | 0.8315% | 1.3514% |

Relative L2 is `norm(actual-reference)/norm(reference)`; peak-relative is
`max(abs(actual-reference))/max(abs(reference))`. Denominators are clamped to
`1e-12`. Metrics are computed per token for output, per channel for convolution
history, and per head after flattening the recurrent 128x128 matrix. The maximum
over those groups is recorded. Both relative metrics must be <=5% for adapter/HF
and AOTI/HF comparisons. FP32 is a diagnostic control, not an eager-parity test.
All passing and failing cases retain their metrics; failures return a nonzero
exit status. Native/AOTI comparisons require exact bits.

The HF prefill function intentionally uses a triangular solver in eager and a
forward-substitution construction during export. Compiled arithmetic is therefore
not assumed to be bitwise equal to eager. No numerical acceptance failure was
observed in these fixtures. The surrounding full model remains unvalidated.

Evidence is in `artifacts/qwen35/block/` (packages, graph dumps and metadata) and
`artifacts/qwen35/validation/` (fixtures, native traces, linkage and summary).
Metadata records checkpoint and source hashes, parameter dtypes, package hashes,
versions, ABI, shapes, compile settings and observed compilation duration.
`work/qwen35/` contains the build, isolated environment, downloaded shard and
logs. Generated data is ignored by Git. The shared GPU was busy: no performance
claim is made from this correctness probe.

References: [pinned model](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a),
[pinned HF implementation](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/qwen3_5/modeling_qwen3_5.py).

## Full-model context scaling — 2026-11-06

Ordinary paired exports and standalone C++ execution completed for Qwen3.5-0.8B
and 4B on L20, using PyTorch 2.13.0+cu126, Triton 3.7.1, Transformers 5.20.2 and
Python 3.12.13. This record supplements the historical block experiment above;
it does not change its result or close the full-model numerical issues.

Schema 4 separates three bounds: an 8,192-token request prompt, 16,384 cache
positions, and 2–512 tokens per compiled prefill call. An 8K prompt uses sixteen
calls to the same prefill package, carrying KV, convolution and recurrent state
between calls. A singleton remainder uses decode. Generation begins after the
whole prompt. The larger state shapes require freshly compiled packages;
raising a JSON limit cannot enlarge an old artifact. Runtime and attention ABI
versions remain 1.

The numerical chain exercised HF, adapted eager, Python AOTI and C++ AOTI with
matched input partitions. It includes prompt boundaries 1, 511, 512, 513, 8,191
and 8,192; nonzero offsets; filled 16K contexts and unchanged-state overflow
rejection; poisoned future cache, reset/reuse and independent FP32 controls.
All model jobs ran sequentially. Native/Python parity and HF numerical
acceptance remain separate gates:

| Check | 0.8B | 4B |
| --- | ---: | ---: |
| Teacher cases / physical steps | 59 / 534 | 59 / 541 |
| Main-stage state checkpoints | 397 | 404 |
| Independent FP32 control steps | 271 | 271 |
| C++ vs Python AOTI bits and state contracts | Pass | Pass |
| HF numerical acceptance | **Fail, remains open** | **Fail, remains open** |
| Required comparison failures | 17,763 | 46,493 |
| Worst AOTI/HF logits: relative L2 / peak-relative | 3.1448% / 1.4938% | 41.49% / 52.26% |

For 0.8B, required failures comprise 17,724 convolution-state comparisons, 34
recurrent-state comparisons, four top-token comparisons and one greedy
sequence. No logit-error or KV threshold fails. For 4B, 209 required logit
comparisons fail across the chain; the remaining failures are state comparisons.
The 4B long-context AOTI/HF logit maxima alone are 10.62% relative L2 and 8.43%
peak-relative. Agreement on the six tested 4B greedy fixtures does not establish
numerical acceptance. Tolerances, references and model arithmetic were not
changed to hide these failures.

The separate native CLI suites pass 48 teacher cases and 58 rejection checks
for each model. They compare 6,231 teacher logit/state tensors for 0.8B and 8,231 for
4B, with exact Python/C++ parity. A rerun of the 4B CLI suite corrects an inherited
summary flag: `native_bitwise` is now set only after successful exact comparisons.
The original report is retained alongside the corrected rerun.

A retained 1K schema 3 4B bundle was also exercised through the old runtime and
the new runtime after conversion into a separate schema 4 bundle. The new
sidecars retain the old 512-token prompt and 1,024-slot cache bounds. Generated
IDs and FP32 logit trace bytes match exactly, and the original sidecars,
compiled packages and shared weight file remain unchanged. This checks metadata
migration, not larger-context compatibility of an old package.

### Native memory at 16K

Fresh native processes consume all 16,384 positions through 32 prefill calls,
checking position and finite logits after each call. Each process has one shared
weight owner and one runtime state; accuracy collectors and Python model copies
are absent. Values are GiB, not decimal GB:

| Measurement | 0.8B | 4B |
| --- | ---: | ---: |
| Allocated after load | 1.612 | 8.413 |
| Inference peak PyTorch allocated | 1.706 | 8.599 |
| Inference peak PyTorch reserved | 1.811 | 8.707 |
| Maximum boundary-sampled total device use | 2.334 | 9.231 |

Allocator peaks cover allocations between samples but exclude allocations
outside the PyTorch allocator. Total-device use is sampled with `cudaMemGetInfo`
after synchronized boundaries, includes CUDA context/library memory and can miss
temporaries released inside a compiled call. It is not an exact global device
peak. Both full-context probes complete without OOM. The new profile has not
been tested on GB10.

### Warm benchmark and cold-load cost

Each length has three rotated HF/native rounds, two warmups and five measured
requests per round (15 measured requests), with 64 teacher-forced decode steps.
The harness checks idle boundaries on the selected GPU and retains worker,
environment, order and telemetry records. HF prefill processes the whole prompt
in one call; native prefill uses chunks of at most 512 tokens. The numbers below
are medians; warm time to first prediction includes the complete prompt loop and
host argmax/synchronization, and excludes loading, CUDA initialization and text
tokenization. Decode is synchronized teacher-forced throughput, not a GPU-only
kernel measurement or free-generation benchmark.

The HF workers explicitly report the PyTorch DeltaNet fallback because
Flash Linear Attention is absent. These comparisons use that ordinary HF
baseline, not an HF installation with optimized FLA kernels or a serving engine.

| Model / prompt | Warm TTFT HF / native (ms) | Decode HF / native (tokens/s) |
| --- | ---: | ---: |
| 0.8B / 512 | 41.371 / 20.447 | 54.92 / 357.59 |
| 0.8B / 8,192 | 424.133 / 336.988 | 55.11 / 329.19 |
| 4B / 512 | 75.850 / 68.987 | 39.60 / 78.21 |
| 4B / 8,192 | 1,360.542 / 1,135.861 | 39.23 / 76.72 |

Median load times are HF/native 0.198/6.180 seconds for 0.8B and 1.002/32.329
seconds for 4B. Native loading includes full SHA256 verification; its individual
costs were not decomposed. The warm numbers therefore do not describe cold
request latency. Benchmark prediction agreement also does not replace the
numerical gate above: 4B differs on one of 65 predictions at length 512.

Generated evidence is retained under `artifacts/scaling/{0.8b,4b}/` in the
isolated validation workspace: `packages/` (4B uses `packages-clean/`),
`validation/`, `runtime/`, `benchmark/`, plus the corrected 4B `runtime-r2/`.
The validation report, native traces, package/native SHA256 identities, separate
memory-probe reports, raw benchmark workers and idle/telemetry records remain
outside Git.

The first export attempts exited during loader teardown after writing their
bundles. A loader-only control reproduces the fault with both the retained 1K
and fresh 16K packages when a login-shell profile puts a profiler's bundled C++
standard library before the image's library. Both pass with the image's normal
environment. All accepted exports were rerun through direct `docker exec` or a
non-login shell and exit normally. No model cleanup or export workaround was
added; see the environment guide for the shell requirement.

The environment follow-up removed the exact inherited Nsight library export
from `/root/.bashrc` in the running test container, retaining a backup and the
image's Torch/CUDA library selection. Both retained 1K and fresh 16K loaders
subsequently exited normally under direct execution and login shells, with the
system `libstdc++` mapped. Native 0.8B generation also matched IDs and FP32 logit
trace bytes across those shell modes. An actual Nsight 2024.2.3 capture through a
login shell exited normally and recorded three CUDA kernels and one NVTX range
while the target continued to load the system C++ runtime. These controls isolate
the environment exit crash; the separate full-model HF numerical failures above
remain open.

An experimental Dockerfile variant applied the same guarded startup-file
correction and selected the installed profiler executable ahead of the CUDA
SDK's stale wrapper on `PATH`. It built successfully as
`xinfer/l20:torch2.13-cu126-envfix`; fresh containers passed direct/login library
and profiler-launcher checks, and login-shell C++ generation matched the
corrected running container bit-for-bit. This image and its checks are retained
as experimental evidence, not as the default environment.

On 2026-11-07, the user decided to leave the Dockerfile unchanged and keep the
temporary workaround. The Dockerfile changes from the image experiment were
withdrawn. Scripted jobs continue through direct `docker exec` or non-login
`bash -c`; a required login shell uses a task-specific backup and removes only
the offending global export in the running container, followed by a fresh
`docker exec`. Profiling uses the installed Nsight launcher by absolute path,
preserving the image's Torch/CUDA library paths. The original image tag remains
the default. A future NVIDIA official base-image evaluation is deferred in
[the project plan](../../PLAN.md); this record does not select or build a new base.

## Export capacity and optional prompt policy — 2026-11-07

Both model families now select compiled capacity through `--max-context-length`
(default 16,384), validated before GPU model loading against the checkpoint's
declared text-config `max_position_embeddings`. Both Qwen3.5 checkpoints declare
262,144 positions. The current graph profile requires capacity at least 512 and
keeps prefill calls dynamic over 2–512 tokens; a singleton uses decode.

Schema 5 records compiled `cache_length` and call bounds but removes the former
`max_prompt_length` package field. Prompt caps are optional deployment policy via
the native `--max-prompt-length` option or shared runtime API. Without a cap, the
prompt plus requested output budget must fit context capacity. Budgets that do
not fit remain explicit errors. A cap can be smaller than 512 and changes
admission only; cache allocation and graph bounds remain fixed. Runtime ABI 1,
Flash ABI 1 and shared-weight ownership are unchanged.

On L20 with the previously recorded Torch 2.13 environment, both sizes pass
native/Python AOTI bitwise checks on separately converted 16K bundles for:

- A 128-token prompt with an explicit 128-token cap and two requested outputs.
- An uncapped 8,193-token prompt with two requested outputs.
- An uncapped 16,383-token prompt with one requested output.
- Five rejection cases per model: cap overflow, zero cap, cap larger than
  capacity, oversized output budget and a full-context prompt with no output room.

The converted copies preserve the original `.pt2` and weight bytes. Original
schema 4 sidecars remain unchanged; migration only sets schema 5 and removes
their prompt-policy field. The earlier 8K policy can be retained explicitly at
deployment. No fresh 4B export was needed for these metadata/policy checks.

A fresh 0.8B paired export at 2,048 slots completes ordinary compilation, the
full HF → eager → Python AOTI → C++ chain and independent FP32 controls. Native
parity and the native CLI suite pass. HF numerical acceptance remains **open**:
17,566 required failures comprise 17,527 convolution-state comparisons, 34
recurrent-state comparisons, four top-token comparisons and one greedy sequence.
These are the same failure types as the earlier 16K run; no logit-error threshold
fails. Worst AOTI/HF logit relative L2 is 3.14483%, and peak-relative error is
1.49375%. Neither references nor tolerances were changed.

The common CPU groups cover 176 distinct Python tests; both native builds pass
generation/state contract checks. Functional benchmark smokes pass with and
without a deployment cap using the 2K 0.8B package, identical HF/native policies,
one warmup, one measured request per case and one process round. The uncapped
workload uses lengths 128/513 and the capped workload length 128. These runs
verify policy propagation and do not support performance conclusions.

Evidence is retained under `artifacts/context-policy-20261107/` in the isolated
validation workspace: `retained-{0.8b,4b}/`, `{0.8b,4b}-policy/`,
`0.8b/{packages-2048,validation,runtime}/`, and `benchmark-{uncapped,capped}/`.
Full reports, native outputs, workload/idle/telemetry records and artifact/source
identities remain outside Git. Dockerfile changes remain withdrawn; the temporary
environment workaround and future official-base evaluation above are unchanged.
