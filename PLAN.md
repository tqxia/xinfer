# Qwen3-0.6B AOTInductor / C++ PoC

Status: steps 1-4 complete. Step 5's benchmark is implemented and functionally validated; final performance measurements will run in a separate environment because the development host is fully loaded. The optional eager-numerics compilation mode passes the complete three-path validation with bitwise agreement. Original fused packages are retained with their documented numerical differences. Step 6 remains planned.
Date: 2026-10-23.
Remote host: root@kikoxia-any4.devcloud.woa.com:36000.
Project: /data/workspace/develop/qwen3-aoti-poc.

The separate [Qwen3.5-4B text-only extension](compiler/qwen35/README.md) has completed
stage 1: a real DeltaNet block exports and compiles for T=64/T=1, with bitwise
C++/Python AOTI parity and passing eager/FP32 diagnostics. Its remaining stages
are recorded there; the plan below describes the original Qwen3-0.6B PoC.

## Goal and scope

Demonstrate a complete autoregressive generation loop for Qwen/Qwen3-0.6B:
load the original model in Python, export and AOT-compile its computation,
then load the compiled packages and generate tokens in a C++ executable.

The user selected token IDs as the runtime interface. An offline Python helper
will apply the official chat template, tokenize prompts, and decode saved output
IDs. The executable will perform all prefill, decode, cache management, greedy
selection, and stopping without invoking or embedding Python.

Initial scope:
- One visible L20 GPU, batch size 1, BF16 weights and KV cache.
- Dense Qwen3-0.6B, using the existing Hugging Face implementation.
- Prompt lengths 1–512 tokens; fixed KV capacity 1,024 tokens.
- Default generation limit 64 tokens; validate the requested context budget.
- Non-thinking chat template and greedy decoding for reproducible validation.
  Greedy decoding is a diagnostic choice, not Qwen's recommended quality setup.
- Separate prefill and decode AOTI packages, one native generation loop.
- No CUDA Graphs, quantization, concurrent requests, paged attention, HTTP server,
  custom CUDA/Triton kernels, or native tokenizer in this first milestone.

Steps 1-5 establish the minimal, validated C++ baseline. Step 6 extends the
scope to performance optimization, including CUDA Graphs, graph transformations
and high-performance kernels. Preserve the baseline so each optimization can
be evaluated independently, while keeping the default code path readable.

## Confirmed environment

- /data/workspace/abo/.venv: Python 3.12, torch 2.7.0, triton 3.3.0.
- Transformers, tokenizers, safetensors, and huggingface_hub are absent there.
- CMake exists at /data/workspace/abo/.venv/bin/cmake (4.3.4).
- Ninja and GCC 12.3.1 are available.
- The visible NVIDIA L20 exposes 14,336 MiB; it was idle during inspection.
- The existing benchmark already uses AOTIModelPackageLoader.
- Model files already exist at:
  /data/workspace/develop/qwen3-rust-demo/models/qwen3-0.6b
- That directory's manifest records model revision:
  c1899de289a04d12100db370d81485cdf75e47ca

Reuse the model directory read-only. Verify its recorded checksums during setup.
Keep dependencies project-local and leave the ABO environment unchanged. Start
with the existing torch 2.7.0 build exposed to an isolated project environment;
install only the additional pinned dependencies there. Transformers 4.51.3 is
the initial candidate (Qwen3 requires >=4.51.0), subject to the compatibility
probe. Build the executable against the exact Torch installation used to
compile its packages. Record package versions, ABI, GPU architecture, and CUDA.

## Architecture and tensor contract

Offline:
  Hugging Face Qwen3 -> small tensor-only wrapper -> torch.export
  -> AOTInductor -> prefill.pt2 + decode.pt2 + metadata

Native:
  prompt IDs -> prefill -> greedy token selection -> repeated decode -> output IDs

Use the same Python wrapper for both exports. Reuse model layers and weights;
do not copy the Qwen implementation into this repository. A small cache adapter
will bridge flat tensor arguments to the Hugging Face forward API.

Proposed exported signature:
- input_ids: int64 CUDA tensor [1, T]
- cache_position: int64 CUDA tensor [T]
- per-layer K/V tensors: 28 pairs, each BF16 [1, 8, 1024, 128]
- result: logits for the last supplied token, [1, 151936]
- side effect: update only the supplied positions in the caller-owned KV tensors

The C++ runtime allocates the cache once and passes that same storage to both
packages. Construct the causal mask from absolute positions inside the wrapper
and mask unused cache entries. Explicitly verify that export/AOTI preserves the
input-mutation contract and that C++ observes the updates. Check generated code
for full-cache copies or accidental replacement storage; semantic correctness
alone is not sufficient to claim an efficient persistent-cache path.

Prefill: bounded dynamic T=2..512.
Decode: fixed T=1, with the position supplied as tensor data.
A one-token prompt uses the decode package at position zero. This keeps the
T=1 specialization explicit and avoids padding/bucket logic in the PoC.

Prefer native PyTorch SDPA with an explicit mask; verify its export and AOTI
lowering in the chosen versions. An eager/math attention fallback may be used
to isolate a correctness failure, and must be identified in any timing report.

Use two independent loaders initially. Accept potential duplicated immutable
weights for this small model; measure actual memory rather than introducing
weight-sharing machinery. Both loaders must still share the caller's KV tensors.
The planned BF16 cache is 112 MiB, calculated from
2 * 28 * 8 * 1024 * 128 * 2 bytes. Other buffers and weights are additional.

## File layout

```text
README.md                     Setup/export/build/run/validate commands and results
PLAN.md                       Design and acceptance criteria
CMakeLists.txt                Root entry point; build executables in build/
compiler/requirements.*         Pinned Python dependencies
compiler/prompts.json           Reference prompt fixtures
compiler/model.py               Thin Qwen wrapper and explicit cache adapter
compiler/export.py              Load weights, export/compile graphs, write metadata
compiler/prepare.py             Offline tokenization and output-ID decoding
compiler/validate.py            Eager reference generation and regression cases
compiler/validate_aoti.py        Cache, compiled-package and FP32-control checks
compiler/validate_cpp.py         Three-path comparisons and FP32 investigation
compiler/benchmark.py           Eager worker, process orchestration and summary
compiler/tests/smoke_cpp.py      Native parity, CLI and Python-free loading checks
compiler/tests/rope_precision.py Minimal BF16 RoPE contraction reproducer
runtime/CMakeLists.txt            Native targets and matching LibTorch dependencies
runtime/runtime.h                 Shared native loader, cache, step and generation
runtime/main.cpp                  Native CLI, input/output and diagnostic timing
runtime/benchmark.cpp             Fixed-token benchmark using the shared Runtime
runtime/tests/validate_native.cpp Repeated-request and cache checks using that Runtime
```

Run scripts and CMake from the repository root. Python code uses sibling imports;
no installable Python package or import-path shim is needed. Build products,
artifacts and the project virtual environment keep their existing root paths.

Ignored generated directories: .venv/, build/, artifacts/, work/.
Use simple token-ID files and a small metadata file; avoid a general framework,
configuration hierarchy, serialization framework, or duplicated model code.
Keep cache layout and position semantics documented beside the code that uses them.

## Implementation sequence

1. Establish the environment and eager reference.
   Verify model files, pin dependencies, load the local checkpoint in BF16, and
   prepare a few non-thinking prompt fixtures with the official tokenizer.
   Record reference next-token logits and a short greedy sequence.

2. Prove the export/cache contract before building the full application.
   Use a focused cache-update probe if necessary, then run the Qwen wrapper
   eagerly. Compare it with the unmodified HF model. Export and compile decode,
   verify several successive positions, then export prefill and verify handoff
   into decode using the same cache. Test T=1 separately from dynamic T>=2.
   Resolve export failures here; do not spread workarounds through the C++ loop.

3. Implement the minimal C++ executable.
   Use AOTIModelPackageLoader and LibTorch tensors. Load metadata, validate token
   IDs and context limits, allocate zeroed GPU cache, execute prefill, select a
   token using GPU argmax, and call decode until EOS or the generation limit.
   Read EOS IDs from the recorded generation configuration (the local model
   lists 151645 and 151643). Support saving generated IDs and an optional
   validation trace. A scalar synchronization per step is acceptable initially.

4. Validate the three execution paths.
   Compare HF eager, Python-loaded AOTI, and the standalone C++ executable.
   Use teacher-forced identical token continuations for logit comparisons so
   that a sampling divergence does not invalidate all later comparisons.
   Also check end-to-end greedy token sequences and repeated requests.
   Implemented: 24 cases, 163 logit rows, full-cache dump/bitwise comparison,
   poisoned future slots, final legal slot, overflow rejection and cache reuse.
   Native parity and logits pass. Full-context AOTI vs HF BF16 cache values fail
   the existing 5% limits on 21 of 56 tensors. Preserve the failing status and
   FP32 evidence; do not mark acceptance complete or relax the limits silently.
   Root-cause investigation locates the first divergence at layer-0 RoPE:
   backend BF16 multiply-add contraction removes an intermediate rounding step.
   Disabling contraction improves eager parity but leaves 15 peak-limit failures
   and worsens worst-case FP32 agreement in this case. Keep it experimental.
   Completed with --eager-numerics: disable contraction and retain native ATen
   mean reductions. All 163 logit rows and all 56 full-context KV tensors match
   HF BF16 and C++ bitwise. The original packages and 5% limits are preserved.
   Keep FP32 controls in passing reports; eager parity is not improved FP32
   accuracy. Native smoke/loading and mixed-profile rejection checks also pass.

5. Measure and document the completed PoC.
   Record export/build commands, pinned versions, correctness results, compilation
   time, package sizes, model-load time, inference-only first-token latency,
   steady-state decode milliseconds/token and tokens/second, and peak GPU memory.
   Compare against the same eager workload, dtype, cache and attention setup.
   Measure the accepted eager-numerics profile and the original fused profile
   separately; native reductions may add latency, while the fused profile has
   the numerical differences documented in step 4.
   Separate warmup from measurement, reset its cache, and synchronize CUDA at
   measurement boundaries. Identify host synchronization included in timings.
   Do not assert a performance target or a vLLM comparison at this stage.
   Implemented: separate native benchmark target and a matching eager worker;
   fixed tokens at prompt lengths 25/128/512 followed by 64 decode steps;
   three fresh-process rounds with rotating profile order, warmup and per-request
   cache reset. Record first-use requests separately from measured requests.
   Report allocator peaks and sampled CUDA device-memory usage, because AOTI's
   weight allocations bypass PyTorch's allocator. Device samples are not exact
   transient or per-process peaks. Preserve source/package hashes and telemetry.
   Shared-GPU runs validate the harness only; final measurements require idle
   checks before and after every worker. Step 5 remains incomplete until those
   measurements and their analysis are recorded.

6. Explore and validate performance optimizations.
   Start from the measured C++ baseline in step 5. Preserve its source revision,
   packages, correctness fixtures and benchmark results. Profile prefill and
   decode separately across prompt lengths and populated cache lengths, including
   short contexts and a nearly full cache. Quantify kernel time, CPU launch and
   synchronization overhead, allocations, memory traffic and peak GPU memory.
   Investigate the observed full-capacity GQA expansion first, without assuming
   it dominates end-to-end latency before measurement.

   Explore the following techniques, using profiling to prioritize experiments:

   - Existing Torch/Inductor options: inventory what the pinned build actually
     supports for AOTI, then test relevant GEMM autotuning, kernel selection,
     fusion, layout and memory-planning options. Record exact settings, compile
     cost and numerical effects. Do not assume JIT-only options work with AOTI.
   - CUDA Graphs: begin with fixed-shape decode in the native C++ runtime. Keep
     input, position, output and KV storage stable; update token/position values
     between replays. Verify warmup/capture, cache reset, stream ordering, memory
     lifetime and EOS handling. Measure remaining host synchronization and graph
     memory overhead. Consider prefill capture/buckets only if profiling supports
     their additional code and memory cost. CUDA Graphs target launch overhead;
     attention changes below target redundant GPU work and memory traffic.
   - Custom graph optimizations: inspect the exported/lowered graph and test
     small, isolated transformations before AOT compilation. Priorities include
     avoiding materialized KV-head repetition, reducing work on unused context,
     and fusing relevant norm/RoPE/cache-update operations when measured useful.
     Evaluate any context-length bucketing or specialization against extra
     packages, memory, dispatch complexity and CUDA Graph compatibility.
   - High-performance kernels: evaluate existing GQA-aware attention and
     KV-cache kernels compatible with L20 and the selected toolchain. Prefer
     reusing suitable kernels; add targeted custom kernels only for remaining
     measured hotspots. Any custom operator needs export support and native
     registration/linkage so the C++ process remains independent of Python.

   Compare each change independently with the baseline, then test useful
   combinations. Reuse identical tokens, shapes, dtype, context lengths and
   stopping rules. Report prefill latency, decode ms/token and tokens/s,
   end-to-end latency, memory, compile/capture cost and repeated-run variability.
   Separate device-only gains from gains visible through the C++ generation loop.
   Rerun numerical, greedy, mutation, masking, boundary and repeated-request checks
   for every candidate; investigate discrepancies without silently relaxing the
   existing correctness criteria. Validate graph replay across changing positions
   and fresh requests, plus every new bucket or dispatch boundary.

   Deliver a baseline/individual/combined comparison table, reproducible commands,
   and a small optional optimized path with documented tradeoffs. Keep changes
   that provide repeatable improvements with acceptable complexity and memory;
   record unsuccessful experiments too. Set performance targets from the step 5
   measurements rather than promising an unmeasured speedup.

## Acceptance criteria

- A fresh C++ process loads the packages and generates without Python execution.
  Confirm the binary does not link libpython/libtorch_python, and run the native
  command independently of the preparation/verification helpers.
- Original HF versus wrapper, wrapper versus AOTI, and Python AOTI versus C++
  comparisons pass with documented BF16-appropriate error criteria.
- Check intermediate logits under identical fed tokens. Report maximum/relative
  error and top-1 agreement. Investigate greedy divergence with logit margins;
  do not conceal mismatches by merely relaxing tolerances.
- Cover prompt lengths 1, 2, an ordinary short prompt, 128, 511, and 512.
- Cover multiple decode steps, EOS, generation limits, cache reset/reuse,
  processing different prompts in succession, and the final legal cache slot.
- Reject invalid token IDs, empty/oversized prompts and invalid context budgets
  before launching model work.
- Verify prefill/decode cache handoff and preserved input-mutation semantics.
- Decode generated IDs offline to show readable text alongside numerical checks.
- README commands reproduce the result, and implementation remains a short
  model adapter plus a straightforward native loop.

## Risks and deliberate tradeoffs

Step 2 resolved export and persistent-cache mutation with the original version
pins. Both native packages are in `artifacts/aoti/`; README records commands,
tensor contracts, numerical checks and results. The eager wrapper matches HF
StaticCache exactly. AOTI matches all fixture tokens with documented BF16 error
limits and an independent FP32 control. Generated code writes directly into the
caller cache, but still expands the full padded K/V inputs for attention.
Step 3 added the standalone C++ loader, input rejection and native trace capture.
The three fixture sequences and all 43 logit rows match Python AOTI bitwise;
native T=1, T=2 and T=512 smoke cases also pass. Linkage and loader tracing confirm
the binary does not load libpython or libtorch_python.

Step 4's complete matrix gives bitwise native/Python AOTI parity on 163 logit
rows and all 56 full-context KV tensors. All tested top-1 tokens, greedy
sequences, masks, write regions, resets, repeated requests and overflow checks
pass. The full-context cache numerical check fails on 21 tensors: maximum
relative L2 is 6.5913% and peak-relative error is 17.8385%, against unchanged 5%
limits. The first divergence is now isolated to layer-0 RoPE multiply-add
contraction, with an exact arithmetic reconstruction and a minimal reproducer.
Recompiling both packages with TRITON_DEFAULT_FP_FUSION=0 makes the observed
layer-0 stages equal to eager and reduces worst full-cache relative L2 to
2.0656%, but 15 cache tensors still fail the peak limit. Worst FP32 agreement
also worsens, so this control is not promoted. The next difference appears in
layer-1 Q RMSNorm: a different FP32 reduction order pushes 3 values across BF16
rounding midpoints in the traced prefix. A probe of the generated kernel
reproduces this exactly.
README records the evidence and distinguishes eager parity from FP32 accuracy.
The optional --eager-numerics mode also retains aten.mean.dim as a native
LibTorch operation, dispatched through AOTI's C++ proxy executor in Torch 2.7.
It closes step 4 acceptance: all 163 logit rows and all 56 full-context cache
tensors match eager BF16 bitwise, with no tolerance changes. The native runtime
remains Python-free. Metadata prevents mixing numerical modes in a package pair.
This profile inherits eager BF16's FP32 errors and is not an accuracy improvement.
Step 5's benchmark is implemented; baseline measurements on a separate
environment are pending.
Step 6 now includes paired native benchmarks, numerical-error collectors, an
optional NVTX profiler and the experimental valid-prefix FlashAttention/native
GQA path. See [development tools](docs/development-tools.md) and
[FlashAttention](docs/flash-attention.md). Flash is disabled by default and its
full-model numerical acceptance remains unresolved. CUDA Graph integration,
FlashInfer and projection packing are not part of this main-branch change.
Keep both the accepted correctness baseline and original fused packages for
comparison; historical GB10 measurements remain on `dev-gb10`.

The initial technical risks were HF cache adaptation and mutation lowering in
torch 2.7.0, followed by dynamic prefill export. Both passed in step 2. Any future
version change must keep export-time Torch and native LibTorch matched.

Fixed-capacity attention does work over padded cache storage and expands KV
heads. Two loaders may duplicate weights. Greedy decoding and scalar
synchronization prioritize a readable correctness PoC. Step 5 measures the
impact of these tradeoffs; step 6 explores compiler, runtime, graph and kernel
optimizations against that baseline. Keep export-time Torch and native LibTorch
matched if an optimization experiment requires an isolated version change.

## References

- Qwen model card:
  https://huggingface.co/Qwen/Qwen3-0.6B
- Qwen configuration:
  https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json
- HF Qwen implementation, proposed initial version:
  https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py
- PyTorch 2.7 AOTInductor native deployment:
  https://docs.pytorch.org/docs/2.7/torch.compiler_aot_inductor.html
