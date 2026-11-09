# Qwen3 implementation and deferred work

Current Qwen3-0.6B scope: one fused BF16 AOTInductor compilation profile using
valid-prefix FlashAttention/native GQA, explicit caller-owned cache state and a
Python-free C++ generation loop. The eager-numerics profile and efficient
compiled backend are retired. Authoritative commands and contracts are in the
[Qwen3 guide](compiler/qwen3/README.md); historical numerical investigations are
in [Qwen3 numerical history](docs/experiments/qwen3-numerics.md).

## Supported Qwen3 pipeline

- Load the original checkpoint read-only; preserve independent BF16 HF eager
  reference execution and offline tokenization/text decoding.
- Install the existing Flash module adapter before export. Export bounded
  dynamic prefill (2–512 tokens per call) and decode (one token) with the same ABI.
- Use batch-one BF16, caller-owned `[1,8,C,128]` K/V tensors and an explicit
  shape-only CPU extent input. Export chooses the context capacity with
  `--max-context-length`, defaulting to the model config's declared context ceiling.
  Requests share that capacity between prompt and output. Shared generation splits prompts into calls of at most
  512 tokens; a single-token chunk uses decode, including at position zero.
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

Run `python -m tools.validation.validate --packages DIR --output DIR` with the
matching model environment. Package metadata selects the family hooks. The shared
chain compares HF, eager, Python AOTI and C++ AOTI on matched inputs, checking
adjacent stages and a direct compiled/HF anchor. Both models generate HF
references during validation; Qwen3 also checks manual HF greedy decoding against
HF `generate()`. Saved reference files and a separate preparation command are not
required. Eager includes all adapters and linear packing selected by the packages.
CLI and deployment contracts remain
separate tools. Tooling unit tests live under `tools/<area>/tests/`;
compiler/model/operator and kernel-AOTI checks remain under `compiler/*/tests/`.

Use independent eager inputs and teacher-forced identical tokens to compare
logits. Hard requirements are finite outputs, bitwise Python AOTI/native logits
and caches, correct write regions/masking, stable storage, reset/reuse, boundary
and overflow rejection, and native/Python AOTI generation agreement. HF logits
retain the 5% relative-L2 and peak-normalized error limits. HF cache differences
and differing top-token/greedy choices are diagnostic reports, with HF parity
separate from overall acceptance. Full-model validation uses the BF16 HF baseline
without a separate FP32 HF control model.

The saved pre-refactor Flash matrix has 26 cases/185 rows, bitwise native/Python
AOTI parity, 14/56 HF cache tensors over the former limit and two top-token
mismatches. Independently tuned builds can differ. Re-export and validate both
entry points for every compiler/toolchain change; historical results are not a
claim that the new artifact passed. Current refactor verification is recorded in
[the validation record](docs/experiments/qwen3-refactor-validation.md).

## Measurement and next work

One metadata-selected `python -m tools.benchmark` command serves both models.
`run` compares ordinary HF dynamic-cache eager and one compiled profile through
the shared cached-forward helper; `compare` measures two compatible package
bundles; `check-idle` records the selected GPU activity without loading a model.
The benchmark prepares its own fixed tokens and derives workload limits from
package metadata. Accuracy validation remains a separate tool. Reports mark
the HF baseline
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

The [Qwen3.5 text-only pipeline](compiler/qwen35/README.md) defaults to 0.8B.
Full-model 4B bring-up uses the same preparation, export and runtime interfaces.
It exports paired dynamic prefill (2–512 tokens) and single-token decode
with packaged Triton DeltaNet kernels and shared native valid-prefix
FlashAttention. Caller-owned KV buffers update in place; conv/recurrent inputs
remain unchanged, and returned tensors become the next state. Both entries share
one weight pool and use metadata schema version 5 with runtime ABI 1, the common
runner and shared
native generation, benchmark and validation executables.

Ordinary L20 exports and C++ execution completed for the new long-context
profile for 0.8B and the complete 4B model. Native/Python AOTI outputs matched
bitwise, and full 16,384-slot inference-memory probes completed. HF numerical
acceptance remains open. Keep the existing BF16 HF comparisons and
Q35-NUM-001/Q35-NUM-002 diagnostics.
The [experiment record](docs/experiments/qwen35.md) preserves earlier block probes,
fixed-shape packages and export failures as historical evidence. Those results
do not establish full-model numerical acceptance.

The completed geometry bring-up generalized the kernels while retaining the
then-current 1,024-context full-model profile:

- DeltaNet supports `Hk=16`, `Hv=16` or `32`, and `K=V=128`: 6,144 or 8,192
  convolution channels. Both short and chunked paths index gates and recurrent
  state by value head. Grouped value heads share Q/K, with one convolution
  history writer per group; the chunked path prepares Q/K per value head.
- The shared native attention operator, fake shape contract and Qwen3.5 adapter
  derive grouped-head geometry from inputs. They accept 4B's 16 query heads,
  four KV heads and dimension 256, alongside the existing models.
- Batch one, width-four SiLU convolution, T=1..512, 1,024-token capacity, C64
  chunking and the provisional 97-token dispatch threshold remain unchanged.
  Kernel unit tests cover both geometries with HF recurrent/chunked and FP32
  controls, state and partition checks. `compiler.qwen35.tests.test_kernels_aoti`
  covers both geometries with BF16/FP32 export, decomposition, exact Python AOTI
  parity and nondefault-stream checks. Full-model validation retains C++ coverage.

The full-model 4B milestone selects its pinned checkpoint with
`python -m tools.prepare download --model Qwen/Qwen3.5-4B`; its default preparation directory
is `work/qwen35/4b/text`. Export and validation use that explicit `--model-dir`
and a separate `artifacts/qwen35/4b/text` package directory. The 0.8B defaults
remain unchanged. Both sizes require a verified prepared manifest, tied
embeddings and default RoPE.

Validate the complete shared-weight pair at the existing limits: logits,
state preservation, reset/reuse, generation and GPU memory. Measure export and
packed-reference peaks as well as inference memory. Use the
[L20 Docker testing environment](environments/l20/docker/README.md) for
the real 4B checkpoint. Export uses ordinary strict tracing and linear packing;
the standard validators run the BF16 HF baseline alongside the adapted and
packed eager stage and AOTI packages. Preserve all existing numerical,
state, native/Python bitwise and generation gates, including the 5% HF limits.
Retain complete reports before accepting the full-model milestone.

The historical [4B L20 rerun](docs/experiments/qwen35.md#full-4b-l20-validation-2026-11-04)
completed ordinary export, full FP32 controls, native parity, generation and
shared-weight memory checks without OOM. Its required HF/adapter/packing
numerical gates still fail, so the full-model milestone is not yet accepted.

The current export defaults derive from the selected model's context ceiling
(40,960 for Qwen3-0.6B; 262,144 for both Qwen3.5 checkpoints).
`--max-context-length` selects another export capacity within that ceiling.
Schema 5 records `cache_length` separately from compiled `token_range: [2,512]`.
There is no separate prompt cap. The full prompt plus the request's
`--max-output-tokens` budget must fit the compiled context.
Runtime ABI 1 and the attention
ABI remain unchanged. Generation and native benchmarks split the complete prompt
into calls of at most 512 tokens; a final single token uses decode. Converted
retained packages keep their original 1,024-slot capacity in separate bundle
directories. Any earlier prompt-admission policy belongs to the calling application.

The recorded 16,384-slot L20 profiles completed export, native execution and
full inference-memory probes for Qwen3-0.6B, Qwen3.5-0.8B and Qwen3.5-4B.
Native/Python AOTI parity passed for all three. Qwen3 passed that suite's HF
numerical gates; Qwen3.5 HF acceptance remains open. Fresh default-capacity
exports and full-capacity native checks also completed for Qwen3-0.6B and
Qwen3.5-0.8B with bitwise native parity. Their focused HF numerical checks retain
failures, recorded in [the interface validation follow-up](docs/experiments/common-runtime.md#model-derived-capacity-and-request-budgets--2026-11-07).
These profiles still need GB10 re-export and validation.

Keep regression coverage for 1/511/512/513/8191/8192-token prompts, a one-token
remainder, nonzero offsets, reset/reuse, the selected capacity boundary and
overflow/budget rejection. Run matched teacher-forced partitions through HF,
adapted/packed eager, Python AOTI and C++ AOTI, retaining the independent BF16 HF
baseline, strict native parity and existing HF numerical limits. Stream state
comparisons and retain selected checkpoints to bound CPU memory and disk use.

Use separate idle-target benchmarks for complete-prompt prefill and time to
first token. Memory probes record allocator peaks and synchronized device-memory
samples through the complete context; sampled device peaks are not exact peaks
of transient allocations. The ordinary HF benchmark still processes a whole
prompt in one call; report that partition difference separately from the
matched numerical validation. Compare larger per-call prefill profiles using
this validated native execution baseline and separately collected measurements.
Wider graphs and GDN workspaces need their own export/kernel checks before a
larger per-call default is selected.

Keep unsupported-shape checks and existing numerical limits. Preserve tested
tuning specializations where they still fit the target configuration.

## Environment follow-up

Keep the Nsight library-path correction as a temporary running-container
workaround; leave the Dockerfile unchanged for now. TODO: revisit the image using
an NVIDIA official base, retaining the pinned model environments and validating
AOTI export, C++ execution and profiling from clean shell environments.

## Qwen3.6-35B-A3B-FP8 integration

| Step | Scope | Status |
|---|---|---|
| 1 | Pinned text checkpoint and GB10 environment | Complete |
| 2 | FP8 loading/adapters and eager memory/execution | Complete; HF numerical acceptance open |
| 3 | Complete-model dynamic export and AOTI compilation | Complete at explicit 1,024-slot capacity |
| 4 | C++ runtime, shared-pool lifetime, memory and deployment | Complete on GB10, including installed/relocated CLI |
| 5 | Full HF → adapted → AOTI → C++ correctness | Matrix executed; HF/eager and compiler/eager numerical gaps retained |
| 6 | L20 integration and benchmarks | Pending; separate target artifacts and correctness gate |

Start on GB10 with pinned mixed-dtype, text-only checkpoint preparation and a
fresh independent environment. Keep FP8 weight/BF16 scale bytes, untied embeddings
and LM head, original generation EOS settings and complete source provenance.
Process source shards incrementally and verify transfers before publication.
See [preparation, loading and eager validation](docs/qwen36-fp8.md).

The mixed-dtype loader and thin FP8 adapters reuse the existing hybrid state,
attention and GDN execution path. On GB10, FP8-only eager replacement agrees
bitwise with the official HF FP8 execution across the four-layer prefix and all
40 layers. Combined adapters complete the functional matrix and short generation;
HF logit/state/hidden numerical acceptance remains open and must be investigated
before claiming full-model correctness. Tolerances and required checks stay unchanged.
The full-model 16,384-slot short memory probe peaks at 34.29 GiB allocated on
GB10, including validation overhead. Numerical investigation and proposed fixes
are recorded in [the Qwen3.6 numerical record](docs/experiments/qwen36-numerics.md).
Fixes are deferred while the current kernels proceed through dynamic 2–512
prefill, decode, quantized shared weights, Python-free C++ generation and carried
state checks. Report export/compilation, compiler-to-eager comparison and exact
native parity independently; these do not establish HF numerical acceptance.
The complete 40-layer dynamic prefill and one-token decode now compile on GB10
with a 1,024-slot cache and one shared FP8/BF16 weight archive. C++ passes all
1,991 bitwise comparisons against Python AOTI, including carried state and greedy
generation. Python AOTI executes the full matrix but fails 1,339 required
comparisons against adapted eager; all teacher argmax and short generated tokens
still agree. Record that compiler/eager numerical gap separately from the
existing HF adapter failures; neither numerical gate is accepted.
Step 4 also corrected the pinned Torch AOTI user-managed handle leak through a
guarded build-only header overlay and fresh packages. Live ATen allocation now
returns exactly to its warmed baseline after C++ runtime destruction. The
installed/moved runtime passes the native matrix, three production generation
cases and 18 CPU-only rejection probes; caller-Spec and file-preflight contracts
pass. The correction preserves all 1,991 old/new recorded outputs bitwise.
Use unchanged HF FP8 as
the primary reference for this checkpoint and retain BF16 block diagnostics
separately. Validate process peaks before extending to L20 or large contexts;
benchmark after correctness. Keep the existing Qwen3/Qwen3.5 paths unchanged.
