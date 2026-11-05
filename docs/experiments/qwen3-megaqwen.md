# Optional MegaQwen decode for Qwen3-0.6B

This experiment adds a native CUDA decode backend to the existing AOTI runtime.
AOTI handles the first call and every multi-token chunk. MegaQwen handles later
single-token calls against the same verified weights and KV storage. The default
backend remains `aoti`; select `megaqwen` explicitly in an enabled build.

The integration is runnable experimental code with **failed numerical acceptance**.
The final full validation completed with six required HF logit-limit failures;
state/ownership contracts passed. Successful loading and plausible generation do
not establish numerical acceptance. The final rebuild retains the original fixed
BF16 boundaries and `--fmad=false` after rejecting two focused arithmetic probes.

The final report is `work/hybrid-validation-final/summary.json`. It reproduces the
initial numerical failures with the final source and library. It covers 22
teacher-forced cases (138 logit rows), seven generation cases, and 63 matched HF
greedy-history rows per backend. The backend remains explicitly experimental.

The user deferred performance measurements to another environment. No decode
speedup, latency target, or throughput result is claimed here. Validation on the
shared development GPU is correctness work; its elapsed time is not a benchmark.

## Execution and ownership

`runtime/common/runner.h` owns position, input validation, reset, state storage,
generation, and dispatch. `runtime/common/decode.h` defines the optional native
decode interface. The runner uses this rule after checking the request:

| Call | Selected implementation |
|---|---|
| First call after construction/reset, including length 1 | AOTI |
| Any call with more than one token | AOTI prefill |
| Later call with one token | Selected decode backend |

Thus a one-token prompt first runs the compiled AOTI decode entry point at position
zero; its following single-token call uses MegaQwen. An appended multi-token chunk
also returns to AOTI after MegaQwen has already updated the cache. Later one-token
calls resume MegaQwen. The native backend has no independent position counter or
reset operation that could disagree with the runner.

The first call consumes all prompt tokens and returns the first prediction. If
its length is `P`, the next call consumes the next supplied/generated token at
position `P`. The final prompt token is not fed again. Teacher-forced validation
uses the same supplied tokens even when the two engines choose different argmax
IDs, so differences are measured on matched histories.

`SharedWeights::source_tensor()` resolves semantic source names through the
verified `weights.pt` manifest. MegaQwen borrows those tensors from the same device
pool already used by the two AOTI loaders. It does not load another HF model or
allocate/copy another weight set. Names include the attention adapter's `original`
component, for example
`model.model.layers.0.self_attn.original.q_proj.weight`. Tied embedding/LM-head
storage remains shared.

The backend also borrows all 56 existing `[1,8,1024,128]` contiguous BF16 KV tensors.
A small per-layer pointer table supplies K and V addresses directly to the kernel.
There is no aggregate cache allocation, layer transpose, prefix copy, or cache
adoption operation at the handoff. AOTI and MegaQwen append into identical buffers;
the runner remains their owner. Backend construction verifies shape, stride,
dtype, device, 16-byte alignment, weight addresses, and cache addresses before use.

The added allocations are pointer metadata, BF16 cosine/sine tables, intermediate
scratch, and one independently owned BF16 `[1,151936]` output tensor per call.
`decode_info()` reports the workspace size and ownership evidence. The runner
destroys the borrowing backend before its state and canonical weight owner.

## Kernel and numerical choices

The implementation under `runtime/qwen3/megaqwen/` derives from upstream MegaQwen
commit `0bc1fdedb82744481373000c5c7d3c58795c7a3a` and the repaired standalone source.
See that directory's `PROVENANCE.md` and `UPSTREAM_LICENSE_NOTICE.txt` for the exact
source hashes, observed upstream license declaration, and adaptation details.

Each native decode call enqueues two kernels on the caller's current CUDA stream:

1. A cooperative kernel runs embedding, all 28 transformer layers, and final norm.
2. A vocabulary-projection kernel computes the full logits once and writes BF16
   output.

The shared generation/benchmark caller still performs argmax and reads the scalar
prediction. Those operations are outside the two-kernel count. This adaptation
does not use the standalone implementation's fused argmax path or its diagnostic
path that computes the LM head twice. It retains the existing logits-returning
runtime contract, trace format, and stopping rules.

The transformer uses row-per-warp GEMV, Q/K head norms and RoPE, direct GQA cache
access, online softmax, and global scratch between phases. It executes eight grid
barriers per layer, plus the embedding barrier: 225 per token. Its scalar FP32
dot-product implementation does not become a tensor-core GEMM merely because its
weights use BF16. Major intermediate vectors still pass through global scratch.
Other blocks issue explicit L2 weight-prefetch hints during the attention phase;
their benefit needs an independent measured comparison.

The cooperative grid is derived once from visible SM count and the compiled
kernel's occupancy limit. The inspected L20 has 92 visible SMs, so its default grid
has 92 blocks of 256 threads. The backend checks cooperative-launch availability
and legal residency rather than hardcoding the original 82-block RTX 3090 grid.

The retained implementation rounds the repaired source's selected intermediate
boundaries to BF16 and accumulates dot products and reductions in FP32. Weights,
KV, inter-layer hidden vectors, and public logits use BF16. Compilation uses
`--fmad=false` without `--use_fast_math`. These choices do not imply bitwise or
bounded-error equivalence to HF/AOTI. `decode_info()` describes the fixed
implementation.

Raw CUDA launches record the use of borrowed and owned CUDA allocations with
PyTorch's caching allocator on first use of each distinct stream, deduplicating
storage bases. This protects allocation lifetime when a runtime is destroyed
while queued work still uses its buffers. It does not order independent streams:
callers must establish dependencies when moving a runner's work between streams.
A runner's mutable KV/scratch is intended for ordered calls, not concurrent
requests. The decode implementation adds no CPU synchronization inside `run()`.

## Numerical and state results

The final full BF16-boundary run failed six required HF logit checks. Each row
below occurred once in the clean case and once in its poisoned-future repeat;
step indices are zero-based:

| Case/step in the final full run | Call backend | HF logit relative L2 error |
|---|---|---:|
| One-token prompt, step 1 | MegaQwen | 5.5140% |
| Mixed handoff back to AOTI, step 4 | AOTI | 5.9753% |
| Mixed handoff back to AOTI, step 5 | MegaQwen | 8.1349% |

The AOTI call at step 4 consumes history written by earlier MegaQwen calls.
The separate handoff test seeds two AOTI calls with identical incoming hybrid
cache and obtains bitwise equality, so returning to AOTI preserves that history
correctly. The numerical divergence remains unresolved.

The existing 5% logit limits were not relaxed. Native AOTI/Python AOTI bitwise
parity and all structural checks passed. The updated asynchronous-destruction
probe also passed while retaining 134,508,608 bytes of allocator churn. These
results establish the tested integration/state behavior, while the logit failures
prevent numerical qualification. Worst HF logit error is 8.1349% relative L2 and
5.1667% peak-normalized. Greedy sequences agree on the tested fixtures; six
top-token comparison differences and 87 numerical diagnostics remain recorded.

The validated build's `libxinfer_megaqwen.so` SHA-256 is
`76f11fce9ae4bebe677f7fe4962cef8102c4b8b556836c7980d46c890cf33355`.
The report also records executable, contract-helper, attention-library, package,
and shared-weight hashes. Backend workspace is 578,400 bytes, excluding each
303,872-byte independently allocated logits tensor; model weights and the 112 MiB
KV cache remain shared with AOTI.

Both enabled and disabled builds compile. The 54 CPU Python tests, 15 CPU state
contract cases, native default-AOTI ownership/preflight checks, and disabled,
unknown, and Qwen3.5 backend rejection checks pass. Pinned formatter checks pass.
An installed-prefix native generation smoke test performs two MegaQwen decode
calls under Compute Sanitizer memcheck with zero reported errors. This short
memory check complements the full structural suite; it is not a performance run.

Two focused probes were rejected and removed from the final code:

| Rejected focused probe | One-token prompt, step 1 | Mixed handoff, step 5 |
|---|---|---|
| FP32 intermediates | Relative L2 6.3012% | Relative L2 9.4636% |
| BF16 rounding of unnormalized attention weights | Focused one-token check passed | Relative L2 12.6553% |

These are focused probe results, not additional full-suite runs or failure counts.
The attention probe improved one case while worsening the mixed handoff. Neither
probe resolved acceptance, so the implementation returned to the original fixed
BF16-boundary arithmetic. Further work needs a same-input, stage-by-stage numerical
diagnosis rather than treating increased intermediate precision as a guaranteed
fix.

## Scope and environment

The experiment uses `Qwen/Qwen3-0.6B` revision
`c1899de289a04d12100db370d81485cdf75e47ca`, batch 1, vocabulary 151936, 28 layers,
hidden size 1024, FFN size 3072, 16 Q heads, 8 KV heads, head dimension 128,
RoPE theta 1000000, and cache capacity 1024. Prompt/multi-token calls accept at
most 512 tokens under the existing package ABI. The current target is NVIDIA L20
SM89. Other geometries, cache capacities, revisions, and GPUs need separate
engineering and qualification.

Use the current L20 Torch 2.13.0+cu126 / Triton 3.7.1 native stack and the Qwen3
Transformers 4.51.3 environment. The optional native core links ATen/LibTorch/CUDA
without PyBind, `libpython`, or `libtorch_python`. It is compiled ahead of time;
generation does not invoke a Python compiler or load the standalone MegaQwen venv.
The old standalone Torch 2.7 extension is not linked into this runtime.

The existing package pair, schema-3 metadata, shared-weight format 2, checksums,
checkpoint manifest, Torch version, GPU architecture, and C++ ABI remain checked.
The experiment does not replace `decode.pt2` with a CUDA shared object. It loads
the original bundle and adds an explicitly selected runtime backend. Its native
library must accompany an enabled deployment.

A build without `XINFER_ENABLE_MEGAQWEN` uses the disabled-backend stub and rejects
an explicit MegaQwen request. Qwen3.5 rejects that request as unsupported. Building
the optional library alone does not change ordinary `aoti` dispatch.

## Build on the existing remote host

The isolated checkout is `/data/workspace/develop/qwen3-aoti-megaqwen`. The examples
reuse the prepared native/model environments and immutable package bundle from
`/data/workspace/develop/qwen3-aoti-poc`; they do not upgrade those environments.
On another machine, provision its matching stack following `environments/README.md`
and replace these absolute paths.

```bash
cd /data/workspace/develop/qwen3-aoti-megaqwen
BASE=/data/workspace/develop/qwen3-aoti-poc
Q3_PY="$BASE/work/l20/qwen3-torch213-venv/bin/python"
CMAKE=/data/workspace/abo/.venv/bin/cmake
BUILD="$PWD/build/megaqwen"
MODEL="$BASE/work/models/qwen3-0.6b"
PACKAGES="$PWD/work/packages/qwen3"
REFERENCE="$BASE/artifacts/reference"
TORCH_PREFIX=$("$Q3_PY" -c 'import torch; print(torch.utils.cmake_prefix_path)')
CUDA_RUNTIME=$("$Q3_PY" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent.parent / "nvidia/cuda_runtime/lib/libcudart.so.12")')

"$CMAKE" -S . -B "$BUILD" -G Ninja \
  -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCUDA_cudart_LIBRARY="$CUDA_RUNTIME" \
  -DCUDA_CUDART_LIBRARY="$CUDA_RUNTIME" \
  -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON="$BASE/work/l20/build-torch213/_deps/nlohmann_json-src" \
  -DTORCH_CUDA_ARCH_LIST=8.9 -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DXINFER_MEGAQWEN_CUDA_ARCHITECTURES=89 \
  -DXINFER_ENABLE_MEGAQWEN=ON \
  -DXINFER_BUILD_VALIDATION=ON -DXINFER_BUILD_BENCHMARK=ON \
  -DCMAKE_BUILD_TYPE=Release
"$CMAKE" --build "$BUILD" --target \
  xinfer_aoti xinfer_validate xinfer_benchmark qwen_hybrid_contract -j2
```

The two CUDA runtime variables select the wheel's CUDA 12.6 runtime while retaining
host nvcc 12.2. The cached nlohmann source avoids downloading those headers again.
The environment guide explains why linking the older host runtime does not work
with this LibTorch. A recompiled library is a different candidate even if the
packages are unchanged.

## Run token-ID generation

Use a fresh output file/directory for each command. Preparation and decoding are
offline Python operations; the middle command runs model inference entirely in
the native executable.

```bash
mkdir -p work/megaqwen-run-001
"$Q3_PY" -m tools.prepare verify --model-dir "$MODEL"
"$Q3_PY" -m tools.prepare encode --model-dir "$MODEL" \
  --prompt 'What is the capital of France? Answer in one short sentence.' \
  --output work/megaqwen-run-001/prompt.ids

"$BUILD/xinfer_aoti" "$PACKAGES" work/megaqwen-run-001/prompt.ids \
  work/megaqwen-run-001/generated.ids \
  --max-new-tokens 64 --decode-backend megaqwen

"$Q3_PY" -m tools.prepare decode --model-dir "$MODEL" \
  --input work/megaqwen-run-001/generated.ids
```

The native stderr summary records `decode_backend` and `megaqwen_calls`. Use
`--decode-backend aoti` with another output filename for the control. An optional
`--trace PATH` writes full float32 logit rows for diagnostics. Generation timings
printed by this CLI include its actual execution boundary and are not controlled
benchmark results.

## Validate the hybrid

```bash
"$BUILD/qwen_hybrid_contract" "$PACKAGES" work/megaqwen-contract-001.json

"$Q3_PY" -m tools.validation.hybrid \
  --packages "$PACKAGES" --model-dir "$MODEL" --reference "$REFERENCE" \
  --native "$BUILD/xinfer_validate" \
  --contract-native "$BUILD/qwen_hybrid_contract" \
  --output work/megaqwen-validation-001
```

The full Python driver runs original HF StaticCache, Python AOTI, native AOTI, and
native hybrid on matched input manifests. The native worker runs under a sanitized
environment and proves that no Python library is loaded. The driver preserves
the original Qwen3 fixtures and adds prompt length 25 plus multi-token handoff
back to AOTI after MegaQwen. It checks dispatch counters so a silent fallback to
AOTI cannot appear to be a tested MegaQwen result.

The native contract helper checks first-call bitwise cache/logit parity for prompt
lengths 1, 2, 25, 128, and 512; continued single-token native execution; positions
1023/1024; reset/reuse; unchanged past/future cache regions; storage identity;
future-cache poisoning; invalid-input/overflow rejection without state advance;
and nondefault-stream behavior. To test the return handoff fairly, it seeds an
independent AOTI control with the hybrid's actual preceding cache, then requires
bitwise equality for the following AOTI multi-token call. It also includes the
asynchronous-destruction/allocator-churn lifetime probe.

Numerical acceptance follows the current Qwen3 policy: finite outputs and strict
state/runtime contracts, bitwise native/Python parity for the AOTI control,
bitwise initial AOTI handoff, and both existing 5% HF logit limits. Those limits
are row-relative L2 and peak-normalized absolute error. HF cache, top-token, and
free-running greedy differences are retained diagnostics rather than discarded
or presented as exact fidelity. HF logits on the hybrid's own generated history
are checked with matched fed tokens. Independent FP32 HF controls are also
recorded. There is no separate Python MegaQwen implementation in this integration,
so the report does not claim native/Python MegaQwen bitwise parity.

The driver exits 0 only when all selected hard checks pass; 1 means completed
acceptance failures; 2 means an exception/incomplete run. It writes
`summary.json`, native reports/logit and cache dumps, logs, and the native contract
report. Read the implementation details, required failure list, diagnostics, and
ownership scope together. `decode_info().validated` refers to the tensor/alias
contract, not an independently established numerical result.

To run a prepared native manifest directly:

```bash
"$BUILD/xinfer_validate" "$PACKAGES" CASES_JSON OUTPUT_DIR \
  --decode-backend megaqwen
```

Use the driver's generated manifest for `CASES_JSON` and a fresh `OUTPUT_DIR`.
These validation traces deliberately synchronize, inspect cache, and retain full
outputs. Their runtime and memory use are not decode performance measurements.

## Compare backends later

The benchmark interface is implemented for the user's later environment. It uses
the same package bundle, AOTI prefill, fixed teacher-forced continuation, request
reset, warmups, and timing boundaries for both selected native backends:

```bash
"$Q3_PY" -m tools.benchmark compare-backends \
  --packages "$PACKAGES" --model-dir "$MODEL" \
  --native "$BUILD/xinfer_benchmark" \
  --baseline-backend aoti --candidate-backend megaqwen \
  --lengths 25 128 512 --decode-steps 64 \
  --warmup 2 --repeats 5 --rounds 4 \
  --output artifacts/megaqwen-backend-comparison-001
```

Do not run this as an accepted performance comparison until the backend is
numerically qualified and the chosen GPU is available. The runner performs idle
checks and records backend dispatch counts, package/checkpoint/workload hashes,
native executable and linked library hashes, backend resources, first use,
memory samples, and telemetry. `--allow-busy` is reserved for functional harness
checks; its timings remain provisional. The current session intentionally leaves
the actual performance run to the other environment.

Record the selected build and library with every result. Changing arithmetic,
launch ownership, or a shared library creates a new candidate even when the
command's package paths remain the same. Compare
prefill, decode, request latency, memory, and numerical results together before
attributing a gain to mega kernel fusion.
