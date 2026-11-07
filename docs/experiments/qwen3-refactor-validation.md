# Qwen3 single-profile refactor validation — 2026-10-30

Current numerical validation uses `python -m tools.validation.validate` and
generates HF references during the run, including Qwen3's HF `generate()`
cross-check. The separate reference-preparation command and saved-reference
prerequisite have been removed. CLI-contract tools remain under
`tools/validation/qwen3/`. The recorded commands below retain this experiment's
source module paths; see the [current model guide](../../compiler/qwen3/README.md)
for a new run.

The Qwen3 cleanup was implemented against source revision
`799647189a6466aba938c5bf4575e0a9e4c04ae5` on the L20 development host, using
PyTorch `2.7.0+cu126`, CUDA build 12.6, Transformers 4.51.3 and the matching
LibTorch C++11 ABI. Qwen3.5 and shared model/runtime code were not changed.

The canonical compiler now always uses the existing Flash adapter, BF16 cast
emulation, TF32 disabled and normal fusion. The native operator library is
mandatory. There are no Qwen3 numerical-mode or attention-backend CLI switches.
The independent HF SDPA eager reference and FP32 controls remain available.

## Verification

| Check | Result |
| --- | --- |
| CMake build | CLI, validation driver, benchmark and attention library pass |
| CPU numerical-policy regressions | Five tests pass |
| Retired CLI arguments | Both rejected before export/model loading |
| Direct attention/operator checks | Nine direct and nine compiled cases pass |
| Adapter checks | Nine shape/length cases and rejection checks pass |
| Fresh decode/prefill export | Both pass without either retired selector |
| Python AOTI validation | Fixtures, boundaries, full-cache state/reuse and FP32 controls pass |
| Native/Python AOTI validation | 26 cases, 185 logit rows, one pair of loaders pass |
| Native CLI smoke | Three fixtures, four boundary/budget cases, 22 input/metadata rejections pass |
| Deployment | No Python library linkage/loading; sanitized native subprocesses pass |
| Benchmark `run` | HF eager / one AOTI profile, two rotated rounds pass |
| Benchmark `compare` | Previous Flash / fresh Flash, two rotated rounds pass |

The policy tests cover accepted bounded top-token drift, excessive logit error,
the independent peak-normalized bound, nonfinite inputs and signed-zero bitwise
parity. Native smoke completes its subprocesses before loading the Python AOTI
pair, avoiding concurrent parent/child model copies. Every subprocess records
both streams and its failure diagnostics include its return code.

Both benchmark smoke runs use lengths 1, 2, 25, 128 and 512, three forced decode
steps, one warmup and one measured repetition per round. The shared GPU was busy;
results are marked `provisional_shared_gpu`. They verify the harness and do not
establish a performance improvement.

## Numerical results and refactor equivalence

All 26 native logit trace files and `context_end.kv.bf16` match the saved
pre-refactor `artifacts/validate-flash-default/` outputs byte-for-byte. Thus the
refactor preserves the previous fused Flash arithmetic on this workload.

- Maximum HF logit relative L2: **3.8491949%**.
- Maximum HF logit peak-normalized error: **3.2751091%**.
- Both remain below the unchanged 5% logit limits.
- HF full-cache diagnostic differences: **14/56** tensors.
- HF top-token differences: **two cases**, the same prefix-boundary row and its
  poisoned repeat; no independent greedy-fixture differences were observed.
- Native/Python AOTI logits and full-cache tensors remain bitwise identical.

The full report explicitly records `passed: true`, `hf_parity_passed: false` and
`hf_reference_limits_passed: false`. Overall acceptance now enforces runtime
correctness and bounded logits; it does not claim exact HF arithmetic or tokens.
Cache/token drift is retained as evidence, not hidden or treated as a precision
fix. The numerical-history document records the earlier policy and comparisons.

## Artifacts and reproduction

Reports are under `artifacts/qwen3-refactor-20261030/` in the development checkout:

- `attention/`, `adapter/`, `validation/` and `smoke-sequential/`.
- `benchmark/` and `comparison/`, including source snapshots and hashes.
- `refactor-equivalence.json`, `retired-cli-checks.json`, `verification.json`.
- `package-metadata/`, with graph text, compiler metadata and Python validation.

Disk space was limited, so the fresh approximately 1.2 GB packages were exported
to `/dev/shm/xinfer-qwen3-refactor-20261030/packages`. This is temporary storage,
not a durable deployment path. Existing packages and user reports were preserved.
Compilation reused the existing tuning cache at
`/data/workspace/develop/xinfer-main-integration/work/inductor-flash` to keep the
arithmetic comparison controlled. Cold-build time was not measured.

The main [Qwen3 guide](../../compiler/qwen3/README.md) gives the persistent build,
fresh-directory export and validation commands. For the full matrix used here:

```bash
AOTI_RUNTIME_CHECK_INPUTS=1 .venv/bin/python -m compiler.qwen3.tests.validate_cpp \
  --packages /dev/shm/xinfer-qwen3-refactor-20261030/packages \
  --native build/qwen_validate --output artifacts/qwen3-validation-next
.venv/bin/python -m compiler.qwen3.tests.smoke_cpp \
  --packages /dev/shm/xinfer-qwen3-refactor-20261030/packages \
  --output artifacts/qwen3-native-smoke-next
```

The existing efficient package directory is intentionally not overwritten. It
is rejected with re-export guidance by the new runtime and loaders. Legacy
eager-numerics packages are also rejected; ordinary Flash packages keep their
existing operator/extent ABI.

## Context scaling validation: 2026-11-06

A fresh Qwen3-0.6B package pair was exported and validated on L20 with Torch
`2.13.0+cu126`, Triton `3.7.1`, Transformers `4.51.3` and matching native
LibTorch. The Flash adapter, BF16 cast emulation, disabled TF32 and existing
numerical limits remain in place.

Metadata schema 4, runtime ABI 1 distinguishes the **8,192-token request prompt
limit** from the **512-token maximum compiled call**. Prefill graphs accept
2–512 tokens; decode accepts one token. Shared request handling partitions a
long prompt into calls of at most 512 tokens, dispatching a one-token remainder
to decode, and returns the completed prompt's final logits. The allocated
context capacity is 16,384 tokens. A larger cache requires fresh paired export;
changing an old bundle's metadata does not enlarge its compiled state tensors.

| Check | Result |
| --- | --- |
| Qwen3/common CPU test group | 132 tests pass |
| Full numerical chain | 29 teacher cases, 307 matched-input steps and 10 greedy requests |
| State comparisons | 29 Python checkpoints; all 26 selected native checkpoints checked bitwise |
| Independent FP32 controls | 132 matched-input steps, including 513/8,192-token prompts and full-context cases |
| Native/Python AOTI | Bitwise agreement for every teacher and greedy logit row and selected state tensor |
| Native CLI | Three chat fixtures, seven boundary/budget cases, 54 rejection checks and 65 native processes pass |
| Deployment | No Python linkage/loading; sanitized subprocesses and shared-weight checks pass |

The teacher matrix covers prompts of 1, 511, 512, 513, 8,191 and 8,192 tokens,
nonzero offsets and one-token remainders. Full-context cases reach position
16,384 and reject another token without mutating position, state or storage.
Poisoned future KV, reset/reuse and repeated requests retain their checks.
The CLI boundary cases include the 513/8,191/8,192-token requests. The
8,191-token greedy case reaches EOS after its first prediction; the complete
prompt is consumed before stopping.

HF, adapted eager, Python AOTI and native AOTI receive identical teacher-forced
partitions. Expanded-context snapshots retain populated KV prefixes and complete
functional state; unused future fill is checked independently. Long cases keep
selected prompt/final checkpoints and every logit row. Per-call native checks
still enforce finiteness, write regions and caller-owned storage. The three
short chat-fixture cache checkpoints remain Python-side diagnostics under the
existing Qwen3 policy; they are not included in the 26 native state checkpoints.

| Logit comparison | Maximum relative L2 | Maximum peak-normalized error |
| --- | ---: | ---: |
| Adapted eager / HF | 4.71235% | 2.92398% |
| Python AOTI / adapted eager | 3.47174% | 2.15311% |
| Python AOTI / HF | 4.52264% | 3.46154% |
| Native / Python AOTI | 0% | 0% |

The full chain passes the existing acceptance policy with zero required
failures. Direct AOTI/HF teacher comparisons retain **seven diagnostic top-token
differences**; all **10 greedy sequences agree**. HF cache errors and independent
FP32 differences remain diagnostic rather than claims of exact HF parity. The
FP32 controls have maximum relative-L2 errors of 5.21664% for HF BF16 and
5.55963% for AOTI; these controls do not replace the unchanged 5% HF-BF16 logit
gates.

A separate native probe fills the complete 16K context. After loading, PyTorch
allocated memory is 2.860 GiB; inference peaks are 2.892 GiB allocated and
2.904 GiB reserved. Synchronized boundary samples observe 3.248 GiB total device
use, a 2.965 GiB increase over CUDA initialization. That sampled device figure is
not an exact global peak; allocator peaks and device-wide samples measure
different memory scopes.

Package metadata, numerical and CLI reports are retained under
`artifacts/scaling/qwen3/`, with the fresh bundle in `packages/`, the common
chain in `validation/` and deployment checks in `runtime/`.

### Warm benchmark

The matched fixed-token benchmark uses three rotated process rounds, two
warmups and five measured requests per round: 15 warm samples per model/length.
GPU idle checks pass before and after workers. The HF baseline uses ordinary
HF DynamicCache with the whole prompt in one call; native prefill uses at most
512 tokens per call. Every request consumes the same 64 teacher-forced decode
tokens, treating EOS as input data.

| Prompt tokens | HF warm TTFT (ms) | Native warm TTFT (ms) | HF decode (tokens/s) | Native decode (tokens/s) |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 16.226 | 10.495 | 65.76 | 353.80 |
| 8,192 | 207.749 | 204.867 | 65.71 | 243.60 |

Warm TTFT includes reset, the complete prompt, argmax and its synchronized host
read. It excludes model loading, CUDA initialization and tokenization. The 8K
TTFT ratio is only 1.014×, so this run establishes essentially unchanged warm
first-token latency there, rather than a broad long-prefill speedup.

Separately measured median loading times are 0.993 seconds for HF and
4.789 seconds for native. Native loading includes artifact SHA-256 verification;
the individual loading components were not timed separately. These costs are
excluded from the warm TTFT table.

HF/native prediction IDs agree on all 65 predictions per measured request for
both lengths. This benchmark agreement is a harness result, not a replacement
for the numerical acceptance report above. Benchmark workload, identities,
worker samples, idle checks and telemetry are retained in
`artifacts/scaling/qwen3/benchmark/`.

## Export capacity and optional prompt policy — 2026-11-07

The exporters now read the declared text-model context ceiling and accept
`--max-context-length`, defaulting to 16,384. Qwen3's pinned config declares
40,960; capacities outside 512..40,960 fail before GPU model loading. The 512
minimum retains the existing `[2,512]` compiled prefill profile.

Schema 5 removes the deployment prompt cap from package metadata. Native
generation accepts an optional `--max-prompt-length`; without it, the complete
prompt and requested output budget need only fit the compiled capacity. Oversized
budgets remain errors. Runtime ABI 1, state ownership, shared weights and the
512-token call limit are unchanged. The earlier schema 4 results above remain
historical evidence, including their explicit 8K request policy.

Verification on the same L20/Torch 2.13 stack completed:

- 176 distinct Python CPU tests pass across the shared and model-specific
  groups, including the additional malformed benchmark-policy regression.
  Both native builds and their generation/state CPU contracts pass.
- A separately converted 16K bundle passes native/Python AOTI bitwise generation
  for a capped 128-token prompt and uncapped 8,193- and 16,383-token prompts.
  Cap overflow, zero/oversized caps and insufficient output space are rejected
  before output creation. Original schema 4 sidecars and compiled files remain
  untouched; the converted bundle reuses the original graphs and weights.
- A fresh paired export with `--max-context-length 2048` passes the full
  HF → eager → Python AOTI → C++ chain, independent FP32 controls and native CLI
  checks. Required numerical failures: zero. Worst AOTI/HF logit relative L2 is
  4.06475%, and peak-relative error is 2.92398%, within the unchanged 5% limits.

Evidence is retained under `artifacts/context-policy-20261107/`: `retained-qwen3/`,
`qwen3-policy/`, and `qwen3/{packages-2048,validation,runtime}/`. This verifies
capacity selection and request admission; it does not establish a performance
change or GB10 compatibility for the new packages.
