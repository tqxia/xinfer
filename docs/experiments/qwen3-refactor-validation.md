# Qwen3 single-profile refactor validation — 2026-10-30

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
