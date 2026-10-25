# Canonical FlashAttention path on GB10

Validated on 2026-10-25 in `/home/tqxia/workspace/xinfer`, branch `dev-gb10`,
using PyTorch 2.13.0+cu130, Triton 3.7.1 and Qwen3-0.6B BF16.
The tested working tree starts at `3e5f5b7`; exact source hashes are recorded in
[source-manifest.json](source-manifest.json).

## Result and code changes

`--attention-backend flash` now selects the read-only custom operator and
post-export graph pass. It is the only Flash exporter. The no-flag default
remains `efficient`, and the chat helper still uses the accepted eager-numerics
packages.

- [Exporter](../../compiler/export.py): capture ordinary SDPA, rewrite attention,
  re-export to rebuild metadata, then compile. The `flash-custom` CLI choice is
  removed. New metadata identifies `flash` with the custom-op ABI contract.
- [Model wrapper](../../compiler/model.py): remove the HF Flash adapter,
  attention registration, mask monkeypatch, dynamic prefix slicing, and padded
  cache allocator branch. Cache updates remain fixed-size in the exported graph.
- [Python registration](../../compiler/attention_ops.py) and [native runtime](../../runtime/runtime.h):
  require the custom-op contract for Flash. Old direct packages fail with a
  clear `re-export with --attention-backend flash` instruction. The legacy
  `flash-custom` package label remains accepted for already compiled custom-op
  packages, including their earlier padded layout.
- Remove the adapter-dependent `compiler/tests/flash_attention.py`; retain the
  [custom-op suite](../../compiler/tests/custom_attention.py), including its
  independent FP32 reference and compiled boundary checks.
- Update setup and usage docs. Preserve the [original issue explanations](../../docs/gb10-flash-attention-implementation.md)
  with 24 verified links pinned to their original source revision.

The operator ABI and native kernel implementation are unchanged. Exports use
contiguous caches, `cache_padding: 0`, and `backed_size_oblivious: false`.
The fixed 1,024-token buffers and shape-only CPU `[L+1]` extent contract remain.

## Validation

| Check | Result |
|---|---|
| Operator suite | 9 direct cases and 9 compiled cases passed; read-only inputs, poisoning, extent contents, opcheck, FP32 reference, invalid lengths and unknown-mask rejection |
| Full graph audit, each entry point | 28 SDPA calls replaced with 28 custom-op calls; all 56 cache writes retained; zero direct-ATen Flash, SDPA or dense-mask calls after rewrite |
| New canonical vs previous unpadded custom packages | All 26 native traces / 185 logit rows and all 56 final cache tensors bitwise identical; SHA-256 comparisons saved |
| Native vs Python AOTI | All 185 logit rows exact; tested top-1 predictions agree with HF BF16 |
| Native smoke | Three fixtures, prompt lengths 1/2/512, generation limit, 20 rejection cases, sanitized environment and Python-free linkage passed |
| Old direct packages | Actual archived packages rejected by Python and C++ before loading weights, with CUDA hidden for the rejection check |
| Old custom packages | Both padded and unpadded legacy-label packages loaded; native capital fixture and logits exact against their saved references |
| Efficient eager-numerics regression | Passed 24 cases / 163 rows and all 56 full-cache tensors with exact HF/Python AOTI/C++ agreement |
| Benchmark defaults | Default efficient baseline and new canonical Flash candidate ran successfully; five predictions agree; idle boundary checks passed |

The full Flash validator still exits **1** for the same **17/56** cache tensors
exceeding the unchanged 5% numerical limits. Its complete case records and failure
records match the previous custom-op run. This cleanup introduces no numerical
improvement or regression on the tested workloads. Structural correctness and
bitwise parity with the previous Flash implementation do not remove that existing
full-model BF16 limitation.

The benchmark check uses one 25-token prompt, one round, one warmup, one measured
request per profile and four forced decode steps. It verifies default wiring;
it is not a new performance comparison. See the [last full custom-op benchmark](../gb10-custom-attention-unpadded-2026-10-25/REPORT.md)
for measured latency results. No CUDA Graph replay was added.

## Package migration and commands

The old direct packages were moved from `artifacts/aoti-flash` to
`artifacts/aoti-flash-direct-legacy`. Newly compiled custom-op packages now occupy
`artifacts/aoti-flash`, so the comparison harness's default candidate selects
the canonical implementation. Old custom package directories remain available.

The relevant commands were run after `cmake --build build -j2`, with
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, offline model loading, and
`AOTI_RUNTIME_CHECK_INPUTS=1` for correctness checks. Export used
`TORCHINDUCTOR_CACHE_DIR="$PWD/work/inductor-custom"` and
`TORCHINDUCTOR_COMPILE_THREADS=2`.

```bash
.venv/bin/python compiler/tests/custom_attention.py --output artifacts/custom-attention-unit-canonical
.venv/bin/python compiler/export.py decode --attention-backend flash --output artifacts/aoti-flash
.venv/bin/python compiler/export.py prefill --attention-backend flash --output artifacts/aoti-flash
.venv/bin/python compiler/validate_cpp.py --packages artifacts/aoti-flash --output artifacts/three-path-flash-canonical
# Expected nonzero exit for the unchanged numerical failures described above.
# Populate package reference/ from the completed validation's *_teacher.aoti.f32 traces.
.venv/bin/python compiler/tests/smoke_cpp.py --packages artifacts/aoti-flash --output artifacts/native-smoke-flash-canonical
.venv/bin/python compiler/validate_cpp.py --packages artifacts/aoti-eager --output artifacts/three-path-eager-after-flash-cleanup
unset AOTI_RUNTIME_CHECK_INPUTS
.venv/bin/python compiler/benchmark.py compare --lengths 25 --rounds 1 --warmup 1 --repeats 1 --decode-steps 4 --output artifacts/benchmark-flash-default-smoke
```

These paths already exist; use fresh output directories for another run. The
custom operator library `build/libxinfer_attention_ops.so` must be built before
export and deployed with the native runtime and matching LibTorch.

## Saved evidence

- [Operator validation](operator-validation.json), [full Flash validation](flash-validation.json),
  [efficient regression](efficient-regression.json), [native smoke](native-smoke.json).
- [Graph audit and package metadata/hashes](graphs.json), [bitwise parity hashes](parity.json),
  [migration contract checks](contracts.json), [legacy custom compatibility](legacy-custom-compatibility.json).
- [Benchmark smoke results](benchmark-smoke.json) and [environment/source hashes](benchmark-environment.json).

Raw graphs, packages, logits and logs remain in the corresponding ignored
`artifacts/` and `work/canonical-*.log` paths on GB10.
