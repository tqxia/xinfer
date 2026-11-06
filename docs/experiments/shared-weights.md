# Shared AOTI weights on L20

Both model exporters produce weightless prefill/decode packages and a single
CPU archive. Both native loaders borrow tensors from one retained device pool
using `load_constants(constants, false, true, true)`. Python uses
`load_constants(..., check_full_update=True, user_managed=True)` and each
compiled model retains its pool owner. Native pool members precede loader
members so the loaders are destroyed first.

The archive's internal `extra/weights.json` record maps each package's exact
runtime FQNs to tensor views over deduplicated storages. No separate
`weights.json` is deployed. Shape, stride, storage offset, dtype and
device are preserved. This also handles tied embeddings and Qwen3.5's packed
linear parameters. Compiler-generated/lifted constants have per-entry mappings;
names are captured during lowering rather than matched by suffix.

Each package's JSON metadata contains the archive filename, identity, size and
SHA-256 checksum. The checksum covers the embedded mappings and weight data;
the mappings also record the exact package hashes. Mismatched or incomplete
bundles are rejected before GPU allocation. One export invocation loads the
model once and compiles prefill and decode sequentially with matching settings.
Their source tensors must agree. The exporter accumulates both constant mappings
and writes the archive once, then records its final checksum in both metadata
files. It stages all files in a sibling temporary directory and publishes the
completed directory only after both entries succeed. Choose a new or empty
output directory; populated directories are preserved.

Deploy five files: `prefill.pt2`, `decode.pt2`, `prefill.json`, `decode.json` and
`weights.pt`. Current runtimes require format 2 and Torch 2.13 or newer. Re-export
older shared-weight bundles, and retain old binaries for embedded-weight bundles.
Graph text and packing reports are diagnostics and can remain with build records.

## L20 measurements, 2026-11-02

The results in this section describe the initial format-1 shared-weight run,
which used a sidecar manifest. They establish the storage-sharing baseline;
format-2 validation after sidecar removal is recorded separately.

The comparison uses Python 3.12, Torch 2.13.0+cu126, Triton 3.7.1, driver
535.161.07 and one NVIDIA L20 (SM89, 14,336 MiB). The embedded baseline is
commit `8b54072`, compiled under the same Torch installation. Checkpoints and
numerical/compiler settings are identical apart from external constant storage.

Each memory sample runs a fresh native process, warms the CUDA context and a
small GEMM, loads both packages and state, then performs an eight-token prefill
and one-token decode. `cudaMemGetInfo` includes AOTI allocations outside PyTorch's
caching allocator. Values below are medians of two samples, including context
memory. These are memory measurements, not latency benchmarks.

| Model | Embedded, after loading | Shared, after loading | Embedded, after decode | Shared, after decode | Saved after decode |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 2,764 MiB | 1,626 MiB | 2,764 MiB | 1,626 MiB | 1,138 MiB |
| Qwen3.5-0.8B | 3,270 MiB | 1,884 MiB | 3,302 MiB | 1,936 MiB | 1,366 MiB |

Native loading checks that every actual runner constant aliases the supplied
pool, including shape/stride/dtype/device. Loading savings must exceed 80% of
one unique CUDA weight set; the comparison fails otherwise. Activation and
allocator reservations account for the small difference between loading and
post-decode savings.

The storage-sharing regression passed for both models. All 79 Qwen3 output
files (454,738,432 bytes) and all 35 Qwen3.5 output files (9,774,436,352 bytes)
matched the embedded baseline byte for byte, with identical case manifests.
Native/Python parity passed in both formats. Qwen3.5's full run checked 14,406
teacher-forced tensors across 294 steps; its additional runtime suite checked
5,243 teacher tensors, 288 greedy state tensors and 32 rejection cases. Qwen3's
CLI generation, limits and package/input rejection checks passed. Eight focused
weight tests and 32 environment/cache/packing/benchmark unit tests passed.

Qwen3 retained its existing bounded-logit acceptance and the same 14 HF cache
differences / two top-1 cases. Qwen3.5's full HF numerical validator still exits
nonzero, with the same 4,722 state comparison failures and one greedy sequence
difference as the embedded Torch 2.13 baseline. No numerical limits or model
algorithms were changed. `comparison.json` keeps this failed full numerical
status separate from the passed sharing regression. Weight/package payloads
also shrank by about 49.8% for each model.

## Format 2: archive-only mappings

The sidecar-free run is `artifacts/l20/shared-weights-archive-20261102/`.
This historical validation used separate export invocations; the current CLI
exports the complete pair in one invocation. The deployment format is unchanged.
Both entry metadata files contain the same `shared_weights` descriptor:

```json
{"format": 2, "file": "weights.pt", "identity": "...", "bytes": 123, "sha256": "..."}
```

`bytes` and `sha256` describe the actual archive. In that run, the second export
refreshed the first entry's descriptor while preserving its other metadata. The
exporter prepared both complete JSON files before publishing the archive/package
files under its lock. Readers verify the checksum before reading
`extra/weights.json`. No standalone mapping file is created or required.

Ten archive tests and 35 environment/cache/packing/benchmark tests passed.
Both native builds, Python/native full-model comparisons and CLI rejection
checks passed, including missing embedded metadata and mismatched archive
checksums. Qwen3.5 has 34 rejection cases in this run. The strict comparison
again matches all 79 Qwen3 and 35 Qwen3.5 output files byte for byte. Post-decode
GPU memory remains 1,626 MiB / 1,936 MiB respectively. Existing HF numerical
failure cases remain unchanged. See this run's `comparison.json`.

The first Qwen3 re-export used a fresh Inductor cache. Although graph texts,
Triton source, all 311 stored weight buffers and mappings were identical, the
autotuner selected `XBLOCK=1` instead of `8` for a decode reduction kernel.
Eight capital-prompt output files differed; logits stayed within the existing
limits and generated tokens were unchanged. Re-exporting with the retained
cache restored identical executable kernel sections and bitwise outputs. The
format-only comparison therefore holds tuning choices fixed; no numerical
settings or correctness limits were changed. The cold-cache packages, report
and evidence are retained as `qwen3/cold-cache-*`, `comparison-cold-cache.json`
and `tuning-diagnostic.json`. Fresh-cache recompilation is a separate source of
compiler variation from archive serialization.

Benchmark comparability checks use original weight-source hashes across two
package pairs. They allow differing archive checksums because each archive
contains its own compiled-package hashes. Within a pair, the complete archive
descriptor must match. Supplied Qwen3.5 correctness reports must also match the
current archive descriptor.

## Complete-pair export validation

The pair-export refactor follows commit `1b2f2d6`. One command compiles both
entry points, with one archive serialization and one completed-directory
publication. Results are in `artifacts/l20/pair-export-20261102/`.

Twenty-two CPU tests passed, including real AOTI sharing of tied and strided
weights, fresh compiler options for each entry, and failure of the second
compilation or metadata factory without publishing a partial bundle. Both CLI
checks passed: the positional `kind` is rejected and populated outputs are
preserved before model loading.

Both models were re-exported on the L20 using the retained tuning caches to
hold compiler choices fixed. Qwen3 passed its full Python/native validator and
CLI suite. Qwen3.5 passed all 35 native runtime cases, including 5,243 teacher
tensors and 288 greedy state tensors, plus the input/package rejection checks.
All 79 Qwen3 validation files and 42 Qwen3.5 runtime files matched the previous
incremental-export outputs byte for byte. Embedded constant storage/view maps
were identical. Fresh-process post-decode memory remained 1,626 MiB for Qwen3
and 1,936 MiB for Qwen3.5.

Qwen3 retained the same 14 HF cache differences, two top-1 differences and no
greedy differences. The Qwen3.5 full HF numerical suite was not rerun for this
export-flow change; its previously documented numerical issues remain open.
Large native state dumps were hashed and removed after parity verification.

## Reproduce and inspect

Use the versioned environments/build from the [L20 guide](../../environments/l20/README.md).
Export each model's complete pair with one command into a fresh directory:

```bash
"$Q3_PY" -m compiler.qwen3.export --model-dir "$Q3_MODEL" --output "$Q3_PACKAGES"
"$Q35_PY" -m compiler.qwen35.export --model-dir "$Q35_MODEL" --output "$Q35_PACKAGES"
```

Run the normal Python/native correctness validators. The focused CPU tests are:

```bash
CUDA_VISIBLE_DEVICES='' "$Q3_PY" -m unittest compiler.common.tests.test_weights
```

The fresh-process memory probe builds separately against either runtime's
headers. `TORCH_PREFIX`, `CMAKE`, `CUDA_RUNTIME_FLAGS` and `BUILD` below come from
the [environment workflow](../../environments/README.md):

```bash
"$CMAKE" -S runtime/common/tests -B "$BUILD/weight-memory" -G Ninja \
  -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DTORCH_CUDA_ARCH_LIST=8.9 "${CUDA_RUNTIME_FLAGS[@]}" \
  -DNLOHMANN_JSON_INCLUDE_DIR="$BUILD/_deps/nlohmann_json-src/include" \
  -DXINFER_ATTENTION_OPS_LIBRARY="$BUILD/libxinfer_attention_ops.so" \
  -DCMAKE_BUILD_TYPE=Release
"$CMAKE" --build "$BUILD/weight-memory" -j2
"$BUILD/weight-memory/qwen_weight_memory" qwen3 "$Q3_PACKAGES" memory-qwen3.json
"$BUILD/weight-memory/qwen_weight_memory" qwen35 "$Q35_PACKAGES" memory-qwen35.json
```

For an embedded comparison, configure the same probe with
`-DXINFER_RUNTIME_ROOT=/absolute/path/to/8b54072/runtime` and its matching
attention library. Re-export the baseline using that checkout and the same
Torch 2.13 model environments; the current runtime rejects embedded bundles.

The retained run is `artifacts/l20/shared-weights-20261102/`, with `qwen3/` and
`qwen35/` subdirectories. Its `comparison.json` records identical case manifests,
exact native-output byte digests, Python/native parity, unchanged numerical
failure cases, GPU savings and deployment size. The experiment-specific report
generator is no longer maintained.

Large native `.states` files are removed only after Python/native parity and
SHA-256 recording; `native-output-digests.json`, case manifests, summaries,
selected final-state fixtures and packages are retained. This avoids keeping
another set of multi-gigabyte state dumps. Qwen3.5's existing HF numerical
acceptance remains separate from storage-sharing regression checks. No 4B
full-model result is claimed by this run.
