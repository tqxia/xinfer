# Package metadata

Both model families use `schema_version: 2` in `prefill.json` and `decode.json`.
`compiler/common/metadata.py` builds, finalizes and reads the shared Python
format. `runtime/common/spec.h` is the common C++ reader. Model-specific modules
validate their supported tensor ABI and retain their state-update policies.

| Fields | Meaning |
| --- | --- |
| `schema_version`, `model_type`, `entry_point`, `token_range` | Format, model selection and entry contract |
| `torch_version`, `cxx11_abi`, `cuda_compute_capability` | Runtime compatibility |
| `checkpoint`, `vocab_size`, `eos_token_ids` | Verified checkpoint identity, vocabulary and an explicit EOS list |
| `batch_size`, `dtype`, `cache_length`, `state` | Tensor shapes, types and ownership |
| `attention_backend`, `custom_attention`, `cache_input`, `kv_extent`, `input_mutation` | Shared Flash/prefix-cache contract |
| `export_options`, `tf32`, `triton_enable_fp_fusion`, `inductor_configs` | Export and numerical contract |
| `package_bytes`, `package_sha256`, `shared_weights` | Package and archive integrity |
| `model_specific` | DeltaNet numerical policy and minimal packing settings; empty for Qwen3 |

Every ordered state descriptor gives its `layer`, `kind`, `shape`, `dtype` and
`update` policy. KV inputs use `inplace`; conv/recurrent inputs use `replace`.
Readers derive state names and contiguous strides. The model ABI defines input
order, logits shape and output order: Qwen3 returns only logits, while Qwen3.5
also returns every state tensor, including aliased KV outputs. Qwen3 cache
geometry and Qwen3.5 layer types are derived from state rather than duplicated
under `model_specific`.

Both entries must agree on all stable fields, including `model_specific`.
Entry point, token range and package size/hash may differ. Unknown or missing
versions, legacy aliases, invalid tensor contracts and mixed pairs are rejected
before device allocation. File sizes and SHA-256 checks bind each descriptor to
its compiled package and shared archive. Checkpoint manifests remain available
to verify that eager references use the same checkpoint.

`inductor_configs` records the effective contract flags
`emulate_precision_casts=True`, `aot_inductor.package_constants_in_so=False` and
`always_keep_tensor_constants=True`. Other resolved compiler options are logged,
so compiler tuning can be compared without changing the numerical contract.
Dependency versions, Torch's CUDA version, export strictness, graph counts,
symbolic ranges, packing reports and compilation timings are printed as JSON
export events. They are not required package fields; `provenance` and
`diagnostics` are no longer serialized in the deployment sidecars. Retain the
export log when investigating a build.

Metadata schema version 2 is independent of `shared_weights.format: 2` and the
embedded weight-archive format. Schema 1 and unversioned sidecars are not read
by current runtimes. Re-export with the current exporter, or convert verified
retained sidecars into a new directory using the canonical schema. This
metadata-only conversion can keep compatible `.pt2` files and `weights.pt` byte
for byte; the archive binds package hashes, not JSON sidecar hashes. Keep the
original bundle and binary when retaining historical evidence. Runtime readers
contain no legacy-field translation paths.

## Historical schema 1 validation: 2026-11-03

Converted bundles are retained under
`artifacts/l20/metadata-schema-20261103/migrated/{qwen3,qwen35}`. Their compiled
packages and archives reuse the original files. All 61 native cases match the
retained output hashes, and all six spec-ownership cases pass. Fresh exports for
both models also match native/Python AOTI outputs and retained hashes on eight
focused cases. The readers agree on 197 malformed-metadata rejections; the CLI
suites pass 46 Qwen3 and 58 Qwen3.5 rejection cases. CPU and optimized-Python
checks and both benchmark harnesses pass. Benchmark runs used a shared GPU and
establish functionality only. Reports, source digests and original binaries are
retained in the same artifact directory.
