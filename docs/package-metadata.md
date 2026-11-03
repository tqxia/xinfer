# Package metadata

Both model families use `schema_version: 1` in `prefill.json` and `decode.json`.
`compiler/common/metadata.py` builds, finalizes and reads the shared Python
format. `runtime/common/spec.h` is the common C++ reader. Model-specific modules
validate their supported tensor ABI and retain their state-update policies.

| Fields | Meaning |
| --- | --- |
| `schema_version`, `model_type`, `entry_point`, `token_range` | Format, model selection and entry contract |
| `torch_version`, `torch_cuda_version`, `cxx11_abi`, `cuda_compute_capability` | Build compatibility |
| `checkpoint`, `vocab_size`, `eos_token_ids` | Checkpoint identity, vocabulary and an explicit EOS list |
| `batch_size`, `dtype`, `cache_length`, `state`, `input_order`, `output_order`, `logits` | Tensor shapes, layouts, types and ownership |
| `attention_backend`, `custom_attention`, `cache_input`, `kv_extent`, `input_mutation` | Shared Flash/prefix-cache contract |
| `export_options`, `tf32`, `triton_enable_fp_fusion`, `inductor_configs` | Actual export and numerical settings |
| `package_bytes`, `package_sha256`, `shared_weights` | Package and archive integrity |
| `model_specific` | Stable model geometry, adapters, DeltaNet policies and packing settings |
| `provenance`, `diagnostics` | Source/version provenance and per-entry graph/timing diagnostics |

Every state descriptor gives its layer, kind, shape, contiguous strides, dtype
and `update` policy. KV inputs use `inplace`; conv/recurrent inputs use
`replace`. Qwen3 returns only logits. Qwen3.5 also returns every state tensor,
including aliased KV outputs. These distinct tensor ABIs remain explicit in the
common format.

Both entries must agree on all stable fields, including `model_specific` and
provenance. Entry point, token range, package size/hash and diagnostics may
differ. Unknown or missing versions, legacy aliases, invalid tensor contracts
and mixed pairs are rejected before device allocation. File sizes and SHA-256
checks bind each descriptor to its compiled package and shared archive.

Metadata schema version 1 is independent of `shared_weights.format: 2` and the
embedded weight-archive format. Unversioned sidecars are no longer accepted.
Re-export with the current exporter, or convert verified retained sidecars into
a new directory using the canonical schema. This metadata-only conversion can
keep the existing `.pt2` files and `weights.pt` byte for byte; the archive binds
package hashes, not JSON sidecar hashes. Keep the original bundle and binary
when retaining historical evidence. Runtime readers contain no legacy-field
translation paths.

## L20 validation: 2026-11-03

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
