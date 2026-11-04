# Package metadata

Both model families use `schema_version: 3` and `runtime_abi_version: 1` in
`prefill.json` and `decode.json`.
`compiler/common/metadata.py` builds, finalizes and reads the shared Python
format. `runtime/common/spec.h` is the common C++ reader. Model-specific modules
validate their supported tensor ABI and retain their state-update policies.

| Fields | Meaning |
| --- | --- |
| `schema_version`, `runtime_abi_version`, `model_type`, `entry_point`, `token_range` | Metadata format, calling convention, model selection and entry bounds |
| `torch_version`, `cxx11_abi`, `cuda_compute_capability` | Runtime compatibility |
| `checkpoint`, `vocab_size`, `eos_token_ids` | Verified checkpoint identity, vocabulary and an explicit EOS list |
| `cache_length`, `state` | Contiguous state shapes, types, ownership and context capacity |
| `export_options`, `tf32`, `triton_enable_fp_fusion`, `inductor_configs` | Export and numerical contract |
| `package_bytes`, `package_sha256`, `shared_weights` | Package and archive integrity |
| `model_specific` | DeltaNet numerical policy and minimal packing settings; empty for Qwen3 |

Every ordered state descriptor gives its `layer`, `kind`, `shape`, `dtype` and
`update` policy. KV inputs use `inplace`; conv/recurrent inputs use `replace`.
Readers derive state names and contiguous strides. Qwen3 cache geometry and
Qwen3.5 layer types are derived from state rather than duplicated under
`model_specific`.

## Runtime ABI 1

`runtime_abi_version: 1` defines the compiled calling convention:

- Batch size is one and model activations/logits use BF16. Each state descriptor
  retains its own dtype; recurrent state uses FP32.
- Inputs are CUDA `int64` token IDs, CUDA `int64` cache positions, the ordered
  contiguous state tensors, and a CPU `uint8` extent vector. Its length is
  `valid_length + 1`; only its shape is read.
- KV state uses full-capacity buffers. The adapter updates appended K/V positions
  in place. Conv/recurrent inputs remain unchanged and their returned tensors
  replace the previous state.
- Attention uses the read-only valid-prefix Flash operator with attention-library
  ABI 1. Python and C++ query the loaded library's actual ABI before use.
- Qwen3 returns last-token logits only. Qwen3.5 returns last-token logits followed
  by every state tensor in descriptor order; returned K/V tensors alias their
  caller-owned input buffers. Logits have shape `[1, vocab_size]`.

Readers validate state shapes, layer order, dtypes, ownership and request bounds
against this supported ABI. The version does not replace those checks. The
seven former descriptions (`batch_size`, `dtype`, `input_mutation`,
`attention_backend`, `custom_attention`, `cache_input` and `kv_extent`) are no
longer serialized, and schema 3 rejects them as unknown fields.

The metadata schema describes representation; the runtime ABI describes calls
between the compiled graph and the runner. They evolve independently of
`shared_weights.format: 2` and attention-library ABI 1. Changing extent
representation, input/output order or mutation semantics requires a new runtime
ABI. New weights or compiler tuning do not change the calling convention.
Numerical settings, toolchain compatibility and artifact checksums remain
explicit metadata fields.

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

Metadata schema version 3 is independent of the runtime ABI and embedded
weight-archive format. Schema 1, schema 2 and unversioned sidecars are not read
by current runtimes. Re-export with the current exporter, or convert verified
retained sidecars into a new directory using the canonical schema. This
metadata-only conversion can keep compatible `.pt2` files and `weights.pt` byte
for byte; the archive binds package hashes, not JSON sidecar hashes. Keep the
original bundle and binary when retaining historical evidence. Runtime readers
contain no legacy-field translation paths.

Verified schema 2 L20 bundles are converted under
`artifacts/l20/runtime-abi-20261104/migrated/{qwen3,qwen35}`. Conversion checks the
previous fixed descriptions before assigning runtime ABI 1; it retains all
numerical, checkpoint and integrity fields. The original bundles are preserved,
and the converted copies reuse their compiled graphs and weights.

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
