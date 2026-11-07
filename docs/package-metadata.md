# Package metadata

Both model families use `schema_version: 5` and `runtime_abi_version: 1` in
`prefill.json` and `decode.json`.
`compiler/common/metadata.py` builds, finalizes and reads the shared Python
format. `runtime/common/spec.h` is the common C++ reader. Model-specific modules
validate their supported tensor ABI; `runtime/common/state.h` handles state for
both models.

| Fields | Meaning |
| --- | --- |
| `schema_version`, `runtime_abi_version`, `model_type`, `entry_point`, `token_range` | Metadata format, calling convention, model selection and entry bounds |
| `torch_version`, `cxx11_abi`, `cuda_compute_capability` | Runtime compatibility |
| `checkpoint`, `vocab_size`, `eos_token_ids` | Verified checkpoint identity, vocabulary and an explicit EOS list |
| `cache_length`, `state` | Compiled context capacity and contiguous state shapes/types/ownership |
| `export_options`, `tf32`, `triton_enable_fp_fusion`, `inductor_configs` | Export and numerical contract |
| `package_bytes`, `package_sha256`, `shared_weights` | Package and archive integrity |
| `model_specific` | DeltaNet numerical policy and minimal packing settings; empty for Qwen3 |

`cache_length` is chosen at export with `--max-context-length`. Its default is
the selected model's declared text-model context ceiling, not a fixed 16K value.
The exporter checks the selected checkpoint's text-config
`max_position_embeddings` before allocating model state. Both supported model
families use their declared default-RoPE ceiling; no scaling factor is inferred.
The compiled capacity must be at least 512 and cannot exceed that model ceiling.
KV state descriptors contain exactly `cache_length` slots.

`token_range` describes one compiled invocation: decode accepts `[1,1]` and
prefill `[2,512]`, independently of the complete prompt length. The runtime splits
a prompt into calls of at most 512 tokens and uses decode for a one-token
remainder. The complete prompt plus requested generation budget must fit the
context before execution begins; an oversized budget is rejected, not clamped.
Changing the selected capacity requires re-export of both entry points.

There is no separate prompt cap in the runtime or package. The native generation
CLI accepts `--max-output-tokens` (default 64), and both prompt and output must
be nonempty and fit their combined context budget. The retired
`--max-prompt-length` and `--max-new-tokens` options are rejected; Hugging Face
library calls retain their upstream `max_new_tokens` keyword. Schema 5 also
rejects the former `max_prompt_length` package field. Runtime requests cannot
enlarge a package's state or symbolic bounds.

Every ordered state descriptor gives its `layer`, `kind`, `shape`, `dtype` and
`update` policy. KV inputs use `inplace`; conv/recurrent inputs use `replace`.
Readers derive state names and contiguous strides. Qwen3 cache geometry and
Qwen3.5 layer types are derived from state rather than duplicated under
`model_specific`.

The C++ reader also derives each state's optional output position from
`model_type` and runtime ABI 1. This is an internal descriptor, not an additional
metadata field. The common state policy uses those positions and update rules
to validate outputs and retain or replace storage. Reset retains KV allocations,
fills them with the requested value, and initializes replacement state to zero.

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
longer serialized, and schema 5 rejects them as unknown fields.

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

Metadata schema version 5 is independent of the runtime ABI and embedded
weight-archive format. Schema 1 through schema 4 and unversioned sidecars are not read
by current runtimes. Re-export with the current exporter, or convert verified
retained sidecars into a new directory using the canonical schema. This
metadata-only conversion can keep compatible `.pt2` files and `weights.pt` byte
for byte; the archive binds package hashes, not JSON sidecar hashes. Keep the
original bundle and binary when retaining historical evidence. Runtime readers
contain no legacy-field translation paths.

For a verified schema 3 or schema 4 bundle, preserve the original directory and
create a separate copy for conversion. Set `schema_version` to 5 in **both**
sidecars and remove `max_prompt_length` if present. Retain the original capacity
(`cache_length: 1024` for the older profile), the
compiled token ranges, state shapes, runtime ABI and every numerical/integrity
field unchanged. The converted bundle can reuse its immutable `.pt2` files and
`weights.pt` without recompilation. Validate the new pair with the current
reader before use. Earlier prompt-admission caps are no longer enforced by the
runtime; a calling application may enforce its own input policy. Conversion does
not promote the cache to a larger profile.

Removing the request-policy field changes metadata representation only. Larger KV buffers
and prompt chunking preserve the input/output order, extent representation,
state ownership and attention semantics of runtime ABI 1 and attention-library
ABI 1. Rebuild the native attention library and runner for generalized capacity
support before using newly exported long-context packages.

Verified schema 2 L20 bundles are converted under
`artifacts/l20/runtime-abi-20261104/migrated/{qwen3,qwen35}`. Conversion checks the
previous fixed descriptions before assigning runtime ABI 1; it retains all
numerical, checkpoint and integrity fields. The original bundles are preserved,
and the converted copies reuse their compiled graphs and weights. Those retained
sidecars predate schema 5; make a separate schema 5 copy before using them with
the current reader.

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
