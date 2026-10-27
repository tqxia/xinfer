# Experimental valid-prefix FlashAttention

The optional `--attention-backend flash` profile targets batch-one BF16
Qwen3-0.6B. It retains eight KV heads and attends only to the populated cache
prefix. The default remains `efficient`; the existing eager-numerics packages
remain the accepted correctness baseline.

## Compilation and runtime contract

1. Load the original HF weights, then install `FlashAttention` adapters on the
   decoder layers' `self_attn` children.
2. `FlashStep` drives the original HF decoder layers with explicit cache state.
   Each attention adapter calls `xinfer::flash_attention` directly. The exported
   graph contains 28 custom attention calls and all 56 cache mutations.
3. Validate the exported graph, then compile it with AOTInductor.
4. `runtime/qwen3/custom_ops/attention.cpp` implements the read-only operator with
   PyTorch's native FlashAttention and lower-right causality. The same kernel
   handles initial prefill, appended chunks and single-token decode.

The custom operator uses fixed `[1,8,1024,128]` unpadded cache tensors and returns
fresh contiguous output. A CPU uint8 tensor with shape `[valid_length + 1]` passes
the valid prefix length through shape metadata; its data is ignored. Encoding
the length this way avoids an input size-one specialization. Runtime prefix
views stay inside the opaque operator, so export keeps `backed_size_oblivious`
disabled. The operator does not update or alias cache inputs.

The native flat input list gains the extent tensor after the 56 cache tensors.
Package metadata records this contract and the operator ABI. The C++ runtime
checks compatible prefill/decode settings before loading weights. Only the
canonical `flash` label and unpadded layout are supported; experimental legacy
packages must be re-exported.

## Module adapters

`--attention-backend flash` always introduces the native operator directly from
Python, before export. The previous FX pass and implementation selector have
been removed. This does not use HF's attention-backend registry or forward hooks.

`compiler/qwen3/adapters/qwen3_flash.py` contains three small pieces:

- `PrefixCache` adds the extent tensor to the existing caller-owned cache adapter.
- `FlashAttention` wraps a loaded HF attention module, reusing its projections,
  Q/K normalization, RoPE helper and output projection. Its forward updates the
  cache, then explicitly calls `torch.ops.xinfer.flash_attention` with native GQA.
- `FlashStep` installs these wrappers as each decoder layer's `self_attn` child.
  A tensor-only loop calls the original HF embedding, rotary embedding, decoder
  layers, final norm and LM head. This bypasses HF's automatic mask construction;
  the custom op implements causality for the populated prefix.

The supported inputs are batch-one BF16, unpadded contiguous cache appends, with
the same position/extent contract enforced by the C++ runtime. The attention
adapter rejects arbitrary masks, other cache types, training and attention-weight
requests. It is not a drop-in replacement for HF training or `generate()`.
Installation reuses the original weights without copying them and happens once,
after checkpoint loading and before export. Wrapped parameter names gain an
`original` component; load checkpoints before wrapping.

Adapter packages retain the earlier Flash packages' native input ABI and custom
op library. The C++ runtime needs no adapter-specific execution path. Metadata
records `flash_implementation: adapter` and `model_adapter`; no SDPA rewrite is
performed.

```bash
.venv/bin/python -m compiler.qwen3.tests.attention_adapter --output artifacts/adapter-unit-new
.venv/bin/python -m compiler.qwen3.export decode --attention-backend flash \
  --output artifacts/aoti-flash-adapter-new
.venv/bin/python -m compiler.qwen3.export prefill --attention-backend flash \
  --output artifacts/aoti-flash-adapter-new
AOTI_RUNTIME_CHECK_INPUTS=1 .venv/bin/python -m compiler.qwen3.tests.validate_cpp \
  --packages artifacts/aoti-flash-adapter-new --native build/qwen_validate \
  --collect-logit-errors --collect-top1-errors --output artifacts/adapter-validation-new
```

Use the Flash-enabled build and environment below. The focused adapter test
compares the real first attention block against unmodified HF BF16 with the
existing 5% limits, and checks eager/exported adapter outputs bitwise. It also
checks exact cache contents, write regions, storage, future-slot poisoning,
rejection behavior and weight/module reuse. `custom_attention.py` independently
checks the native operator against FP32 and compiles direct custom-op calls.
Full-model numerical acceptance is a separate check; changing how the operator
enters the graph does not resolve the Flash profile's numerical limitations.

### Adapter results — L20, Torch 2.7.0+cu126

Both entry points export and compile successfully without an FX rewrite. The
focused test passes nine shape/length cases and five rejection cases. The
adapter reuses all original parameter objects and HF decoder layers.

After retiring the pass, the independent HF-block test passes with maximum
relative L2 error 0.289% and peak-relative error 0.747%. Nine direct operator
cases, nine compiled cache-update cases and three invalid-length cases pass.
Compiling with only `--attention-backend flash` reproduces the previously
validated adapter's 185 logit rows and all 56 cache tensors bitwise. A two-round
paired benchmark smoke test also passes and snapshots the adapter sources.

Before removing the graph pass, a comparison using its cached Inductor tuning
choices established equivalence of the two compilation approaches. Full-model
validation covers 26 cases and 185 logit rows. Python AOTI and C++ agree bitwise.
All 26 native logit files and all 56 full-cache tensors also agree bitwise with
the graph-pass artifacts; the complete validation summaries match. This includes
prefill/decode handoff, appended chunks, poisoned future slots, full context,
stable storage and repeated requests.
The native CLI also passes three generation fixtures, prompt lengths 1/2/512,
20 rejection cases and Python-free linkage checks.

HF acceptance still fails with the existing **14/56 cache-tensor failures and
two top-token mismatches**. The collector exits nonzero as intended. This is an
equivalent way to build the experimental Flash profile, not a precision fix.

An independently tuned build produced 20 cache-tensor failures and the same two
top-token mismatches. Its generated Triton reduction source matched the baseline,
but autotuning chose different `XBLOCK`/warp configurations for fused Q/K
normalization and RoPE kernels. Reusing the baseline cache removed the additional
drift. Preserving the reference's contiguous-Q boundary alone did not remove it.
For comparisons between builds, preserve `TORCHINDUCTOR_CACHE_DIR` and the
toolchain, and validate each resulting artifact. A fresh environment may choose
different configurations; shared tuning does not imply cross-GPU reproducibility.

Recorded artifacts in the primary checkout:

- `artifacts/attention-adapter-default/` and `artifacts/attention-op-default/`:
  independent checks after removing the pass.
- `artifacts/aoti-flash-default/` and `artifacts/validate-flash-default/`:
  simplified-CLI packages and validation, including
  `default-vs-explicit-adapter.json` with exact comparisons.
- `artifacts/compare-flash-default/`: benchmark/source-snapshot smoke check.
- `artifacts/attention-adapter-unit-v3/`: focused adapter checks.
- `artifacts/aoti-flash-adapter-matched/`: packages using the baseline tuning cache.
- `artifacts/validate-flash-adapter-matched/`: full validation, FP32 controls and
  `adapter-vs-graph-pass.json` with exact file comparisons.
- `artifacts/validate-flash-adapter-v2/`: independently tuned numerical results.
- `artifacts/smoke-flash-adapter-matched/`: native CLI and deployment checks.
- `work/adapter-diagnosis/tuning-differences.json`: differing kernel configurations.

The baseline packages and tuning cache are in the retained integration worktree,
under `artifacts/aoti-flash/` and `work/inductor-flash/`. The development GPU remains
shared; these checks establish functionality and numerical behavior, not speed.

## Build, export and validate

Configure the matching LibTorch/CUDA toolchain as described in README.md. Add
these options to that build, or use a separate build directory with the same
toolchain configuration:

```bash
cmake -S . -B build -DQWEN_ENABLE_FLASH_ATTENTION=ON \
  -DQWEN_BUILD_VALIDATION=ON -DQWEN_BUILD_BENCHMARK=ON
cmake --build build -j2
export XINFER_ATTENTION_OPS_LIBRARY="$PWD/build/libxinfer_attention_ops.so"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_COMPILE_THREADS=2

.venv/bin/python -m compiler.qwen3.tests.custom_attention --output artifacts/attention-unit-new
.venv/bin/python -m compiler.qwen3.export decode --attention-backend flash --output artifacts/aoti-flash-new
.venv/bin/python -m compiler.qwen3.export prefill --attention-backend flash --output artifacts/aoti-flash-new
AOTI_RUNTIME_CHECK_INPUTS=1 .venv/bin/python -m compiler.qwen3.tests.validate_cpp \
  --packages artifacts/aoti-flash-new --native build/qwen_validate \
  --collect-logit-errors --collect-top1-errors --output artifacts/flash-validation-new
```

The library defaults to `build/libxinfer_attention_ops.so` for Python export
and isolated operator tests. Set `XINFER_ATTENTION_OPS_LIBRARY` for another build
directory. Native validation resolves the library actually linked by `--native`
under the same sanitized environment used for its C++ subprocess. The benchmark
records that resolved library and its hash without loading it into Python.

`QWEN_ENABLE_FLASH_ATTENTION` defaults to OFF. An ordinary efficient-only build
has no attention-library dependency and rejects Flash packages before CUDA/model
loading. Deploy the shared operator library with Flash-enabled executables and
matching LibTorch; the `.pt2` packages do not bundle this external operator.

## Numerical status and evidence

The GB10 source branch (commit `807289d`) validates nine isolated operator cases
against FP32 attention, nine compiled cases, cache mutation/order/storage,
poisoned future slots, unusual strides and lower-right causality. Full-model
validation covers 26 cases and 185 logit rows. Python AOTI and C++ agree bitwise,
and all tested top-token predictions agree with HF BF16. However, **17/56 cache
tensors fail the unchanged 5% numerical limits**. The full validator intentionally
exits nonzero; this experimental profile is not an accepted replacement for the
efficient eager-numerics baseline. Enabling `--eager-numerics` does not promise
bitwise agreement across different attention backends.

The recorded GB10 experiments show roughly 1.27–1.32x decode and 1.38–1.49x prefill
speedups over fused efficient attention; later canonical custom-op measurements
retain approximately those timings. These are GB10 results with Torch 2.13/CUDA
13, not a performance claim for other toolchains. Complete historical evidence
remains in `dev-gb10` under `benchmarks/gb10-flash-2026-10-24/`,
`benchmarks/gb10-custom-attention-unpadded-2026-10-25/` and
`benchmarks/gb10-flash-canonical-2026-10-25/`.

Use `compiler/qwen3/benchmark.py compare` after correctness validation to measure a new
environment. CUDA Graphs, FlashInfer and projection packing are not part of this
profile. See [development tools](development-tools.md) for comparison/profiling.

## Historical graph-pass integration checks — L20, Torch 2.7.0+cu126

The integration was rebuilt and exercised on the original development host:

- Flash-enabled and Flash-disabled C++ builds succeed. The disabled build has no
  operator-library dependency and rejects Flash packages before CUDA loading.
- Nine direct and nine compiled attention cases pass, including full-range HF
  views retained by Torch 2.7. Nine rejection cases cover invalid lengths and
  unknown/cropped/strided masks. Each full-model graph retains 56 cache writes
  and replaces all 28 attention sites.
- Full-model Flash completes 26 cases / 185 rows with bitwise C++/Python AOTI
  logits and full caches, masking/storage/reset checks and independent FP32
  controls. Three native generation fixtures, prompt lengths 1/2/512 and 20
  input/metadata rejection cases pass, including Python-free native linkage.
- HF numerical acceptance **fails**: 14/56 cache tensors exceed tolerance, and
  183/185 top-token predictions agree. The mismatch is row 1 of
  `prefix_boundaries` and its poisoned repeat: HF chooses 12555, Flash chooses
  99692, with the selected token 0.375 below HF's maximum. Maximum logit relative
  L2 is 3.8492%, within 5%. This differs from the saved GB10 result and remains
  an experimental numerical limitation; no threshold was changed.
- The existing efficient eager-numerics profile still passes all 24 cases / 163
  rows and all 56 full-cache tensors bitwise against HF, using the Flash-disabled
  runtime.

Evidence stays in ignored `artifacts/attention-unit-hf-views/`,
`artifacts/validate-flash/`, `artifacts/smoke-flash/` and
`artifacts/validate-efficient/` in the integration worktree. The development GPU
is shared, so benchmark smoke timings do not establish a performance result.
