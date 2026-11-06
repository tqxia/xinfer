# xinfer

PyTorch AOTInductor compilation with Python-free C++ inference.

- [Qwen3-0.6B](compiler/qwen3/README.md): export, generation, validation and benchmarks.
- [Qwen3.5](compiler/qwen35/README.md): text-only 0.8B export, native generation and functional benchmarks; HF numerical acceptance remains open.
- [L20 and GB10 environments](environments/README.md): target setup, build/export paths and validation matrix.
- [Development tools](docs/development-tools.md) and [FlashAttention adapter](docs/flash-attention.md).
- [Plan and deferred work](PLAN.md).

`compiler/<model>/` owns HF loading, tensor/state adapters and export settings.
`runtime/<model>/` owns model-specific package ABI checks. Shared compiler
artifact, metadata and weight helpers live in `compiler/common/`. Numerical validation
and its model hooks live in `tools/validation/`; benchmark orchestration, measurement
and model hooks live in `tools/benchmark/`. Checkpoint download, verification and
offline tokenization use the shared `tools/prepare/` command.
`tools/common.py` supplies logged subprocess execution and native-library
discovery; focused numerical diagnostics live under `tools/diagnostics/`.
`environments/` owns project setup, native-stack checks and target profiles;
its Bash entry points are `setup.sh` and `check.sh`.
`runtime/common/` owns the runner, state storage, device and package preflight,
positions, prefill/decode dispatch, greedy generation and fixed-token benchmark loops.
One `xinfer_aoti`, `xinfer_benchmark` and `xinfer_validate` executable supports
both models, selected from package metadata. One state policy retains persistent
KV buffers and replaces functional conv/recurrent state using metadata update
rules and the runtime ABI's output order. Generation stops at EOS or the token
budget. Benchmark requests
consume all supplied tokens, including EOS, with argmax/host reads timed.
Unit tests follow code ownership: tooling tests live under `tools/<area>/tests/`,
while compiler/model/operator tests remain beside their code.
See the [CPU unit-test commands](docs/development-tools.md#cpu-unit-tests).
The common numerical CLI is `tools.validation.validate`; model-specific
CLI/deployment contract tools remain under `tools/validation/{qwen3,qwen35}/`. Detailed results live
in `docs/experiments/`. Generated packages/results belong in `artifacts/`;
generated native builds belong in `build/<model>/`; venvs and scratch scripts
belong in `work/`. These directories are ignored.

Use the pinned Ruff and clang-format tools with the repository settings; see
[formatting commands](docs/development-tools.md#formatting).

Both exporters and runtimes use the [versioned package metadata format](docs/package-metadata.md).
Model-specific extras live under `model_specific`; common fields have identical
names and validation for both families. Schema 3 records the calling convention
as `runtime_abi_version: 1`. Retained schema 1, schema 2 and unversioned sidecars
need conversion or re-export before use with the current runtime.

Run Python tools from the repository root as modules, for example:

```bash
Q3_PY=work/l20/qwen3-torch213-venv/bin/python  # choose the environment for your target
"$Q3_PY" -m tools.prepare --help
"$Q3_PY" -m compiler.qwen3.export --help
"$Q3_PY" -m tools.validation.validate --help
"$Q3_PY" -m tools.benchmark --help
```

Use a target's native stack and each model's isolated HF environment, following the
[environment guide](environments/README.md). Qwen3 uses one fused FlashAttention
compilation profile and always links its native attention library. Existing efficient Qwen3 packages must be re-exported;
see its guide for the fixed package contract. Build from the repository root
with CMake. Each model retains its own compiler environment and numerical policy.
The [runtime installation guide](environments/README.md#install-and-relocate-the-runtime)
shows how to install and move the native runtime prefix with its attention library.
Python tools default to native executables and the attention library in the
repository's `build/qwen3/` or `build/qwen35/` directory for their model.
Existing executable-path options and
`XINFER_ATTENTION_OPS_LIBRARY` remain optional overrides for another build or an
installed runtime.

Numerical validation follows HF → eager → Python AOTI → C++ AOTI. The eager
stage includes adapters and any linear packing selected by package metadata.
Package metadata selects the model hooks. The shared runner uses identical
teacher-forced inputs, checks adjacent stages and retains a direct
Python-AOTI/HF comparison. Both models generate HF references during validation;
Qwen3 also cross-checks manual HF greedy decoding against HF `generate()`. No
saved-reference preparation is required. CLI/deployment contract checks remain
separate. See the model guides for numerical policies and commands.

CPU portability remains a known limitation: AOTI package wrappers inherit the build
host's instruction set (including AVX-512 on L20), and the project does not enforce
CPU compatibility. Use a compatible CPU or re-export on the destination host.

Both exporters now produce weightless prefill/decode packages and one shared
`weights.pt` archive containing the constant mappings. Both package metadata
files record its checksum. Deploy the complete five-file package directory. The Python and C++ runtimes retain one weight pool and lend its
storage to both loaders with `user_managed`; rebuild with Torch 2.13 or newer
and re-export existing embedded-weight packages before using the current runtime.

`python -m tools.benchmark` selects the model from package metadata. Both models
support `run`, paired package `compare` and standalone `check-idle`. Workloads are
tokenized by the benchmark itself, with prompt/context limits read from metadata.
Benchmarks split reset, prefill, decode and whole-request timing. Both model
eager workers use the unchanged HF model and its dynamic cache, with automatic
masks and positions; reset discards the cache and initial allocation is timed
in prefill. Reports identify the baseline as `eager_baseline: hf_dynamic_cache`.
Earlier masked-cache speedups require fresh comparisons. Both model workers
record reset-plus-prefill and consumed positions. The Python benchmark
controller shares environment capture, rotated process order, failure handling
and summaries across models. Warmup predictions
are checked as well as measured repeats. Benchmark reports include package,
binary and workload checksums. Benchmark reports contain performance and harness
results; numerical accuracy reports remain in the validation tooling. Qwen3 keeps HF eager/FP32 references for
validation, with bounded HF logit error and separate cache/token drift
diagnostics. CUDA Graph integration is deferred.
