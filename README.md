# xinfer

PyTorch AOTInductor compilation with Python-free C++ inference.

- [Qwen3-0.6B](compiler/qwen3/README.md): export, generation, validation and benchmarks.
- [Qwen3.5](compiler/qwen35/README.md): text-only 0.8B export, native generation and functional benchmarks; HF numerical acceptance remains open.
- [L20 and GB10 environments](environments/README.md): target setup, build/export paths and validation matrix.
- [Development tools](docs/development-tools.md) and [FlashAttention adapter](docs/flash-attention.md).
- [Plan and deferred work](PLAN.md).

`compiler/<model>/` owns HF loading, tensor/state adapters and export settings.
`runtime/<model>/` owns model-specific package contracts and state policies. Shared file,
validation and benchmark helpers live in `compiler/common/`.
`environments/` owns project setup, native-stack checks and target profiles;
its Bash entry points are `setup.sh` and `check.sh`.
`runtime/common/` owns the runner, device and package preflight, positions,
prefill/decode dispatch, greedy generation and fixed-token benchmark loops.
One `xinfer_aoti`, `xinfer_benchmark` and `xinfer_validate` executable supports
both models, selected from package metadata. State policies retain persistent KV
buffers and, for Qwen3.5, replace functional conv/recurrent state. Generation
stops at EOS or the token budget. Benchmark requests
consume all supplied tokens, including EOS, with argmax/host reads timed.
Tests and focused diagnostics live beside their model. Detailed results live
in `docs/experiments/`. Generated packages/results belong in `artifacts/`;
generated venvs, builds and scratch scripts belong in `work/`. Both are ignored.

Run Python tools from the repository root as modules, for example:

```bash
Q3_PY=work/l20/qwen3-torch213-venv/bin/python  # choose the environment for your target
"$Q3_PY" -m compiler.qwen3.prepare --help
"$Q3_PY" -m compiler.qwen3.export --help
"$Q3_PY" -m compiler.qwen3.tests.validate_cpp --help
```

Use a target's native stack and each model's isolated HF environment, following the
[environment guide](environments/README.md). Qwen3 uses one fused FlashAttention
compilation profile and always links its native attention library. Existing efficient Qwen3 packages must be re-exported;
see its guide for the fixed package contract. Build from the repository root
with CMake. Each model retains its own compiler environment and numerical policy.

Both exporters now produce weightless prefill/decode packages and one shared
`weights.pt` archive containing the constant mappings. Both package metadata
files record its checksum. Deploy the complete five-file package directory. The Python and C++ runtimes retain one weight pool and lend its
storage to both loaders with `user_managed`; rebuild with Torch 2.13 or newer
and re-export existing embedded-weight packages before using the current runtime.

Benchmark measurement boundaries are unchanged: reset, prefill, decode and
whole request. Both model workers record reset-plus-prefill and consumed
positions; the native report schema is shared. Warmup
predictions are checked as well as measured repeats. Source snapshots include
the shared implementation. Qwen3 keeps HF eager/FP32 references for validation, with bounded HF logit error
and separate cache/token drift diagnostics. CUDA Graph integration is deferred.
