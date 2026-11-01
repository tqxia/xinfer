# xinfer

PyTorch AOTInductor compilation with Python-free C++ inference.

- [Qwen3-0.6B](compiler/qwen3/README.md): export, generation, validation and benchmarks.
- [Qwen3.5](compiler/qwen35/README.md): text-only 0.8B export, native generation and functional benchmarks; HF numerical acceptance remains open.
- [L20 and GB10 environments](environments/README.md): target setup, build/export paths and validation matrix.
- [Development tools](docs/development-tools.md) and [FlashAttention adapter](docs/flash-attention.md).
- [Plan and deferred work](PLAN.md).

`compiler/<model>/` owns HF loading, tensor/state adapters and export settings.
`runtime/<model>/` owns package contracts and state management. Shared file,
validation and benchmark helpers live in `compiler/common/`.
`environments/` owns project setup, native-stack checks and target profiles;
its Bash entry points are `setup.sh` and `check.sh`.
`runtime/common/` provides file helpers, greedy generation and fixed-token
benchmark loops. Each runtime exposes `reset()`, `step(tokens)` and `position()`;
it retains its own state ownership and dispatch. Generation validates through
the model runtime, then stops at EOS or the token budget. Benchmark requests
consume all supplied tokens, including EOS, with argmax/host reads timed.
Tests and focused diagnostics live beside their model. Detailed results live
in `docs/experiments/`. Generated packages/results belong in `artifacts/`;
generated venvs, builds and scratch scripts belong in `work/`. Both are ignored.

Run Python tools from the repository root as modules, for example:

```bash
Q3_PY=work/l20/qwen3-venv/bin/python  # choose the environment for your target
"$Q3_PY" -m compiler.qwen3.prepare --help
"$Q3_PY" -m compiler.qwen3.export --help
"$Q3_PY" -m compiler.qwen3.tests.validate_cpp --help
```

Use a target's native stack and each model's isolated HF environment, following the
[environment guide](environments/README.md). Qwen3 uses one fused FlashAttention
compilation profile and always links its native attention library. Existing efficient Qwen3 packages must be re-exported;
see its guide for the fixed package contract. Build from the repository root
with CMake. Qwen3.5 retains its separate configuration and validation policy.

Benchmark measurement boundaries are unchanged: reset, prefill, decode and
whole request. Both model workers record reset-plus-prefill and consumed
positions; report headers/profile comparisons remain model-specific. Warmup
predictions are checked as well as measured repeats. Source snapshots include
the shared implementation. Qwen3 keeps HF eager/FP32 references for validation, with bounded HF logit error
and separate cache/token drift diagnostics. CUDA Graph integration is deferred. Profiling capture/classification refactoring remains
deferred in PLAN.md.
