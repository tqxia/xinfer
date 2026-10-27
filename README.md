# xinfer

PyTorch AOTInductor compilation with Python-free C++ inference.

- [Qwen3-0.6B](compiler/qwen3/README.md): export, generation, validation and benchmarks.
- [Qwen3.5](compiler/qwen35/README.md): text-only 0.8B export, native generation and functional benchmarks; HF numerical acceptance remains open.
- [Development tools](docs/development-tools.md) and [FlashAttention adapter](docs/flash-attention.md).
- [Plan and deferred work](PLAN.md).

`compiler/<model>/` owns HF loading, tensor/state adapters and export settings.
`runtime/<model>/` owns package contracts and state management. Shared file,
validation and benchmark helpers live in `compiler/common/`.
`runtime/common/` provides file helpers, greedy generation and fixed-token
benchmark loops. Each runtime exposes `reset()`, `step(tokens)` and `position()`;
it retains its own state ownership and dispatch. Generation validates through
the model runtime, then stops at EOS or the token budget. Benchmark requests
consume all supplied tokens, including EOS, with argmax/host reads timed.
Tests and focused diagnostics live beside their model. Detailed results live
in `docs/experiments/`. Generated packages/results belong in `artifacts/`;
environments, builds and scratch scripts belong in `work/`. Both are ignored.

Run Python tools from the repository root as modules, for example:

```bash
.venv/bin/python -m compiler.qwen3.prepare verify
.venv/bin/python -m compiler.qwen3.export decode
.venv/bin/python -m compiler.qwen3.tests.validate_cpp --help
```

Use each model's pinned Python environment. Native executable names and package
formats remain unchanged; build from the repository root with CMake.

Benchmark measurement boundaries are unchanged: reset, prefill, decode and
whole request. Both model workers now also record reset-plus-prefill and consumed
positions; report headers/profile comparisons remain model-specific. Warmup
predictions are checked as well as measured repeats. Source snapshots include
the shared implementation. No CUDA Graphs, compiler or arithmetic changes are
part of this refactor. Profiling capture/classification refactoring remains
deferred in PLAN.md.
