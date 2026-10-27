# xinfer

PyTorch AOTInductor compilation with Python-free C++ inference.

- [Qwen3-0.6B](compiler/qwen3/README.md): export, generation, validation and benchmarks.
- [Qwen3.5](compiler/qwen35/README.md): the existing single-DeltaNet-block experiment.
- [Development tools](docs/development-tools.md) and [FlashAttention adapter](docs/flash-attention.md).
- [Plan and deferred work](PLAN.md).

`compiler/<model>/` owns HF loading, tensor/state adapters and export settings.
`runtime/<model>/` owns package contracts and state management. Small reusable
file and validation utilities live in `compiler/common/` and `runtime/common/`.
Tests live beside their model. Generated packages/results belong in `artifacts/`;
environments, builds and scratch scripts belong in `work/`. Both are ignored.

Run Python tools from the repository root as modules, for example:

```bash
.venv/bin/python -m compiler.qwen3.prepare verify
.venv/bin/python -m compiler.qwen3.export decode
.venv/bin/python -m compiler.qwen3.tests.validate_cpp --help
```

Use each model's pinned Python environment. Native executable names and package
formats remain unchanged; build from the repository root with CMake.
