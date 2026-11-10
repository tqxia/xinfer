# Working on xinfer

Read [README.md](README.md) and the relevant model guide before changing code.
Use [environments/README.md](environments/README.md) for setup and builds,
[docs/development-tools.md](docs/development-tools.md) for formatting and tests,
and [docs/package-metadata.md](docs/package-metadata.md) for package contracts.
[PLAN.md](PLAN.md) describes planned work. Files under `docs/experiments/` preserve
historical evidence; their commands and results may describe retired code.

## Code ownership

- `compiler/<model>/` owns model loading, adapters, kernels and export.
  `compiler/common/` owns shared compilation, metadata and weight helpers;
  `compiler/common/kernels/` owns shared native kernels and their C shims.
- `runtime/common/` owns shared native loading, state, generation and measurement.
  Model-specific runtime headers validate their model's ABI.
- `tools/` owns preparation, numerical validation, benchmarking and diagnostics.
  Production compiler code must not depend on tooling.
- `environments/` owns setup, device/model requirements and hardware profiles.
- Tests follow the code they cover. Keep module/operator/kernel checks in unit
  tests and full export-to-native accuracy checks in `tools/validation/`.
- Functions and helpers used only by tests or tooling belong in their respective
  test or tool modules. Production code must not expose test-only or tool-only
  APIs. Put shared test helpers in test modules and test-only native helpers in
  test translation units.
- Extend shared implementations and model hooks before introducing parallel
  model-specific orchestration. Derive geometry and bounds from inputs or
  metadata where supported; make actual model/ABI restrictions explicit.

## Environment and builds

Use `environments/<target>/<model>_requirements.txt` as the dependency source for
that environment. Keep package pins out of setup scripts and hardware profiles.
Create fresh independent venvs with the device's Bash setup script; preserve
existing environments and use `--venv` to choose another destination.

Run Python tools as modules from the repository root with the matching model
interpreter. Build native code against that interpreter's LibTorch in
`build/<model>/`. Build the compiler's `xinfer_attention_ops` target before
model export or standalone attention tests. Runtime targets build independently
and consume the kernels carried by compiled packages. Use environment checking
to verify the selected stack and configured build.

## Correctness contracts

Preserve BF16 HF → adapted/optimized eager → Python AOTI → C++ AOTI validation,
with identical teacher-forced inputs and an unmodified BF16 HF baseline. Generate
HF references during validation. Full-model validation does not load a separate
FP32 HF control model; standalone kernel FP32 checks remain required. Keep
native/Python bitwise equality, finite-value, state ownership, masking, reset/reuse,
generation and boundary checks strict.

Do not loosen tolerances, change references or suppress failures to make an
optimization pass. Report HF numerical acceptance separately from native parity;
passing native checks does not resolve existing HF failures.

Preserve shared-weight ownership and metadata-driven state handling. Metadata or
ABI changes must update the shared Python writer/validators, C++ readers, tests
and documentation together. Follow the documented migration/re-export contract
for retained packages. Keep metadata focused on runtime and artifact contracts; put build
diagnostics in export logs.

## Verification and measurement

Use the pinned Ruff and clang-format tools with the repository configurations.
Run relevant CPU tests first in the correct model environment. Follow the model
guides for GPU, export/AOTI and native checks affected by a change. Report what
ran, what could not run and any remaining failures.

Keep benchmarks separate from accuracy validation. Use the same GPU UUID for
workers, idle checks and telemetry. Establish idle conditions and repeat
measurements before making performance claims; busy-GPU runs demonstrate only
functionality. Use suitable resources rather than changing tracing or model
semantics to accommodate a constrained test host.

## Repository hygiene

Generated environments and scratch scripts belong in ignored `work/`, native
builds in `build/`, and packages/results in `artifacts/`. Preserve unrelated files
and retained checkpoints/packages. Keep private infrastructure identifiers and
access details out of tracked code and documentation.

Generate one-off profiling and experiment scripts in scratch space; maintain
only reusable project tooling. Update current usage documentation when interfaces
change, while preserving clearly marked historical records.

Commit, push or rewrite history only when requested.
