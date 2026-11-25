# Working on xinfer

Read [README.md](README.md) and the relevant model guide before changing code.
Use [environments/README.md](environments/README.md) for setup and builds,
[docs/development-tools.md](docs/development-tools.md) for formatting and tests,
and [docs/package-metadata.md](docs/package-metadata.md) for package contracts.
[PLAN.md](PLAN.md) tracks pending work; [validation status](docs/validation-status.md)
summarizes current tested profiles, open findings and gaps (keep detailed run
histories in local artifacts).

## Code ownership

- `compiler/<model>/` owns model loading, adapters, kernels and export.
  `compiler/common/` owns shared compilation, metadata and weight helpers;
  `compiler/common/kernels/` owns shared native kernels and their C shims.
- `runtime/common/` owns shared native loading, state and generation;
  model-specific headers validate their model's ABI.
- `tools/` owns preparation, validation and benchmarking, with native C++
  drivers under `tools/{benchmark,validation}/native/`. Production compiler code
  must not depend on tooling.
- `environments/` owns setup, device/model requirements and hardware profiles.
- `plugins/` owns coding-agent skills that reuse project tools; keep
  model-specific details in model guides.
- Tests follow the code they cover: module/operator/kernel checks belong in unit
  tests; full export-to-native checks belong in `tools/validation/`.
- Keep test-only and tool-only helpers in test/tool modules (and native test
  translation units); never expose them from production code.
- Extend shared implementations and model hooks before adding parallel
  model-specific orchestration. Derive geometry and bounds from inputs or
  metadata where supported, and make actual model/ABI restrictions explicit.
- Keep native tools read-only on caller-supplied inputs, stage multi-model GPU
  validation sequentially (at most one full GPU model resident at a time), and
  refuse occupied output paths including dangling symlinks (`path_exists` /
  `symlink_status` in C++, `exists() or is_symlink()` in Python).

## Environment and builds

Use `environments/<target>/<model>_requirements.txt` as the dependency source
for each environment; keep package pins out of setup scripts and hardware
profiles. Keep shared Python dependency pins aligned across models (updating
Docker lockfiles and `SHA256SUMS` together) unless a verified incompatibility
requires divergence. Create fresh independent venvs with the device's setup
script (use `--venv` for another destination; preserve existing environments).

Run Python tools as modules from the repository root with the matching model
interpreter, and build native code against that interpreter's LibTorch in
`build/<model>/`. Build `xinfer_attention_ops` before export or standalone
attention tests; runtime targets build independently and consume packaged
kernels. Enforce C++20 as the minimum native standard, avoid deprecated
serialization (`torch.jit`) and extra native runtime dependencies (e.g.,
OpenSSL, Python), and use environment checking to verify the stack and build.

## Correctness contracts

Preserve the BF16 `HF → adapted/optimized eager → Python AOTI → C++ AOTI` chain
with identical teacher-forced inputs and an unmodified live HF baseline.
Preserve checkpoint-native parameter dtypes on load (`dtype="auto"`, keeping
native FP32 weights such as `A_log` and `norm.weight`) rather than globally
casting all parameters to BF16. Component tests compare matching input dtypes
(BF16 against BF16 with algorithm-defined FP32 accumulation and recurrent state;
FP32 against FP32); neither full-model nor BF16 kernel validation loads a
separate upcast FP32 control.

Keep native/Python bitwise equality, finite-value, state ownership, masking,
reset/reuse, generation and boundary checks strict. Both models follow the
[shared numerical policy](docs/development-tools.md#shared-numerical-policy):
matched-input logit bounds are required, while full-model state and HF
token/greedy differences are diagnostic; component kernel/block tests retain
their own required output and state limits. After changing export tracing,
dynamic shapes or graph passes, verify from export diagnostics or tests that
expected passes still match (`replaced > 0`) rather than becoming silent no-ops.
Do not loosen tolerances, alter references or suppress failures to make an
optimization pass, and report HF numerical acceptance separately from native
parity.

Preserve shared-weight ownership and metadata-driven state handling. Keep
checkpoint manifests, metadata schemas and logical model identities uniform
across model families without model sizes or separate format/schema version
fields; put build diagnostics in export logs. Metadata or ABI changes must
update shared Python writers/validators, C++ readers, tests and documentation
together; re-export retained packages when contracts change.

## C++ spacing

Use blank lines as paragraphs: one blank line between function definitions and
between distinct logical phases within a function (validation, preparation,
execution, reporting). Keep declarations with their immediate uses and
operations with their error checks; related one-line accessors may stay grouped.
Avoid leading blank lines inside function bodies and consecutive blank lines
(`clang-format` caps excess blank lines but does not group logical phases).

## Verification and measurement

Use the pinned Ruff and clang-format tools with repository configs: verify
`ruff format --check .`, `ruff check --select E,F,W .`, and
`clang-format --dry-run --Werror`. Run CPU unit tests via
`CUDA_VISIBLE_DEVICES="" python -m unittest discover` rather than hand-picked
module lists, and keep preflight and CPU contract checks runnable with
`CUDA_VISIBLE_DEVICES=""`. Follow model guides for affected GPU, export/AOTI
and native checks, and report what ran, what could not run and any failures.

Keep benchmarks separate from accuracy validation. Use the same GPU UUID for
workers, idle checks and telemetry. Establish idle conditions and repeat
measurements before making performance claims; busy-GPU runs demonstrate only
functionality. Use suitable resources rather than altering tracing or model
semantics for a constrained host.

## Repository hygiene

Generated environments and scratch/one-off scripts belong in ignored `work/`,
native builds in `build/`, and packages/run artifacts in `artifacts/`; maintain
only reusable project tooling in Git. Preserve unrelated files and retained
checkpoints/packages, and keep private infrastructure identifiers and access
details out of tracked files.

Give shared commands, contracts and validation status one documented home linked
from model guides, keeping model-specific design and constraints in those
guides. Preserve reusable technical conclusions and dated evidence links when
retiring development journals, and update documentation when interfaces change.

Commit, push or rewrite history only when requested.
