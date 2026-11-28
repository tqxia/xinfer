# xinfer

Compile Hugging Face Qwen models with PyTorch AOTInductor and run them natively
in Rust (with a minimal static C++ LibTorch shim) without Python. xinfer
provides BF16 and block-scaled FP8 inference, paired prefill/decode packages,
shared weights, explicit in-place model state, and native continuous-batching
serving. `xinfer_aoti` runs batch-one greedy generation; `xinfer_serve` and
explicit batch steps support dynamic batches of 1–8 with independent request
lengths from the same bundle.

Model frontends cover [Qwen3-0.6B / 1.7B](compiler/qwen3/README.md),
[Qwen3.5-0.8B / 4B](compiler/qwen35/README.md) and
[Qwen3.8-27B-FP8](compiler/qwen38/README.md) (text only). Target setup and
builds are documented for [L20, PRO 5000 and GB10](environments/README.md).
Numerical acceptance depends on the model, package and workload; see
[validation and limitations](#validation-and-limitations).

## Compiler architecture

Python prepares the checkpoint, adapts the model to an explicit tensor/state
interface, runs model-specific graph passes, and compiles both prefill and
decode entry points with AOTInductor in one invocation. Shared helpers collect
tensor constants, embed native operator libraries, bundle tokenizer files, and
write `metadata.json`.

![Compiler architecture: model adapters, graph optimization and AOTInductor produce one deployment bundle.](docs/diagrams/compiler-architecture.svg)

All model families use native batched FlashAttention over the populated KV
prefix (`libxinfer_attention_ops.so`). Qwen3.5 and Qwen3.8 also use the native
C++/CUDA DeltaNet operator (`libxinfer_deltanet_ops.so`), linear packing, and
packaged Triton decode GEMV kernels for profiled BF16 geometries. Build
`xinfer_attention_ops` (and `xinfer_deltanet_ops` for Qwen3.5/Qwen3.8) before
export; both libraries are embedded inside each `.pt2` and loaded automatically
by AOTI. See the [attention guide](docs/flash-attention.md).

Each deployment bundle contains `prefill.pt2`, `decode.pt2`,
`weights.safetensors`, `metadata.json`, and bundled tokenizer files
(`tokenizer.json`, `tokenizer_config.json`, optional `chat_template.jinja`).
Graph dumps and build logs are export diagnostics. Deploy the complete bundle
with matching LibTorch/CUDA; see [package metadata](docs/package-metadata.md)
and [runtime installation](environments/README.md#install-and-relocate-the-runtime).

## Runtime architecture

The same native Rust executables serve all model families, selecting model ABI
checks from `metadata.json`. `runtime/core/` (`xinfer-runtime` + minimal static
C++ LibTorch shim in `runtime/core/shim/`) owns loading, state, validation,
prompt chunking and prefill/decode dispatch. `runtime/engine/` (`xinfer-engine`)
layers continuous batching, GPU-assisted sampling and native tokenization/chat
templates on top, and `runtime/server/` provides `xinfer_serve`.

![Runtime architecture: a shared native Rust runner executes prefill and decode models using shared weights and model state.](docs/diagrams/runtime-architecture.svg)

Both loaders borrow one weight pool via `user_managed` loading, outlived by
`Model`'s weight owner. All state tensors (KV buffers and Qwen3.5/Qwen3.8
convolution and recurrent state) update in place on caller-owned buffers.

| Executable | Purpose |
| --- | --- |
| `xinfer_aoti` | Batch-one greedy generation from token IDs, stopping at EOS or `--max-output-tokens`. |
| `xinfer_serve` | OpenAI-compatible HTTP/SSE server with continuous batching (`B=1..8`), interleaved chunked prefill, and native tokenization. |
| `xinfer_benchmark` | Fixed-token prefill/decode timing and memory measurements. |
| `xinfer_validate` | Logit/state traces and runtime correctness checks. |

Native execution requires LibTorch/CUDA but loads neither Python nor
`libtorch_python` (nor OpenSSL). `tools.prepare` handles checkpoint preparation
and offline tokenization for `xinfer_aoti`; `xinfer_serve` tokenizes natively
from the bundle.

## Getting started

1. Follow the [environment guide](environments/README.md) to create a target
   venv (`environments/setup.sh`) and build compiler kernels and native binaries
   (`environments/build.sh`).
2. Follow the [Qwen3](compiler/qwen3/README.md),
   [Qwen3.5](compiler/qwen35/README.md) or [Qwen3.8](compiler/qwen38/README.md)
   guide to prepare a checkpoint, export a bundle, and run `xinfer_aoti` or
   `xinfer_serve`. Use fresh output paths.
3. Run [numerical and runtime validation](docs/development-tools.md#python-validation-tools)
   on each new bundle, then measure it with
   [benchmark tools](docs/development-tools.md#benchmark-hf-and-native-aoti).

Run Python commands as modules from the repository root with the target venv
(e.g., `PY=work/l20/venv/bin/python`):

```bash
"$PY" -m tools.prepare --help
"$PY" -m compiler.qwen3.export --help
"$PY" -m tools.validation.validate --help
"$PY" -m tools.benchmark --help
```

`--max-context-length` sets allocated per-row context capacity at export time
(defaulting to the model config ceiling; changing it requires re-export).
Compiled prefill accepts **2–512 tokens per call**; longer prompts are chunked
automatically with decode handling a 1-token remainder. Prompt plus output
budget must fit the bundle's capacity.

## Validation and limitations

Accuracy validation follows **HF → adapted/optimized eager → Python AOTI →
native AOTI (`cpp_aoti`)** on identical teacher-forced inputs. Native/Python
AOTI parity must be bitwise; HF logit comparisons follow the
[shared numerical policy](docs/development-tools.md#shared-numerical-policy).
Benchmarks run separately via `tools.benchmark` (`run`, `compare`, `check-idle`)
against unchanged HF with a dynamic cache on an idle GPU.

- **Numerics:** see [validation status](docs/validation-status.md) for tested
  targets, capacities and open required logit failures (Qwen3.5-0.8B scalar
  profiles pass on PRO 5000 and focused L20; dynamic-batch profiles and
  experimental Qwen3.8-27B-FP8 retain required HF logit failures while passing
  bitwise native parity).
- **CPU portability:** compiled wrappers can inherit the build host's ISA
  (including AVX-512). CPU ISA compatibility is not checked at load time; use a
  compatible CPU or re-export on the target host.
- **Scope:** text generation (`xinfer_aoti`) and OpenAI-compatible HTTP/SSE
  serving (`xinfer_serve`) with dynamic `B=1..8` continuous batching,
  interleaved chunked prefill, and native tokenization/chat templates.
  PagedAttention, ragged 1D batching, `B > 8`, and CUDA Graphs are deferred.

## Code and documentation map

| Location | Responsibility |
| --- | --- |
| [`compiler/`](compiler/) | Model loading, adapters, graph passes, kernels, paired export, metadata and shared-weight helpers. |
| [`runtime/`](runtime/) | Native Rust stack: [`core/`](runtime/core/) (`xinfer-runtime`, `xinfer_aoti`, static C++ LibTorch shim), [`engine/`](runtime/engine/) (`xinfer-engine` scheduler, sampler, tokenizer), and [`server/`](runtime/server/) (`xinfer_serve`). |
| [`tools/`](tools/) | Checkpoint preparation, validation and benchmarks, with native Rust drivers under `tools/{benchmark,validation}/native/`. |
| [`environments/`](environments/README.md) | Per-target dependency pins, setup, checks, builds, Docker images and installation. |
| [`plugins/xinfer-perf/`](plugins/xinfer-perf/README.md) | Fusion and kernel optimization workflows for GB10 in Codex and Claude Code. |
| [`docs/`](docs/) | [Development tools](docs/development-tools.md), [package metadata/ABI](docs/package-metadata.md), [FlashAttention](docs/flash-attention.md) and [validation status](docs/validation-status.md). |

Tests follow the code they cover. Ignored directories hold builds (`build/`),
bundles/reports (`artifacts/`), and venvs/scratch files (`work/`). See
[AGENTS.md](AGENTS.md) for contribution rules and [PLAN.md](PLAN.md) for planned
work.
