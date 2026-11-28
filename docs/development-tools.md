# Development tools

Run commands from the repository root with the target interpreter `PY`
(`work/<target>/venv/bin/python`) and `build/` configured via the
[environment guide](../environments/README.md). Build `xinfer_attention_ops`
(and `xinfer_deltanet_ops` for Qwen3.5/Qwen3.8) before export or standalone
operator tests. Use fresh output paths for all full-model tools.

This guide owns shared developer commands and the numerical policy. See
[validation status](validation-status.md) for recorded results.

## Formatting and linting

Install pinned formatters with `bash environments/setup.sh --dev` (or a
standalone venv with `pip install -r requirements-dev.txt`). `requirements-dev.txt`
pins Ruff `0.17.3` (`ruff.toml`, 4-space, 100 columns) and clang-format `23.1.3`
(`.clang-format`, LLVM 2-space, 100 columns); Rust uses `cargo fmt` and
`cargo clippy`.

Format tracked files:

```bash
ruff format .
git ls-files -z '*.cpp' '*.h' | xargs -0 clang-format -i
cargo fmt --all
```

Check without modifying files:

```bash
ruff format --check .
ruff check --select E,F,W .
git ls-files -z '*.cpp' '*.h' | xargs -0 clang-format --dry-run --Werror
cargo fmt --all --check
XINFER_LINK_CONFIG=build/xinfer_rust_link.env CARGO_TARGET_DIR=build/rust   cargo clippy --workspace --locked --all-targets -- -D warnings
```

## CPU unit tests

Tests follow the code they cover (`compiler/*/tests/`, `environments/tests/`,
`tools/**/tests/`, and Rust `#[cfg(test)]` modules across `runtime/` and
`tools/{benchmark,validation}/native/`). CPU checks need no checkpoints, model
bundles or GPU (`test_weights` compiles tiny CPU AOTI packages and requires a
C++ compiler):

```bash
CUDA_VISIBLE_DEVICES='' "$PY" -m unittest discover
XINFER_LINK_CONFIG=build/xinfer_rust_link.env CARGO_TARGET_DIR=build/rust   CUDA_VISIBLE_DEVICES='' cargo test --workspace --locked
```

With `CUDA_VISIBLE_DEVICES=''`, CUDA-only tests skip cleanly while all CPU
contract tests run. Run GPU operator, adapter and export structural tests with
CUDA visible and the required native operator libraries built (listed in each
[model guide](../README.md#getting-started)).

## Checkpoint preparation and tokenization

`python -m tools.prepare` handles pinned checkpoint downloads, manifest
generation (`model_manifest.json`), verification, and offline tokenization for
`xinfer_aoti`. Run `download` before enabling `HF_HUB_OFFLINE=1` or
`TRANSFORMERS_OFFLINE=1`:

```bash
"$PY" -m tools.prepare download --model Qwen/Qwen3-0.6B
"$PY" -m tools.prepare download --model Qwen/Qwen3-1.7B
"$PY" -m tools.prepare download --model Qwen/Qwen3.5-0.8B
"$PY" -m tools.prepare download --model Qwen/Qwen3.5-4B
"$PY" -m tools.prepare download --model Qwen/Qwen3.8-27B-FP8
```

Default output directories are `work/models/qwen3-{0.6b,1.7b}`,
`work/qwen35/{0.8b,4b}/text`, and `work/qwen38/27b-fp8/text` (override with
`--output` and `--cache-dir`). Qwen3.5 and Qwen3.8 extract text-only weights
into `text.safetensors` on CPU (`tools/prepare/text_checkpoint.py`). Existing
output paths are refused, and directories publish only after verification.

Verify or tokenize locally (model family is inferred from `model_manifest.json`):

```bash
"$PY" -m tools.prepare verify --model-dir work/models/qwen3-0.6b
"$PY" -m tools.prepare encode --model-dir work/models/qwen3-0.6b   --prompt "What is the capital of France?" --output work/prompt.ids
"$PY" -m tools.prepare decode --model-dir work/models/qwen3-0.6b   --input work/generated.ids
```

`encode` uses the official non-thinking chat template, checks the model's
context ceiling, and refuses existing output files.

## Shared native executables

`build/` provides four Rust executables sharing `xinfer-runtime` (`runtime/core/`):

- `xinfer_aoti`: batch-one greedy generation from token IDs, stopping at EOS or
  `--max-output-tokens` and refusing occupied output/trace paths.
- `xinfer_serve`: OpenAI-compatible HTTP/SSE server (`/health`, `/ready`,
  `/v1/models`, `/v1/completions`, `/v1/chat/completions`) built when
  `XINFER_BUILD_SERVER=ON`.
- `xinfer_benchmark` (`tools/benchmark/native/`, `XINFER_BUILD_BENCHMARK=ON`):
  fixed-token prefill/decode timing and memory measurements (consumes all
  supplied tokens, including EOS).
- `xinfer_validate` (`tools/validation/native/`, `XINFER_BUILD_VALIDATION=ON`):
  logit/state trace capture and parity checks driven by a case manifest.

Model ABI is selected from `metadata.json`; native-path options (`--native`,
`--binary`, `--cli`) override each Python tool's `build/` default. See
[package metadata](package-metadata.md) and
[runtime installation](../environments/README.md#install-and-relocate-the-runtime).
Native `load_ms` includes integrity preflight, weight loading and loader setup;
first-use and steady-state timings are reported separately.

## Python validation tools

One entry point runs the full numerical chain for any model family (dispatched
by `metadata.json` `model_type`):

```bash
"$PY" -m tools.validation.validate   --packages artifacts/qwen35/0.8b/text --output artifacts/qwen35/0.8b/validation-new
```

`--packages` and a fresh `--output` are required; `--model-dir` and `--native`
override inferred defaults. HF references are generated live on the unmodified
checkpoint (Qwen3.5/Qwen3.8 scope memory-efficient SDPA for HF reference calls
only; Qwen3 also cross-checks manual HF greedy decoding against `HF.generate()`).

The chain compares **HF → adapted/optimized eager → Python AOTI → native AOTI
(`cpp_aoti`)** on identical teacher-forced inputs and partitions. Teacher-forced
state snapshots use sparse memory-mapped files under each Python stage's output
directory. Eager validation loads the prefill package's bundled attention
library; native execution relies on standard AOTI loading.

Dynamic-batch (`B=1..8` and batch transitions) and runtime-contract validators:

```bash
"$PY" -m tools.validation.qwen3.validate_batch   --packages artifacts/qwen3-batched --model-dir work/models/qwen3-0.6b   --output artifacts/qwen3-batch-new
"$PY" -m tools.validation.qwen35.validate_batch   --packages artifacts/qwen35/0.8b/text --model-dir work/qwen35/0.8b/text   --output artifacts/qwen35/0.8b/batch-new
"$PY" -m tools.validation.qwen38.validate_batch   --packages artifacts/qwen38/27b-fp8/text --model-dir work/qwen38/27b-fp8/text   --output artifacts/qwen38/27b-fp8/batch-new

"$PY" -m tools.validation.qwen3.validate_runtime   --packages artifacts/qwen3-flash --output artifacts/qwen3-runtime-new
"$PY" -m tools.validation.qwen35.validate_runtime   --packages artifacts/qwen35/0.8b/text --output artifacts/qwen35/0.8b/runtime-new
"$PY" -m tools.validation.qwen38.validate_runtime   --output artifacts/qwen38/27b-fp8/runtime-new
```

`qwen3.validate_runtime` accepts `--native` for `xinfer_aoti`; `qwen35`/`qwen38`
`validate_runtime` accept `--binary` (`xinfer_validate`) and `--cli`
(`xinfer_aoti`). Validation modules require Python assertions and refuse `-O`,
`-OO` and `PYTHONOPTIMIZE`. Inspect complete reports even on nonzero exits;
passing native parity does not clear required HF numerical failures.

### Shared numerical policy

All models follow `numerical_policy.name: shared_bf16` (version `1`):

| Check | Acceptance rule |
| --- | --- |
| Matched-input logits (`eager/HF`, `Python AOTI/eager`, `Python AOTI/HF`) | **Required:** worst-row relative L2 and peak-relative error `<= 5%` |
| HF logits replayed on the compiled greedy continuation | **Required:** same `<= 5%` logit bounds |
| Full-model KV, convolution and recurrent-state differences (`HF/eager/AOTI`) | **Diagnostic:** measured errors retained |
| HF-relative teacher top-1 and independent greedy-sequence differences | **Diagnostic:** retained with margins and first divergence |
| Native (`cpp_aoti`) vs Python AOTI on the same bundle | **Required:** bitwise logits/states and exact tokens/positions |
| Shapes, finite values, state ownership, masking, reset/reuse and overflow | **Required** independently of numerical errors |

Per-row metrics use a `1e-12` denominator floor:
`rel_l2 = norm(actual - ref) / max(norm(ref), 1e-12)` and
`peak_rel = max(|actual - ref|) / max(max(|ref|), 1e-12)`. `max_abs` is reported
without a separate full-model limit. Reports store all tensor `comparisons`,
required `failures`, and separate `diagnostics` (plus `top1_differences` and
`greedy_differences` views). Structural violations abort validation as errors.

For Qwen3.8-27B-FP8, the HF reference retains the checkpoint's FP8 weights and
dynamic activation quantization (`kernels-community/finegrained-fp8` v4),
measuring execution fidelity rather than quantization loss against unquantized
weights. Full-model validation never loads an upcast FP32 HF control; component
kernel tests run separately with their own output/state bounds.

## Benchmark HF and native AOTI

`python -m tools.benchmark` dispatches by package `model_type` and supports
`run`, `compare` and `check-idle`:

```bash
"$PY" -m tools.benchmark run   --packages artifacts/qwen3-flash --output artifacts/benchmark-new
```

`run` compares unchanged HF with a dynamic cache (`hf`) against native AOTI
(`cpp_aoti`, `build/xinfer_benchmark`) on fixed prompt and forced decode tokens.
HF processes the full prompt in one call while native prefill chunks at 512
tokens (`prefill_execution` records the difference). Defaults: `--lengths 25 128 512`,
`--decode-steps 64`, `--warmup 2`, `--repeats 5`, `--rounds 3`. Use lengths
`511 512 513 8191 8192` for chunk-boundary and long-prompt tests when device
memory allows.

All subcommands honor `CUDA_VISIBLE_DEVICES` / `CUDA_DEVICE_ORDER` and accept
`--gpu` (physical index or UUID). Idle checks sample GPU/memory activity once
per second, requiring 5 consecutive readings `<= 10%` within ~30s (`--allow-busy`
marks timings provisional):

```bash
"$PY" -m tools.benchmark check-idle --output artifacts/idle.json
```

Reports record GPU identity, package/binary/workload checksums, reset/prefill/
`first_prediction_ms`/decode/whole-request distributions, and separate allocator
peaks vs sampled device memory (do not sum them).

## Compare two compiled profiles

```bash
"$PY" -m tools.benchmark compare   --baseline-packages artifacts/aoti-baseline   --candidate-packages artifacts/aoti-candidate   --baseline-label baseline --candidate-label candidate   --output artifacts/comparison-new
```

Both bundles must share compatible checkpoints, toolchains, contracts and
logical weight identities. `compare` alternates execution order across 4 process
rounds by default (`--native` selects a custom benchmark binary). Inspect
per-workload and per-round results, use an identical-bundle A/A check when noise
is uncertain, keep profiler runs separate from unprofiled timings, and validate
accuracy independently on the same bundles.
