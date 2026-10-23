# xinfer

Qwen3-0.6B AOTInductor / C++ PoC.

For NVIDIA GB10 / ARM64 with PyTorch 2.13 and CUDA 13.0, see
[GB10.md](GB10.md) for setup, architecture, validation and run commands.
The [full GB10 benchmark](benchmarks/gb10-2026-10-23/REPORT.md) records the completed eager/AOTI comparison.
The historical environment and validation sections below describe the original L20 run.

The [Qwen3.5-4B text-only extension](compiler/qwen35/README.md) has completed its
first milestone: single DeltaNet block export, AOTI compilation and C++ validation.
Its compiler/runtime code, dependencies and artifacts are separate from the
Qwen3-0.6B baseline described below.

Steps 1-4 are complete: eager references, AOTI packages, a minimal C++ executable,
and three-path validation. The optional `--eager-numerics` compilation mode
passes with bitwise eager/AOTI/C++ agreement on the tested logits and full KV
cache. The original fused packages remain available; their numerical differences
and the compiler controls are documented below. Step 5's full benchmark has
completed on GB10 with idle checks, three process rounds and both AOTI profiles.
Optimization remains step 6 in [PLAN.md](PLAN.md).

The runtime interface is token IDs. Python handles offline tokenization and
text decoding; the C++ executable handles all model inference and generation.

## Repository layout

```text
compiler/                 Export/compilation, offline helpers and Python validation
  export.py, model.py   AOTI compilation and the thin Hugging Face model adapter
  prepare.py            Offline tokenization and decoding
  validate*.py, tests/  Eager/AOTI/native comparisons and numerical probes
  benchmark.py          Eager benchmark and native-worker orchestration
  requirements.*        Python dependency pins
  prompts.json          Reference prompt fixtures
runtime/                    Native runtime and its build configuration
  runtime.h, main.cpp   Shared runtime and token-ID CLI
  benchmark.cpp         Native benchmark driver
  tests/                Native validation driver
  CMakeLists.txt        LibTorch dependencies and native targets
CMakeLists.txt          Root build entry point
```

Run the commands below from the repository root. `.venv/`, `build/`, `artifacts/`
and `work/` remain there; executable paths stay `build/qwen_aoti`,
`build/qwen_validate` and `build/qwen_benchmark`. The Python scripts use sibling
imports and need no package installation or `PYTHONPATH` override.

## Environment

Run these commands on `kikoxia-any4.devcloud.woa.com`, port `36000`, as `root`.

```bash
cd /data/workspace/develop/qwen3-aoti-poc
```

The project `.venv` contains its own Hugging Face dependencies. Only Torch,
Triton, and their NVIDIA packages are linked from `/data/workspace/abo/.venv`.
ABO's unrelated packages are not on the project's import path. These links are
read-only dependencies by convention: do not upgrade or uninstall the shared
packages through this venv. The C++ build must use this same Torch build.

- Python 3.12.13; PyTorch runtime `2.7.0+cu126`; Triton 3.3.0.
- Transformers 4.51.3; tokenizers 0.21.4; safetensors 0.5.3.
- CUDA build 12.6; C++11 ABI enabled; NVIDIA L20, compute capability 8.9.
- `compiler/requirements.txt` pins the main packages; `compiler/requirements.lock` records the
  complete resolved package set. Torch's distribution metadata says `2.7.0`;
  its runtime version includes `+cu126`.

To recreate the project venv on this host, first create it in an absent `.venv`
directory, then link the existing native packages and install the locked extras:

```bash
/data/workspace/abo/.venv/bin/python -m venv .venv
.venv/bin/python - <<'PY'
from pathlib import Path
base = Path('/data/workspace/abo/.venv/lib/python3.12/site-packages')
dest = Path('.venv/lib/python3.12/site-packages')
paths = [base / name for name in ('torch', 'torchgen', 'functorch', 'triton', 'nvidia')]
for pattern in ('torch-*.dist-info', 'triton-*.dist-info', 'nvidia*.dist-info'):
    paths.extend(base.glob(pattern))
for source in paths:
    if not source.exists():
        raise FileNotFoundError(source)
    (dest / source.name).symlink_to(source, target_is_directory=True)
PY
.venv/bin/python -m pip install -r compiler/requirements.lock
.venv/bin/python -m pip check
```

## Model and offline commands

The default checkpoint directory is now `work/models/qwen3-0.6b` under the
repository root; `XINFER_MODEL_DIR` and `--model-dir` can override it.
The original L20 checkpoint lives at
`/data/workspace/develop/qwen3-rust-demo/models/qwen3-0.6b`; set that override
when reproducing the historical run below. Its manifest records revision
`c1899de289a04d12100db370d81485cdf75e47ca`.
The helpers read these files without modifying them or downloading another copy.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1

# Check the size and SHA-256 of all eight files in the existing manifest.
.venv/bin/python compiler/prepare.py verify

# Apply the official non-thinking chat template and write whitespace-separated IDs.
.venv/bin/python compiler/prepare.py encode \
  --prompt 'What is the capital of France? Answer in one short sentence.' \
  --output work/prompt.ids

# Verify the checkpoint and generate the reference fixtures.
.venv/bin/python compiler/validate.py --output artifacts/reference

# Turn a saved generated sequence back into text.
.venv/bin/python compiler/prepare.py decode \
  --input artifacts/reference/capital/generated.ids
```

Use `--model-dir PATH` before the `compiler/prepare.py` subcommand to select another
checkpoint directory. `compiler/validate.py` accepts the same option. Reference generation
refuses to overwrite an existing output directory; use a fresh `--output` for a
new run. Failed runs may leave a partial directory without `summary.json`.

## What the reference checks

`compiler/prompts.json` contains English factual, arithmetic, and Chinese prompts.
`compiler/validate.py` loads the unmodified HF model in BF16 with SDPA and performs:

1. A manual eager prefill/decode loop using HF's own DynamicCache.
2. A fresh-cache repetition, requiring identical tokens and bitwise-equal logits.
3. An independent call to HF `generate`, requiring the same greedy token sequence.

That call sets `use_model_defaults=False` so Transformers 4.51 cannot merge the
checkpoint's sampling defaults back into the explicit greedy configuration.

All logits must be finite. Thinking is disabled, generation is greedy, and the
default limit is 64 new tokens. Both checkpoint EOS IDs (`151645`, `151643`) are
honored. Prompt length and total context limits are checked before inference.
Greedy decoding is for deterministic checks, not a recommended quality setting.

The eager baseline deliberately uses HF DynamicCache. The separate step 2 checks
below validate the explicit static-cache wrapper and AOTI against that baseline
and HF's own StaticCache.

## Reference artifacts

Each case under `artifacts/reference/` contains:

| File | Meaning |
| --- | --- |
| `prompt.ids` | Whitespace-separated prompt IDs including the chat template |
| `generated.ids` | Generated IDs, including EOS if reached |
| `logits.f32` | Headerless, row-major, little-endian float32 logits |
| `metadata.json` | Shape, hash, stopping reason, and check results |
| `output.txt` | Decoded generated text with special tokens removed |

Logit row `i` predicts `generated.ids[i]` from the prompt plus
`generated.ids[:i]`. Row zero is the last prompt position; subsequent rows are
decode steps. BF16 model logits are converted to float32 for interchange. A C++
reader can read the raw floats directly; no Python serialization is required.

The root directory also contains `environment.json`, the verified model manifest,
and `summary.json` (written only when all cases pass). This run captures correctness
fixtures, not benchmark timings: copying logits to CPU and repeating generation
are intentionally outside the later performance measurement design.

`artifacts/`, `work/`, `.venv/`, and future `build/` are Git-ignored.

## Step 1 results — 2026-10-22

All eight checkpoint file sizes and SHA-256 values match the local manifest.
Dependency checks pass. The original ABO environment still has no Transformers
installation; the new packages are confined to the project venv.

| Fixture | Prompt tokens | Generated tokens, including EOS | Result |
| --- | ---: | ---: | --- |
| capital | 25 | 8 | The capital of France is Paris. |
| arithmetic | 28 | 11 | 17 + 25 = 42 |
| chinese | 24 | 24 | 机器学习是一种通过数据训练模型，使系统能够自动学习并做出预测或决策的计算机科学技术。 |

All three cases reached EOS, matched HF `generate` token-for-token, and reproduced
bitwise-identical logits with a fresh cache. All 43 saved prediction rows are
finite. The checkpoint loads as 596,049,920 unique parameters. The encode helper
reproduced the saved capital prompt IDs, and the decode helper reproduced the
Chinese output. These are correctness checks, not inference benchmarks.

Step 1 was committed as `e03359f` (`Establish Qwen3 eager reference baseline`).

## Export and validate AOTI

`compiler/model.py` is a small adapter around the original HF model. `TensorCache` writes
new K/V entries into caller-owned tensors using `index_copy_`. `QwenStep` builds
the causal mask from absolute positions and returns only last-token logits.
There is no copied Qwen implementation or custom kernel.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/work/inductor"
export TORCHINDUCTOR_COMPILE_THREADS=2

.venv/bin/python compiler/export.py decode
.venv/bin/python compiler/export.py prefill
.venv/bin/python compiler/validate_aoti.py
```

These commands use `artifacts/aoti/` by default. Export refuses to overwrite an
existing package; use `--output work/new-aoti` on both export calls and
`--packages work/new-aoti` on validation for a fresh build. All three commands
accept `--model-dir`. These original fused packages pass the step 2 checks;
the broader full-cache comparison finds a numerical limitation. The step 4
commands below build the accepted `--eager-numerics` correctness baseline in
`artifacts/aoti-eager/` without replacing `artifacts/aoti/`.

| File under `artifacts/aoti/` | Purpose |
| --- | --- |
| `prefill.pt2`, `decode.pt2` | Native AOTI packages, each including weights and kernels |
| `prefill.json`, `decode.json` | Tensor contract, model revision, build versions and options |
| `*.graph.txt` | Exported graph text, including input mutation outputs |
| `validation.json` | Successful validation metrics |
| `reference/*.f32` | Teacher-forced AOTI logits for subsequent C++ comparison |
| `code_audit.json`, `reload.json` | Recorded code inspection and fresh-process reload evidence |

The last two JSON files record the manual checks performed for this run; the
export and validation commands generate the other artifacts. AOTI reference
files have the same row order and float32 layout as the eager reference files.

### Package contract

Python signature: `(input_ids, cache_position, (K0, V0, ..., K27, V27))`.
The native loader receives 58 flat tensors in that order:

- CUDA int64 `input_ids[1,T]` and `cache_position[T]`.
- 56 contiguous CUDA BF16 cache tensors, each `[1,8,1024,128]` with strides
  `[1048576,131072,128,1]`. Cache storage totals 112 MiB.
- Output: one CUDA BF16 tensor `[1,151936]`.

Prefill accepts `T=2..512`; decode accepts `T=1`. A one-token prompt uses decode
at position zero. Positions must be contiguous absolute positions in `0..1023`,
with all earlier positions already populated. Token IDs must be in `0..151935`.
The caller owns these checks and must reset every cache tensor between requests.
Both loaders receive the same buffers. Inputs need 16-byte-aligned storage;
ordinary LibTorch allocations satisfy this. Do not pass offset storage views.
The C++ executable enforces token IDs, prompt length and context budgets before
loading weights. It creates aligned tensors itself and checks the package pair's
metadata, Torch version and C++ ABI.

Compile and load with the matching Torch/CUDA build recorded in the JSON files.
The packages target the inspected L20 (SM89); portability to other GPUs/builds
has not been tested. Two loaders currently hold independent copies of weights.

### Step 2 results — 2026-10-22

The eager adapter matched unmodified HF StaticCache **bitwise** for all 43 logit
rows and all 56 K/V tensors. Compiled prefill followed by repeated compiled
decode reproduced all three original greedy sequences, including EOS.

| Fixture | Max absolute logit difference, AOTI vs BF16 static | Max row relative L2 | Top-1 matches |
| --- | ---: | ---: | ---: |
| arithmetic | 0.6484375 | 3.052% | 11/11 |
| capital | 0.4375 | 2.648% | 8/8 |
| chinese | 0.5 | 2.200% | 24/24 |

The initial pointwise `atol=0.25, rtol=0.02` check failed. We investigated with
the same checkpoint weights promoted to FP32: eager BF16 differs from that
control by up to 3.646% row relative L2, while AOTI differs by up to 3.391%.
All 43 top-1 tokens match FP32 too. This is consistent with accumulated BF16
rounding across fused operations, rather than a cache-adapter discrepancy.
PyTorch also documents [numerical changes from compilation and fusion](https://pytorch.org/blog/training-production-ai-models/).

Validation now requires finite values and limits both `||actual-reference||2 /
||reference||2` and `max(abs(actual-reference)) / max(abs(reference))` to 5%,
per logit row and per whole K/V tensor. It reports absolute error separately,
requires exact fixture tokens and greedy sequences, and reruns the FP32 control.
These are PoC acceptance limits, not a general model-quality guarantee.
`emulate_precision_casts=True` retains BF16 cast boundaries but does not make
fused arithmetic bitwise equal to eager execution.

Additional checks passed:

- Prompt lengths 1, 2, 7, 128, 511 and 512, followed by decode. Worst logit row
  relative L2 is 3.927%; every tested top-1 prediction matches.
- Chunked prefill through position 1021, then decode at 1022 and 1023.
- Stable addresses and bitwise-unchanged slots outside each update in all 56
  cache tensors. Worst fixture K/V tensor relative L2 is 3.309%.
- Poisoning future cache slots leaves logits bitwise unchanged; unused slots
  retain the poison. Reset/reuse after a full context reproduces the capital case.
- Fresh-process reload after moving the packages to `artifacts/aoti/` reproduces
  the capital AOTI logits bitwise. Transformers was not imported and an empty
  compiler-cache location was not created. That step 2 check used Python's AOTI
  loader; the standalone C++ results are below.

Generated-code inspection found direct writes into all caller K/V buffers and
no full-cache copy-back. The generated alignment guards can clone unaligned
inputs, hence the alignment requirement. SDPA lowers to ATen's CUDA efficient
attention backend. It still materializes padded K/V expansion from 8 KV heads
to 16 query heads across all 1,024 slots. Persistent cache updates work, but this
attention path is not optimized for low-latency LLM serving.

| Package | Bytes | Export + compile time |
| --- | ---: | ---: |
| decode.pt2 | 1,193,995,187 | 41.9 s |
| prefill.pt2 | 1,194,162,006 | 56.8 s |

Compilation reused caches from earlier probes, so these are not cold-build
benchmarks. Controlled runtime performance measurements remain for step 5.

## Build and run the C++ executable

`runtime/main.cpp` handles the CLI and diagnostic timing. `runtime/runtime.h` contains
metadata/input validation, two AOTI loaders, one shared cache, and a greedy
generation loop. Its checked `step()` advances contiguous cache positions;
`generate()` resets the same storage before each request. Both the CLI and the
optional validation driver use this implementation. It uses the native
[`AOTIModelPackageLoader`](https://docs.pytorch.org/docs/2.7/torch.compiler_aot_inductor.html)
API. `runtime/CMakeLists.txt` links LibTorch and fetches the header-only nlohmann/json
3.11.3 release with its published SHA-256. The first configuration needs network
access for that header dependency; subsequent builds reuse `build/_deps/`.

```bash
cd /data/workspace/develop/qwen3-aoti-poc

/data/workspace/abo/.venv/bin/cmake -S . -B build -G Ninja \
  -DCMAKE_PREFIX_PATH=/data/workspace/abo/.venv/lib/python3.12/site-packages/torch/share/cmake \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DTORCH_CUDA_ARCH_LIST=8.9 -DCMAKE_BUILD_TYPE=Release
/data/workspace/abo/.venv/bin/cmake --build build -j2

# Run the native executable directly; no Python interpreter or activated venv.
./build/qwen_aoti artifacts/aoti \
  artifacts/reference/capital/prompt.ids work/generated.ids \
  --max-new-tokens 64

# Optional validation trace: one raw float32 row per generated token, including EOS.
./build/qwen_aoti artifacts/aoti \
  artifacts/reference/capital/prompt.ids work/generated.ids \
  --max-new-tokens 64 --trace work/native-logits.f32

# Offline text decoding, separate from native inference.
.venv/bin/python compiler/prepare.py decode --input work/generated.ids

# Reproducible native smoke checks; Python launches the binary and checks results.
.venv/bin/python compiler/tests/smoke_cpp.py
```

Usage: `qwen_aoti PACKAGE_DIR PROMPT_IDS OUTPUT_IDS [--max-new-tokens N]
[--trace LOGITS_F32]`. The default budget is 64 new tokens. The requested prompt
plus generation budget must fit 1,024 tokens, even if EOS might arrive earlier.
Output IDs contain only generated tokens, including EOS when reached. Trace row
`i` predicts output token `i`; its shape is `[generated_tokens,151936]`, raw
little-endian float32 on this host. Input, ID-output and trace paths must differ.
Output directories are created as needed; existing output files are overwritten.

The executable uses CUDA device 0; select the visible GPU with
`CUDA_VISIBLE_DEVICES` before launch. Every request resets the caller-owned cache.
The CLI handles one request per process. Tensor allocation and a scalar GPU-to-CPU
synchronization per token are deliberate baseline choices for later profiling.
Reported load/generation times are cold-run diagnostics, with trace overhead
included when enabled; they are not the step 5 performance benchmark.

This host has CUDA Toolkit 12.2 for CMake/compiler discovery; Torch is the same
`2.7.0+cu126` build used to produce the packages. The build compiles C++ only and
loads the existing compiled GPU kernels. CMake needed the explicit `nvcc` path.
The binary's runpath locates the existing LibTorch/CUDA shared libraries; copying
only the executable to another host is not a complete deployment bundle.

### Step 3 results — 2026-10-22

- Standalone C++ generated all three reference sequences with the same EOS
  stopping behavior. All 43 logit rows are bitwise equal to Python AOTI results.
- Native prompt lengths 1, 2 and 512, each followed by up to three generated
  tokens, match Python AOTI tokens and logits bitwise.
- Twelve invalid-input/CLI cases are rejected, including empty, malformed,
  negative and out-of-vocabulary IDs, oversized prompts, invalid generation
  budgets, unknown/incomplete options and colliding trace/output paths. These
  checks run with CUDA hidden to verify rejection precedes model loading.
- A one-token generation budget stops after exactly one prediction.
- `ldd` resolves the native dependencies without `libpython` or `libtorch_python`;
  loader tracing during real inference confirms neither is dynamically loaded.
  Child runs clear venv/Python/library-path overrides and use an invalid
  `PYTHONHOME`, with no Python interpreter invocation in the executable.

Step 3 was committed as `20b7c98` (`Add minimal native Qwen3 AOTI generation runtime`),
with author and committer email `txiadev@163.com`.
Smoke artifacts, raw traces, dependency/loader logs and `summary.json` are saved
under `artifacts/native-smoke/`. These checks also pass after step 4 extracted
the shared runtime into `runtime/runtime.h`.

## Validate all three execution paths

Build the optional test executable using the existing CMake configuration:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export TORCHINDUCTOR_COMPILE_THREADS=2
export TORCHINDUCTOR_CACHE_DIR="$PWD/work/inductor-eager-reductions"

.venv/bin/python compiler/export.py prefill --eager-numerics --output artifacts/aoti-eager
.venv/bin/python compiler/export.py decode --eager-numerics --output artifacts/aoti-eager
.venv/bin/python compiler/validate_aoti.py --packages artifacts/aoti-eager

/data/workspace/abo/.venv/bin/cmake -S . -B build -DQWEN_BUILD_VALIDATION=ON
/data/workspace/abo/.venv/bin/cmake --build build -j2

.venv/bin/python compiler/validate_cpp.py --packages artifacts/aoti-eager \
  --output artifacts/three-path-eager
.venv/bin/python compiler/tests/smoke_cpp.py --packages artifacts/aoti-eager \
  --output artifacts/native-smoke-eager
```

Export refuses to replace packages; choose a fresh package directory for a
rebuild. The `compiler/validate_cpp.py` output directory must be absent; choose a fresh
`--output` for another run.
`--native`, `--packages`, `--reference`, and `--model-dir` select alternate inputs.
This is a correctness test, with cache snapshots and CPU transfers; do not use
its timings as benchmarks. The eager-numerics packages pass. The original fused
packages in `artifacts/aoti/` still return exit status 1 for the full-cache
numerical failures below. Every completed report records `passed`, all cases,
any violating tensors and an independent FP32 control, including passing runs.

`runtime/tests/validate_native.cpp` runs one pair of loaders and one set of 56 KV buffers
through 24 cases. `compiler/validate_cpp.py` runs the original HF StaticCache model and
Python-loaded AOTI on the same inputs, then compares their results with C++:

- Teacher-forced reference continuations and explicit alternative tokens,
  including both EOS IDs as ordinary inputs to `step()`.
- Prompt lengths 1, 2, 7, 128, 511 and 512, each followed by three decode steps.
  Repeating each case with future cache slots filled with 3 verifies masking.
- A 1,024-token prefix in chunks of 512, 510, 1 and 1. All cache buffers retain
  their addresses; each update preserves every slot outside its write region.
  The next step is rejected without changing the cache or its position.
- Greedy generation with EOS and limits of 1 and 3, different requests in
  succession, and the same request repeated after filling the entire cache.

The directory contains `cases.json`, `native.json`, `native.log`, raw C++/HF/AOTI
logit traces, `context_end.kv.bf16`, and `summary.json`. The cache dump concatenates
K0, V0, ..., K27, V27 as contiguous BF16 tensors in the documented layout.
Native/Python AOTI cache comparison checks the raw 16-bit contents exactly.

### Original fused-package results — 2026-10-22

| Check | Result |
| --- | --- |
| C++ vs Python AOTI logits | All 163 rows bitwise equal |
| C++ vs Python AOTI full KV cache | All 56 tensors bitwise equal |
| Top-1 predictions vs HF BF16 | 163/163 match |
| Greedy sequences, EOS and generation limits | Pass |
| Cache addresses, reset/reuse, write regions and overflow rejection | Pass |
| Future-slot poisoning and repeated-request logits | Bitwise equal |
| AOTI vs HF BF16 logits | Pass: max relative L2 3.9264%, peak-relative error 2.9240% |
| AOTI vs HF BF16 full-context KV tensors | **Fail: 21/56 exceed one or both 5% limits** |

The full-cache comparison is stronger than step 2, which checked numerical KV
contents on the short fixtures and checked logits/write regions at full context.
The new check finds maximum whole-tensor relative L2 **6.5913%**, maximum
peak-relative error **17.8385%**, and maximum absolute error **8.5625**. Logit
error limits are unchanged (maximum absolute difference remains 0.6484375).

Promoting the same checkpoint weights to FP32 gives this independent full-cache
control. The original BF16 model's rotary-frequency buffer is kept in FP32:

| Comparison | Worst KV relative L2 | Worst KV peak-relative error |
| --- | ---: | ---: |
| AOTI BF16 vs HF FP32 | 3.3822% | 8.7940% |
| HF BF16 vs HF FP32 | 6.0668% | 21.2242% |

Both BF16 paths show outliers against FP32; AOTI has smaller worst-case errors
in this control. This supports numerical sensitivity as an explanation, but it
does not make the existing cache criterion pass or establish model quality.
An isolated prefill build with `max_fusion_size=1` also failed (worst full-cache
relative L2 6.5222%, peak-relative error 30.7927%); it was not promoted. Its
probe and results remain under `work/`.

These failure metrics apply to the original packages, which remain unchanged.
Exact native/AOTI agreement establishes that the discrepancy is also present
through Python AOTI. The eager-numerics mode below passes the same 5% limits
without changing the model or the acceptance criteria.

### First-divergence investigation — 2026-10-22

The first differing BF16 activation is **layer 0's rotary position embedding
(RoPE)**, in `q * cos + rotate_half(q) * sin`. Layer numbers here are zero-based.
The trace uses the first 512-token chunk of the actual `context_end` case,
which already contains the large layer-14 V outlier.

To preserve the original fusion decisions, a diagnostic package saves existing
buffers after kernel launches instead of adding model outputs. Its logits and
all 56 cache tensors were checked bitwise against the original package.
HF hooks capture the corresponding eager BF16 and FP32 stages. Input RMSNorm
and Q/K/V projections match eager BF16 exactly. Same-input probes also establish
equal BF16 Q normalization, sine/cosine and separate multiplication results;
the divergence appears when the multiplication and addition execute together.

| Layer-0 stage | AOTI vs BF16 relative L2 | AOTI vs full-model FP32 relative L2 | BF16 vs full-model FP32 relative L2 |
| --- | ---: | ---: | ---: |
| Input RMSNorm | 0% | 0.22278% | 0.22278% |
| Q projection | 0% | 0.20606% | 0.20606% |
| Q after RoPE | 0.13216% | 0.39093% | 0.39816% |
| K after RoPE | 0.01752% | 0.29277% | 0.29293% |
| Attention output projection | 0.25745% | 0.41853% | 0.42183% |
| MLP down projection | 0.58394% | 0.72890% | 0.73222% |

141,635 of 1,048,576 Q values and 69,599 of 524,288 K values differ after RoPE;
the maximum absolute difference is 0.125 in both. The FP32 reference promotes
the same checkpoint weights, preserves the original FP32 rotary-frequency
buffer, and disables TF32. It is a numerical control for these inputs, not a
model-quality evaluation.

**The arithmetic mechanism is confirmed.** Let `R` round to BF16, with
`p = q*cos` and `r = rotate_half(q)*sin` computed from identical BF16 operands:

```text
Eager: R(R(p) + R(r))
AOTI:  R(R(p) +   r )
```

The second formula reproduces every Q and K value from the original compiled
layer-0 kernel exactly. PTX/SASS inspection finds `fma.rn.bf16x2` /
`HFMA2.BF16_V2`: the backend contracts a multiply and add, eliminating one
intermediate rounding step. `emulate_precision_casts=True` is enabled, but
cast boundaries in the generated Triton source do not prevent this contraction.
This is distinct from Inductor graph fusion; limiting `max_fusion_size` did not
resolve the acceptance failure.

For example, Q at `[batch=0, head=0, position=423, channel=32]` has operands
`q=18.5`, `cos=0.91015625`, `rotated_q=0.76171875`, `sin=0.41015625`:

```text
p = 16.837890625       R(p) = 16.875
r =  0.3124237060547   R(r) =  0.3125
Eager sum before final rounding: 17.1875          -> BF16 17.25
AOTI fused sum:                  17.1874237060547 -> BF16 17.125
FP32 p+r with those same inputs: 17.1503143310547
```

The eager sum is exactly a BF16 midpoint and rounds to 17.25; the fused sum
lies just below that midpoint. AOTI is closer to FP32 for this same-input
example. The full-model FP32 activation at this coordinate is instead
17.30856514 because its upstream activations differ: the local arithmetic
control and full-model control answer different questions.

`compiler/tests/rope_precision.py` reproduces this with four values repeated 256 times
to exercise packed BF16 arithmetic. It needs only Torch and CUDA, without a
checkpoint or Transformers. A one-element kernel has a different lowering
and does not reproduce the discrepancy on this build.

```bash
.venv/bin/python compiler/tests/rope_precision.py --output work/rope-vector-default
.venv/bin/python compiler/tests/rope_precision.py --disable-fp-fusion \
  --output work/rope-vector-no-fusion
```

Use fresh output directories. Both runs preserve the separate multiplication
results; the default compiled sum is 17.125 and the disabled-contraction sum
is 17.25, equal to eager. The control uses `TRITON_DEFAULT_FP_FUSION=0`, a
[Triton 3.3 compiler setting](https://github.com/triton-lang/triton/blob/v3.3.0/third_party/nvidia/backend/compiler.py).
It must be set at compilation; changing it when loading a package has no effect
on already generated kernels.

We also recompiled **both full-model packages** with that setting and reran the
complete 24-case/163-row native validation matrix:

| Full-context KV comparison | Default compilation | FP contraction disabled |
| --- | ---: | ---: |
| AOTI vs HF BF16: worst relative L2 | 6.5913% | 2.0656% |
| AOTI vs HF BF16: worst peak-relative error | 17.8385% | 15.3285% |
| Tensors failing the existing limits | 21/56 | 15/56, peak limit only |
| Layer-14 V: relative L2 | 6.5913% | 1.7124% |
| Layer-14 V: peak-relative error | 17.8385% | 1.3184% |
| AOTI vs HF FP32: worst relative L2 | 3.3822% | 6.1121% |
| AOTI vs HF FP32: worst peak-relative error | 8.7940% | 21.3546% |

Native/Python AOTI parity, all 163 top-1 predictions, greedy sequences and cache
behavior checks still pass. Layer-0 K/V now match eager bitwise throughout the
full context. In the 512-token diagnostic trace, all observed layer-0 stages
match eager exactly. The next difference is **layer-1 Q RMSNorm's FP32
sum-of-squares reduction**. Its Q-projection inputs are equal. Adding diagnostic
stores to the actual generated Triton kernel preserves its output bitwise and
shows that different FP32 reduction order pushes 3 normalized values across
BF16 rounding midpoints. The post-RoPE output differs in 3 of 1,048,576 values,
with maximum absolute difference 0.001953125. Sine and cosine still match eager.
Substituting the compiled sum into eager RMSNorm reproduces the compiled
normalization exactly; applying eager RoPE to that result reproduces the full
compiled kernel exactly. This isolates the residual difference to the reduction.
Isolated normalization/RoPE probes happen to match eager; only the original
fused kernel reproduces these differences, so the exact lowering matters.

For example, at position 440/head 5, the squared-input sum is 2.332119941711426
in the generated kernel versus 2.3321194648742676 in eager. At channel 67,
the normalized FP32 value before conversion is 0.3056640625 versus
0.3056640923023224. The tiny difference crosses a BF16 midpoint; after the
learned scale, the results are 0.30859375 versus 0.310546875. The same-input
FP32 RMSNorm result is 0.3104400932788849, closer to eager at this coordinate.
Across the entire normalized Q tensor, both BF16 paths have approximately
0.23328% relative L2 error versus the same-input FP32 control.

Disabling contraction improves eager parity while worsening the worst FP32
errors in this stress case. It is a diagnostic control, not an accepted
accuracy fix. The original packages, runtime and 5% limits remain unchanged;
this contraction-only control still fails acceptance. Neither configuration's
finite fixture results establish general generation quality, and performance
of the control was not measured.

Evidence on the remote host, relative to the project directory:

- `work/precision-trace/`: stage comparisons, same-input probes, arithmetic
  reconstruction, saved tensors and generated instruction inspection.
- `work/precision-trace-no-fma/` and `work/precision-trace-layer1-no-fma/`:
  contraction-disabled layer traces and probes.
- `work/no-fp-fusion-aoti/`: separate experimental packages and setting audit.
- `artifacts/three-path-no-fp-fusion/summary.json`: complete control validation,
  including all remaining failures and FP32 comparisons.

### Step 4 acceptance with eager numerics — 2026-10-23

`compiler/export.py --eager-numerics` combines two controls: it disables Triton FP
contraction and retains `aten.mean.dim` as a native LibTorch operation. This
preserves eager's RMSNorm reduction order on the tested workload. All other
operators follow the existing AOTI compilation path, with
`emulate_precision_casts=True`. The Qwen model and C++ inference loop are unchanged.

Torch 2.7 has no dedicated AOTI C shim for this mean overload, so the generated
package dispatches it through the native AOTI proxy executor. The standalone
C++ checks pass without loading Python. There is no Python fallback or custom
operator registration. Package metadata records the numerical mode and actual
FP-fusion setting; the runtime rejects a pair with different numerical settings.
Omitting the flag retains the original compilation behavior.

| Check on `artifacts/aoti-eager/` | Result |
| --- | --- |
| HF BF16 StaticCache vs Python AOTI vs C++ logits | All 163 rows bitwise equal |
| HF BF16 StaticCache vs Python AOTI vs C++ full-context KV | All 56 tensors bitwise equal |
| Existing 5% numerical limits | Pass; measured BF16 differences are zero |
| Greedy sequences, EOS and generation limits | Pass |
| Cache write regions, masking, reset/reuse, final slot and overflow | Pass |
| Step 2 wrapper, handoff, shape and FP32 regression | Pass |
| Native CLI smoke and Python-free linkage/loading | Pass |
| Invalid inputs and mixed numerical settings | 13 rejection cases pass |

The bitwise checks compare the stored representations, including signed zero.
These results establish equality on the recorded test matrix, not a guarantee
of bitwise equivalence for every input or future compiler/GPU version.

Against full-model FP32, both this profile and eager BF16 have worst full-cache
relative L2 6.0668% and peak-relative error 21.2242%. The original fused AOTI
profile remains closer to FP32 in that stress case (3.3822% and 8.7940%). This
mode establishes faithful eager execution for the PoC; it is not an accuracy
improvement over FP32 or a performance optimization. The extra native reductions
may cost latency; step 5 will measure this profile and the original fused profile
separately, and step 6 can evaluate arithmetic/performance tradeoffs explicitly.

Final reports are `artifacts/three-path-eager/summary.json`,
`artifacts/aoti-eager/validation.json`, and `artifacts/native-smoke-eager/summary.json`.
The initial diagnostic build and passing report remain under
`work/eager-reductions-aoti/` and `artifacts/three-path-eager-reductions/`.
Step 4 was committed as `02587ac` with author and committer email `txiadev@163.com`.

## Benchmark the baseline

`runtime/benchmark.cpp` uses the same checked `Runtime` as the native CLI. Its small
driver adds fixed-token requests and measurement only. `compiler/benchmark.py` runs the
matching eager wrapper and orchestrates fresh processes for three profiles:

| Profile | Execution | Packages |
| --- | --- | --- |
| `eager` | Python, original HF layers through `QwenStep` | Local checkpoint |
| `aoti_eager` | C++, eager-numerics compilation | `artifacts/aoti-eager/` |
| `aoti_fused` | C++, original fused compilation | `artifacts/aoti/` |

The fused profile retains the step 4 numerical failures; a timing comparison
does not establish equivalent numerical accuracy or generation quality.

```bash
cd /data/workspace/develop/qwen3-aoti-poc

# Use the existing Release build configured in the C++ section above.
/data/workspace/abo/.venv/bin/cmake -S . -B build -DQWEN_BUILD_BENCHMARK=ON
/data/workspace/abo/.venv/bin/cmake --build build --target qwen_benchmark -j2

# Requires an idle GPU; choose an absent output directory for every run.
.venv/bin/python compiler/benchmark.py run --output artifacts/benchmark

# Functional smoke check only; shared-GPU results are marked provisional.
.venv/bin/python compiler/benchmark.py run --output artifacts/benchmark-smoke-new \
  --rounds 1 --warmup 1 --repeats 2 --lengths 1 512 \
  --decode-steps 2 --allow-busy

# Run only the native worker against a saved workload, without Python.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./build/qwen_benchmark \
  artifacts/aoti-eager artifacts/benchmark/workload.json work/native-benchmark.json
```

The default workload uses prompt lengths **25, 128 and 512**, followed by
**64 decode steps**. The capital fixture's prompt and continuation IDs repeat
as needed to construct each sequence. All profiles consume identical IDs:
this is teacher-forced throughput measurement, with EOS treated as ordinary
input. It produces 65 predictions per request, including the prefill prediction.
The production CLI still stops generation at EOS.

For each shape, a worker records one first-use request, runs two additional
warmup requests, then measures five requests. Three fresh-process rounds rotate
profile order, giving 15 measured requests per profile and shape. Each request
zeros the entire shared cache. All workers use one OpenMP/MKL CPU thread. Native
workers clear Python and library-path overrides and use an invalid `PYTHONHOME`.

Timings use synchronized host wall clocks:

- `cuda_init_ms` measures CUDA initialization separately.
- `load_ms` measures model/package loading and cache allocation after CUDA
  initialization. Native loading includes both packages. Imports and executable
  dynamic linking are outside this interval. Filesystem caches are uncontrolled;
  package hashing may warm them, so this is not cold-disk load latency.
- `reset_ms` measures cache reset separately from prefill.
- `prefill_ms` is inference-only time to the first predicted token. It includes
  input construction/transfer, model execution, GPU argmax, scalar CPU transfer
  and synchronization. It excludes loading, cache reset, tokenization and I/O.
- `decode_ms_per_token` is the synchronized 64-step block divided by 64. It
  includes input construction, native runtime checks, GPU argmax and the same
  scalar GPU-to-CPU synchronization used by the production loop. It is not a
  device-only kernel metric. Tokens/second is 1,000 divided by the median ms/token.
- `request_ms` includes reset, prefill and decode, including boundary syncs.

First-use requests are retained separately and excluded from warm statistics.
The first use of a later shape is not a cold process. Medians, p10/p90, means
and ranges summarize request samples; decode percentiles describe **block
averages**, not individual-token latency. Predictions must be stable across
repetitions and process rounds. The eager-numerics profile must match eager
prediction IDs; fused-profile agreement is reported. Full-logit/cache numerical
validation remains the separate step 4 suite.

Memory reports include PyTorch allocator allocated/reserved peaks and CUDA
device-memory samples after context initialization, loading and each request.
AOTI's raw CUDA weight allocations bypass the allocator counters, so allocator
peaks alone materially undercount native memory. `sampled_peak_device_used_bytes`
is the highest request-boundary device usage; `sampled_peak_device_delta_bytes`
subtracts that worker's post-initialization usage. These device-wide samples can
include other allocations and miss transient peaks. The native baseline loads
two independent weight pools; both packages share the same 112 MiB KV cache.

The runner requires five consecutive samples with both GPU and memory-controller
utilization at or below 10% before and after every worker. It aborts after 30
unsuccessful checks. One-second GPU utilization, memory, clocks and power logs
help assess interference, but boundary checks alone do not prove exclusivity.
`--allow-busy` bypasses this requirement for harness checks and labels the report
`provisional_shared_gpu`; those results must not be used as the final baseline.

The output directory saves the exact workload, source snapshots/hashes, native
binary hash, model manifest, package metadata/sizes/hashes, GPU/driver details,
individual worker reports/logs, idle checks, telemetry and `summary.json`.
Package metadata includes the recorded export/compile elapsed time. Those
existing builds used different compiler-cache states, so their times are not
a controlled comparison of compilation speed. Interrupted runs preserve partial
evidence and do not produce a completed summary; choose a new directory to retry.

### Step 5 implementation and original L20 checks — 2026-10-23

Recorded export/compile wall times and package sizes:

| Profile | Prefill compile (s) | Decode compile (s) | Prefill package (bytes) | Decode package (bytes) |
| --- | ---: | ---: | ---: | ---: |
| Eager numerics | 54.74 | 41.88 | 1,194,398,762 | 1,194,224,280 |
| Original fused | 56.81 | 41.91 | 1,194,162,006 | 1,193,995,187 |

These are metadata from the existing builds, with the compiler-cache caveat
above. The final benchmark-target rebuild took 9.63 seconds using the existing
Release configuration and dependencies; this is not a clean project build time.

The benchmark builds successfully. A shared-GPU smoke run at prompt lengths 1
and 512 passed for all three profiles: repeated predictions were stable and
all three predictions per case agreed with eager. The native consumed-position
check also passed. A further run at lengths 25, 128 and 512 with 64 decode steps
passed for all profiles, with 65/65 predictions agreeing with eager at each
length and one CPU thread confirmed in each worker. These checks include
first-use, warmup and two measured requests per case. They validate the harness;
their timings are provisional because the GPU was shared.

The idle guard correctly rejected a timing run while the otherwise occupied
GPU reported 99-100% utilization. Evidence is saved under
`artifacts/benchmark-smoke/`, `artifacts/benchmark-validation/`,
`work/benchmark-position-smoke.json` and `work/benchmark-idle-check.log`.

Full measurements subsequently completed on GB10 using PyTorch 2.13/CUDA 13.0.
The [report](benchmarks/gb10-2026-10-23/REPORT.md) records 135 measured requests,
15 samples per profile/length and passing idle checks around all nine workers.
AOTI eager numerics delivers 1.48–1.49x decode and 1.33–1.50x prefill speedups;
fused AOTI delivers 1.61–1.62x decode speedup but fails the unchanged full-cache
criterion on 9/56 tensors. All profiles agree on the benchmark predictions.
These measurements used automatic clocks, with active-sample median SM clock
611 MHz. Step 6 has not started.

## Planned optimization phase

Step 6 in [PLAN.md](PLAN.md) follows the C++ implementation, validation and
baseline measurements in steps 3-5. It covers supported Torch/Inductor options,
native CUDA Graph capture/replay, targeted graph transformations, and compatible
high-performance attention/KV-cache kernels. Profiling will quantify the padded
GQA expansion and other bottlenecks before choosing changes. Each experiment
must preserve the correctness checks and report latency, memory and compilation
or capture costs against the reproducible baseline. Keep optimizations isolated
and optional so the minimal runtime remains readable.
