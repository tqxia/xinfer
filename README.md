# xinfer

Qwen3-0.6B AOTInductor / C++ PoC.

Steps 1-3 are complete: eager references, validated AOTI packages, and a minimal
C++ executable. Broader validation, benchmarking and optimization are steps 4-6
in [PLAN.md](PLAN.md).

The runtime interface is token IDs. Python handles offline tokenization and
text decoding; the C++ executable handles all model inference and generation.

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
- `requirements.txt` pins the main packages; `requirements.lock` records the
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
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip check
```

## Model and offline commands

The default checkpoint directory is
`/data/workspace/develop/qwen3-rust-demo/models/qwen3-0.6b`.
Its manifest records revision `c1899de289a04d12100db370d81485cdf75e47ca`.
The helpers read these files without modifying them or downloading another copy.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1

# Check the size and SHA-256 of all eight files in the existing manifest.
.venv/bin/python prepare.py verify

# Apply the official non-thinking chat template and write whitespace-separated IDs.
.venv/bin/python prepare.py encode \
  --prompt 'What is the capital of France? Answer in one short sentence.' \
  --output work/prompt.ids

# Verify the checkpoint and generate the reference fixtures.
.venv/bin/python validate.py --output artifacts/reference

# Turn a saved generated sequence back into text.
.venv/bin/python prepare.py decode \
  --input artifacts/reference/capital/generated.ids
```

Use `--model-dir PATH` before the `prepare.py` subcommand to select another
checkpoint directory. `validate.py` accepts the same option. Reference generation
refuses to overwrite an existing output directory; use a fresh `--output` for a
new run. Failed runs may leave a partial directory without `summary.json`.

## What the reference checks

`prompts.json` contains English factual, arithmetic, and Chinese prompts.
`validate.py` loads the unmodified HF model in BF16 with SDPA and performs:

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

`model.py` is a small adapter around the original HF model. `TensorCache` writes
new K/V entries into caller-owned tensors using `index_copy_`. `QwenStep` builds
the causal mask from absolute positions and returns only last-token logits.
There is no copied Qwen implementation or custom kernel.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/work/inductor"
export TORCHINDUCTOR_COMPILE_THREADS=2

.venv/bin/python export.py decode
.venv/bin/python export.py prefill
.venv/bin/python validate_aoti.py
```

These commands use `artifacts/aoti/` by default. Export refuses to overwrite an
existing package; use `--output work/new-aoti` on both export calls and
`--packages work/new-aoti` on validation for a fresh build. All three commands
accept `--model-dir`. The existing validated packages are ready for step 3.

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

`main.cpp` contains metadata/input validation, two AOTI loaders, one shared cache,
and a greedy generation loop. It uses the native
[`AOTIModelPackageLoader`](https://docs.pytorch.org/docs/2.7/torch.compiler_aot_inductor.html)
API. `CMakeLists.txt` links LibTorch and fetches the header-only nlohmann/json
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
.venv/bin/python prepare.py decode --input work/generated.ids

# Reproducible native smoke checks; Python launches the binary and checks results.
.venv/bin/python tests/smoke_cpp.py
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

Smoke artifacts, raw traces, dependency/loader logs and `summary.json` are saved
under `artifacts/native-smoke/`. Step 4 will extend the three-path checks across
the full boundary matrix and repeated requests in one native process. Step 5
will establish controlled performance measurements.

## Planned optimization phase

Step 6 in [PLAN.md](PLAN.md) follows the C++ implementation, validation and
baseline measurements in steps 3-5. It covers supported Torch/Inductor options,
native CUDA Graph capture/replay, targeted graph transformations, and compatible
high-performance attention/KV-cache kernels. Profiling will quantify the padded
GQA expansion and other bottlenecks before choosing changes. Each experiment
must preserve the correctness checks and report latency, memory and compilation
or capture costs against the reproducible baseline. Keep optimizations isolated
and optional so the minimal runtime remains readable.
