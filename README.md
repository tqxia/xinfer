# Qwen3-0.6B AOTInductor / C++ PoC

Step 1 establishes a reproducible eager reference. Export and the C++ runtime
are subsequent steps in [PLAN.md](PLAN.md).

The runtime interface will be token IDs. Python handles offline tokenization and
text decoding; the future C++ executable handles all model inference and generation.

## Environment

Run these commands on `kikoxia-any4.devcloud.woa.com`, port `36000`, as `root`.

```bash
cd /data/workspace/develop/qwen3-aoti-poc
```

The project `.venv` contains its own Hugging Face dependencies. Only Torch,
Triton, and their NVIDIA packages are linked from `/data/workspace/abo/.venv`.
ABO's unrelated packages are not on the project's import path. These links are
read-only dependencies by convention: do not upgrade or uninstall the shared
packages through this venv. The future C++ build must use this same Torch build.

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

The eager baseline deliberately uses HF DynamicCache. Step 2 will introduce and
validate the explicit static cache wrapper and AOTI; this baseline does not prove
that contract yet. This step contains no export, compilation, or C++ implementation.

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

`artifacts/`, `work/`, `.venv/`, and future `build/` are Git-ignored. The source
consists of `prepare.py` and `validate.py` at this stage.

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

Step 2 (static cache adapter and AOTI export) has not started.
