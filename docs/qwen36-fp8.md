# Qwen3.6-35B-A3B-FP8 text workflow

Step 1 prepares a pinned text-only checkpoint and provides a fresh GB10 environment.
Step 2 adds mixed-dtype HF loading, inference adapters and isolated eager
validation. The original complete 40-layer model exported and compiled with the
current kernels at capacity 1,024, and standalone C++ matched Python AOTI bitwise.
Python AOTI executes the validation matrix but fails the required comparison
against adapted eager. Benchmarks have not been established.
The subsequent native memory check found a Torch 2.13 constant-handle lifetime
leak in those original packages. Recompiled wrappers now pass native memory,
ownership, relocated-runtime parity and production CLI checks; see below.
FP8-only adaptation passes the full-model HF bitwise gate on GB10. Combined
GDN/attention adaptation executes but fails HF logit/state numerical acceptance;
that correctness issue remains open. Its fix is deferred at the user's direction
while the export/runtime pipeline is explored; functional progress must be
reported separately from HF numerical acceptance. See the
[numerical investigation and recommended fixes](experiments/qwen36-numerics.md).
The HF architecture is `qwen3_5_moe`; checkpoint preparation preserves its
quantization metadata and does not load a model or execute GPU kernels.

## Prepare on a CPU host or Mac

Use Python 3.12 and the dedicated preparation requirements. The Bash entry point
creates a new independent venv and refuses an existing destination:

```bash
bash environments/prepare-qwen36.sh --python python3.12 \
  --venv work/prepare/qwen36-venv
PREP_PY="$PWD/work/prepare/qwen36-venv/bin/python"
MODEL_DIR="$PWD/work/qwen36/35b-a3b-fp8/text"

"$PREP_PY" -m tools.prepare download \
  --model Qwen/Qwen3.6-35B-A3B-FP8 \
  --cache-dir work/qwen36/hub --output "$MODEL_DIR"
HF_HUB_OFFLINE=1 "$PREP_PY" -m tools.prepare verify --model-dir "$MODEL_DIR"
```

Requirements live in `environments/qwen36_prepare_requirements.txt`. A compatible
platform Torch wheel is sufficient for this CPU workflow. Keep the source cache
and output on a filesystem with enough space for both: the selected source files
are about 36.63 GB and prepared tensor data occupies 35.71 GB, before file headers
and environment packages. Downloads can be prefetched into the selected Hub cache.
Preparation still verifies the pinned source metadata and file hashes.

The revision is `c653bba46dfa06e9658dcc0b0dc87fe4685f14a2`. Only text tensors and
the independent `lm_head.weight` are retained. The shared source shard containing
embeddings also contains vision tensors; those tensors are filtered out. MTP
weights are excluded. Shards are processed individually, without retaining a
full-model weight dictionary or converting FP8 weights to BF16.

The published directory contains:

- `text-00001-of-00041.safetensors` through `text-00041-of-00041.safetensors`.
- `text.safetensors.index.json`: prepared tensor names and shard locations.
- `text_inventory.json`: original names/shards, dtype, shape, byte count and
  SHA256 for every tensor, plus the translated quantization exclusion list.
- `source.safetensors.index.json`: the original pinned source index.
- Original config, generation config, tokenizer files and chat template.
- `model_manifest.json`: source/prepared file hashes, dtype totals and mapping identity.

There are 62,303 text tensors and 35,707,978,496 tensor bytes: 30,970 E4M3 FP8
weights and 31,333 BF16 tensors, including 30,970 block-scale tensors. No expert
packing occurs during preparation. Text names change from
`model.language_model.*` to `model.*`; `lm_head.weight` is retained separately.
The original config remains unchanged, while `text_inventory.json` records the
corresponding prepared `modules_to_not_convert` names for the text loader.
Embeddings and the LM head remain untied. The generation config preserves both EOS
IDs, `[248046, 248044]`, rather than the single nested text-config EOS value.

Publication is atomic into a new output directory. Offline verification checks
all file and tensor hashes, expected geometry, original/prepared mappings,
quantization exclusions, generation configuration and dtype/byte totals.

## Transfer and verify on GB10

Copy the complete published directory into a fresh staging directory on GB10.
Use resumable transfer and verify before publishing that directory for loading:

```bash
# DESTINATION is user@gb10:/absolute/path/to/a/fresh/staging-directory
rsync -a --partial "$MODEL_DIR/" "$DESTINATION/"
```

Create a fresh model environment using the existing GB10 Bash entry point:

```bash
bash environments/gb10/setup.sh --model qwen36 --venv work/gb10/qwen36-venv
Q36_PY="$PWD/work/gb10/qwen36-venv/bin/python"
bash environments/check.sh --python "$Q36_PY" --target gb10 --model qwen36 \
  --output artifacts/gb10/qwen36/environment.json
HF_HUB_OFFLINE=1 "$Q36_PY" -m tools.prepare verify --model-dir "$MODEL_DIR"
```

On GB10, set `MODEL_DIR` to the transferred staging directory for verification.
After verification succeeds, rename it to the intended final checkpoint path.
The self-contained GB10 requirements pin Torch/CUDA, Triton, HF, NumPy and the
HF-compatible `kernels==0.17.2`. Existing model environments remain separate.
This model's setup/check selection currently supports GB10 only.

Preload the pinned kernel source on an online host before offline model work,
or transfer a complete, verified kernel cache:

```bash
export KERNELS_CACHE="$PWD/work/qwen36/kernel-cache"
"$Q36_PY" -c 'from huggingface_hub import snapshot_download; snapshot_download("kernels-community/finegrained-fp8", repo_type="kernel", revision="03ae31933313f38fae2b050dbe2731024386c369", cache_dir="work/qwen36/kernel-cache")'
HF_HUB_OFFLINE=1 "$Q36_PY" -c 'from kernels import get_kernel; get_kernel("kernels-community/finegrained-fp8", revision="03ae31933313f38fae2b050dbe2731024386c369", backend="cuda")'
```

Use the complete repository snapshot for offline loading with this client;
a CUDA-only partial snapshot may be rejected by the Hub cache validator.
The source is loaded before model execution, and no model inference is needed to
check this cache. Cache readiness and standalone operator packaging are separate
from the full-model export and native checks recorded below.

## Offline tokenization and CPU checks

The shared preparation CLI also supports the pinned tokenizer and non-thinking
chat template locally:

```bash
HF_HUB_OFFLINE=1 "$Q36_PY" -m tools.prepare encode --model-dir "$MODEL_DIR" \
  --prompt 'Hello.' --output work/qwen36/hello.ids
HF_HUB_OFFLINE=1 "$Q36_PY" -m tools.prepare decode --model-dir "$MODEL_DIR" \
  --input work/qwen36/hello.ids
CUDA_VISIBLE_DEVICES='' "$Q36_PY" -m unittest \
  tools.prepare.tests.test_qwen36 tools.prepare.tests.test_common \
  tools.prepare.tests.test_qwen35 tools.prepare.tests.test_qwen3 \
  environments.tests.test_setup environments.tests.test_check
```

## Load and adapt the text model

`compiler.qwen36.model.load_model` constructs the official HF
`Qwen3_5MoeForCausalLM` text model on meta, then streams the prepared shards into
its parameters. It preserves E4M3 weights and BF16 weights/scales without a
whole-model dtype cast or a dequantized expert bank. Expert IDs are packed in
numeric order; each expert's gate rows precede its up rows, with the same layout
applied to block scales. One CPU bank is packed and assigned to the destination
before the next bank is allocated. The original untied head and both generation
EOS IDs remain intact; nonpersistent RoPE buffers are recreated in FP32.

The returned model has unmodified HF forwards and official FP8 modules. Its
expert backend is HF's default `grouped_mm`. On CUDA, loading also preloads the
pinned kernel into HF's official dependency cache before forward execution.
The cache resolver is restored afterward; HF arithmetic is not replaced.

Install adapters separately when constructing the explicit-state path:

```python
import os

from compiler.qwen36.adapters import install_fp8
from compiler.qwen36.model import TextStep, load_model, state_specs, zero_state
from compiler.qwen35.adapters.delta import install_delta

# Select a library built against this model's environment before TextStep.
os.environ["XINFER_ATTENTION_OPS_LIBRARY"] = "build/qwen36/libxinfer_attention_ops.so"
model = load_model("work/qwen36/35b-a3b-fp8/text", device="cuda", num_hidden_layers=4)
fp8_counts = install_fp8(model)
install_delta(model)
step = TextStep(model)  # Installs the shared valid-prefix attention adapter.

# Use a bounded fixture explicitly; the model's declared context is much larger.
capacity = 1024
specs = state_specs(model.config, capacity)
state = zero_state(model.config, capacity)
```

`num_hidden_layers` selects a decoder prefix; omit it to load all 40 layers.
A four-layer prefix covers one complete cycle of three GDN layers and one
full-attention layer, retaining the embedding, final norm and LM head.
`install_fp8` wraps only official FP8 linear/expert modules. Routers, shared-expert
scalar gates and checkpoint-excluded BF16 projections retain their HF forwards.
All adapters retain the original parameter objects and storage pointers.
`TextStep`, `TextCache`, `state_specs` and `zero_state` reuse the Qwen3.5
implementation; there is no separate generation loop for this model.

`load_model` expects an already verified prepared directory. It checks identity,
typed inventory, finiteness, complete parameter materialization and byte budgets;
use `tools.prepare verify` for file and per-tensor SHA256 verification. Its
`checkpoint_storage` attribute reports the original and selected layer/tensor
counts, selected source bytes, packed parameter bytes by dtype and largest CPU
packed bank. Packing changes parameter names/counts, so source and destination
tensor counts need not match. Their total bytes must match. The largest packed
bank is not a measurement of total process memory.

## Run isolated eager validation

Build the attention library and native validation executable with LibTorch from
the Qwen3.6 environment. The library is needed for the adapted eager stage;
building these targets does not demonstrate full-model AOTI or C++ execution.

```bash
Q36_ENV="$PWD/work/gb10/qwen36-venv"
Q36_PY="$Q36_ENV/bin/python"
Q36_BUILD="$PWD/build/qwen36"
TORCH_PREFIX=$("$Q36_PY" -c 'import torch; print(torch.utils.cmake_prefix_path)')
CUDA_ARCH=$("$Q36_PY" -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')
CUDA_RUNTIME=$("$Q36_PY" -c 'from pathlib import Path; import torch; p = Path(torch.__file__).resolve().parent.parent / "nvidia/cu13/lib/libcudart.so.13"; assert p.is_file(), p; print(p)')
PATH="$Q36_ENV/bin:$PATH" "$Q36_ENV/bin/cmake" -S . -B "$Q36_BUILD" -G Ninja \
  -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" -DTORCH_CUDA_ARCH_LIST="$CUDA_ARCH" \
  -DCUDA_cudart_LIBRARY="$CUDA_RUNTIME" -DCUDA_CUDART_LIBRARY="$CUDA_RUNTIME" \
  -DCMAKE_BUILD_TYPE=Release -DXINFER_BUILD_VALIDATION=ON
PATH="$Q36_ENV/bin:$PATH" "$Q36_ENV/bin/cmake" --build "$Q36_BUILD" \
  --target xinfer_attention_ops xinfer_aoti xinfer_validate \
    qwen_weight_memory qwen_runtime_contract -j2
ATTENTION_LIBRARY="$Q36_BUILD/libxinfer_attention_ops.so"
```

Run each stage in a separate process, using new output directories. Only one
model bank is resident per stage. The first stage generates references from the
unmodified HF FP8 model; both adapter stages consume that same serialized
reference, protocol and checkpoint identity.

```bash
RUN="$PWD/artifacts/qwen36/step2-prefix4-001"
HF_HUB_OFFLINE=1 "$Q36_PY" -m tools.validation.qwen36.eager \
  --model-dir "$MODEL_DIR" --stage hf --layers 4 --capacity 1024 \
  --profile matrix --capture-layers 0 3 --output "$RUN/hf"
HF_HUB_OFFLINE=1 "$Q36_PY" -m tools.validation.qwen36.eager \
  --model-dir "$MODEL_DIR" --stage fp8 --layers 4 --capacity 1024 \
  --profile matrix --capture-layers 0 3 --reference "$RUN/hf" --output "$RUN/fp8"
HF_HUB_OFFLINE=1 "$Q36_PY" -m tools.validation.qwen36.eager \
  --model-dir "$MODEL_DIR" --stage adapted --layers 4 --capacity 1024 \
  --profile matrix --capture-layers 0 3 --reference "$RUN/hf" \
  --attention-library "$ATTENTION_LIBRARY" --output "$RUN/adapted"
```

The `hf` stage uses HF execution with `StaticCache`. `fp8` changes only FP8
linear/expert forwards and requires bitwise agreement with that reference.
`adapted` also enables the shared GDN and explicit-cache attention adapters;
logits, state, decoder outputs and generated tokens are required comparisons
under the existing numerical policy. A completed run and numerical acceptance
are reported separately. Inspect and retain failures when increasing the prefix
or proceeding with the current-kernel export experiment; successful execution
is not numerical acceptance.

`--profile smoke` runs a short teacher-forced sequence and the chat controls.
`matrix` additionally covers lengths 1, 2, 96, 97, 98, 511 and 512, the GDN
dispatch boundary, reset/repeat and poisoned-future-KV controls, and a 513-token
prompt split into calls of 512 and 1. A nonzero-offset case appends 98 tokens
after 96 tokens, then decodes one token. Both profiles include decode and bounded
greedy generation. The matrix needs at least 515 context slots; each invocation
still accepts at most 512 input tokens. Set `--capacity` explicitly for each
fixture and keep it identical across the three stages. This is validation state
capacity, not a compiled artifact limit or evidence that a full context window
was executed. `--layers 1 --capture-layers 0` isolates the first GDN decoder;
`--layers 40` selects the full text model. Its current HF numerical failures
remain required results rather than a prerequisite that has already passed.

Each output contains `summary.json`, `protocol.json` and per-case trace files.
The protocol fixes input IDs/partitions, EOS IDs, checkpoint identity, prefix,
capacity, captured layers and state descriptors. Summary fields include
`execution_passed`, `comparison_passed`, failures/first divergence, storage
inventories before/after adapters, per-boundary memory samples and peaks, and
request/state ownership checks. The boundary overflow check uses a synthetic
position and does not execute a full context window. GB10 CUDA and system-memory
figures overlap because memory is unified; do not add those figures together.

The GB10 full-model matrix passed 1,918 FP8-only bitwise comparisons. The combined
path completed execution and generated the same short answer, but its worst logit
errors reached 40.04% relative L2 and 54.38% peak-normalized error, exceeding the
unchanged 5% limits. It must not be treated as numerically accepted.
Registered checkpoint storage is 33.256 GiB before and after adapters. With a
16,384-slot cache, the short adapted memory probe peaked at 34.285 GiB allocated
and 34.529 GiB reserved, including validation workspaces and state clones. This
probe does not execute a complete 16K-token prompt or measure production latency.

## Export/runtime validation gates

Create complete 40-layer references for the compiled comparison. Run each
command as a separate process; all three stages use the same capacity, capture
layers and serialized inputs. Choose a new reference directory:

```bash
FULL_REFERENCE="$PWD/artifacts/qwen36/step2/full40-export-new"
ADAPTED_REFERENCE="$FULL_REFERENCE/adapted"
HF_HUB_OFFLINE=1 "$Q36_PY" -m tools.validation.qwen36.eager \
  --model-dir "$MODEL_DIR" --stage hf --layers 40 --capacity 1024 \
  --profile matrix --capture-layers 0 3 39 --output "$FULL_REFERENCE/hf"
HF_HUB_OFFLINE=1 "$Q36_PY" -m tools.validation.qwen36.eager \
  --model-dir "$MODEL_DIR" --stage fp8 --layers 40 --capacity 1024 \
  --profile matrix --capture-layers 0 3 39 \
  --reference "$FULL_REFERENCE/hf" --output "$FULL_REFERENCE/fp8"
HF_HUB_OFFLINE=1 "$Q36_PY" -m tools.validation.qwen36.eager \
  --model-dir "$MODEL_DIR" --stage adapted --layers 40 --capacity 1024 \
  --profile matrix --capture-layers 0 3 39 --reference "$FULL_REFERENCE/hf" \
  --attention-library "$ATTENTION_LIBRARY" --output "$ADAPTED_REFERENCE"
```

The adapted stage currently exits nonzero for its required HF numerical
failures. Keep that status and inspect `summary.json`: only a completed
`execution_passed: true` reference can be used below. Its failures remain
attached to the functional AOTI/native reports. An execution error or an
incomplete reference must be resolved before proceeding.

Run the complete-model compiler after building the matching attention library.
Use the same fresh bundle path for export and both validation stages:

```bash
PACKAGES="$PWD/artifacts/qwen36/35b-a3b-fp8/context1024-new"
HF_HUB_OFFLINE=1 "$Q36_PY" -m compiler.qwen36.export \
  --model-dir "$MODEL_DIR" --max-context-length 1024 \
  --output "$PACKAGES"
```

Omit `--max-context-length` to derive capacity from the model config. Production
export requires all 40 layers and has no prefix-layer flag. It retains the current
Triton GDN/FP8 and native FlashAttention implementations, with FP8 linear packing
disabled. `--enable-inductor-optimization` retains the optional combo-kernel
setting; the initial bring-up uses the default settings.
When changing compiled capacity, regenerate references with the same
`--capacity`; the validator rejects a reference/package capacity mismatch.

The guarded PyTorch workarounds live in
[`compiler/patches/`](../compiler/patches/README.md), with focused tests and
documentation of the bugs, guards and related upstream issues/fixes.

Torch 2.13's Triton access analyzer can expand shared read-dependency paths
exponentially for the grouped FP8 kernel. The compiler uses a scoped, exact-source
guarded read-DAG memoization workaround. Write counts, mutation checks and
epilogue eligibility are unchanged; the original analyzer is restored afterward.
No installed Torch files or model arithmetic change. Unsupported versions or
analyzer sources fail explicitly and require reviewing the compatibility helper.
Deployment does not execute this compiler-only workaround.

The shared compiler also uses a scoped temporary include overlay for Torch
2.13's AOTI constant-handle ownership bug. The pinned Torch version, Git revision
and original header SHA256 are checked before the overlay is enabled; installed
Torch headers are unchanged. This correction changes handle lifetime, not model
arithmetic or weight storage. Existing packages require **recompilation of both
wrappers**; rebuilding only the native executable does not fix a wrapper already
embedded in a `.pt2` file. Use a fresh bundle directory and regenerate its Python
AOTI reference before running the following gates.

The current-kernel experiment keeps these conclusions separate:

- Export must retain one dynamic prefill graph for 2–512 tokens and a static
  one-token decode graph, with explicit carried state and context bounds.
- AOTI packages must preserve FP8/BF16 checkpoint storage in one shared pool,
  without an unexplained BF16 expert bank or duplicate prefill/decode weights.
- Python AOTI must be compared with saved adapted-eager traces on identical
  tokens, partitions, EOS and state descriptors. Record bitwise agreement as
  well as the existing required numerical/token checks.
- Standalone C++ must match Python AOTI logits, state and greedy outputs
  **bitwise**, with the existing finite, ownership, masking, reset and budget
  guards. This proves runtime parity rather than HF accuracy.
- HF numerical acceptance remains open until the original HF FP8 comparisons
  pass. Existing BF16 block diagnostics remain separate and unchanged.

Validation workers run serially with per-case CPU traces, so an HF/eager
model bank does not coexist with the compiled or native bank. Compilation,
loading and execution peaks need separate measurements on GB10's unified
memory; package byte budgets alone do not establish those peaks.

The functional validator consumes an existing complete 40-layer adapted matrix
reference. Its Python worker loads only the compiled weight pool; `--model-dir`
is used to read the small `model_manifest.json` and bind the reference to the
package checkpoint. It does not load an HF model or tokenizer. Run Python and
native stages in fresh processes and output directories:

```bash
COMPILED_RUN="$PWD/artifacts/qwen36/step3/validation-new"
HF_HUB_OFFLINE=1 "$Q36_PY" -m tools.validation.qwen36.aoti \
  --stage python --packages "$PACKAGES" --model-dir "$MODEL_DIR" \
  --reference "$ADAPTED_REFERENCE" --attention-library "$ATTENTION_LIBRARY" \
  --output "$COMPILED_RUN/python"
HF_HUB_OFFLINE=1 "$Q36_PY" -m tools.validation.qwen36.aoti \
  --stage native --packages "$PACKAGES" --model-dir "$MODEL_DIR" \
  --reference "$COMPILED_RUN/python" --native "$Q36_BUILD/xinfer_validate" \
  --output "$COMPILED_RUN/native"
```

The earlier CMake commands build `xinfer_validate` with
`XINFER_BUILD_VALIDATION=ON` in the matching native build. The matrix records
34 teacher calls, 22 complete selected state
checkpoints and two greedy cases, including reset/poison and nonzero prefill.
Reports separate `functional_passed`, `native_bitwise`, preserved
`hf_numerical_reference` failures and `model_accepted: false`. A completed
reference with numerical failures remains usable for the next independent gate;
an incomplete execution, changed checkpoint, capacity, protocol or package pair
is rejected. The Python comparison uses the unchanged numerical limits and
strict token checks; the native comparison requires bitwise outputs.

## Original compiled results — 2026-11-08

The complete 40-layer model compiled on GB10 with dynamic prefill over 2–512
tokens, one-token decode and capacity 1,024. Combo kernels were disabled. Both
entry points use one shared FP8/BF16 weight archive:

| Artifact | Bytes |
|---|---:|
| `prefill.pt2` | 12,118,569 |
| `decode.pt2` | 8,457,816 |
| `weights.pt` | 35,709,190,649 |

Python AOTI completed all 34 teacher calls, 22 selected state checkpoints and
two greedy cases. Its adapted-eager comparison **failed**: 1,339 required
failures across 1,957 comparisons, with 501 comparisons bitwise equal.
There were 29 failing logit comparisons; worst logit relative L2 was 36.9162%
and peak-relative error was 48.1579%, both on the 98-token prefill. Teacher
argmax and greedy IDs all agreed. All 163 recorded T=2 comparisons were
bitwise; the T=1 sequence had one small passing recurrent-state difference.
This compiled/eager discrepancy remains open, with no root cause assigned.

The fresh standalone C++ run **passed all 1,991 comparisons bitwise**, including
raw logits, selected state and greedy outputs. It completed the same matrix
with 80 state tensors, one loader pair and no Python library loaded. Teacher
reset, KV alias/write-region and unchanged functional-input checks passed;
greedy requests retained stable KV storage. The native report preserves
`compiler_eager_parity_passed: false` and the regenerated adapted reference's
1,526 HF failures. Native parity does not make either numerical gate pass;
`model_accepted` remains false.

During Python AOTI validation, Torch peak allocation was 33.609 GiB and peak
reservation was 33.623 GiB. These are allocator measurements for this bounded
profile, including validation state checks. They are not total GB10 unified
memory, compilation peaks, a full 1,024-slot execution or a benchmark. The
largest teacher request consumed 515 positions; the model-default capacity
of 262,144 has not been validated by this run.

Raw summaries are retained under
`artifacts/qwen36/step3/python-aoti-r1/summary.json` and
`artifacts/qwen36/step3/native-aoti-r1/summary.json`. The
[numerical issue record](experiments/qwen36-numerics.md) keeps the HF adapter
issue and the additional compiled/eager discrepancy separate.

## Native memory and deployment checks

The original packages retained their approximately 35 GB GPU weight pool after
`Runner` destruction. Torch's `user_managed` branch clones an owning tensor
handle but does not assign RAII ownership to that clone. This is a lifetime
leak, separate from copying the weight data: native binding already checks that
both loaders alias the same canonical pool. A focused BF16/FP8 CPU control
reproduced reference counts `1 → 2 → 2`; the corrected header gives
`1 → 2 → 1`. The full-model results below validate cleanup after recompilation;
the CPU controls alone do not establish it.

After recompiling into a fresh bundle and regenerating its Python AOTI traces,
run the native probes serially in fresh processes:

```bash
MEMORY_RUN="$PWD/artifacts/qwen36/step4/native-memory-new"
"$Q36_BUILD/qwen_weight_memory" qwen36 "$PACKAGES" "$MEMORY_RUN/phases.json"
"$Q36_BUILD/qwen_runtime_contract" qwen36 "$PACKAGES" "$MEMORY_RUN/contract.json"
```

`qwen_weight_memory` samples CUDA free/used bytes, ATen allocated/reserved
current and peak bytes, process RSS/high-water RSS and system available memory.
Its load checkpoints distinguish CPU archive validation, the temporary overlap
of CPU archive and canonical GPU pool, loader construction, prefill/decode
binding and runtime readiness. It then samples after an eight-token prefill,
one decode and runtime destruction. Binding checks verify both loaders retain
the canonical pointers. This is a short lifetime/memory probe, not full-context
accuracy or timing validation. Boundary-sampled CUDA use is not an exact global
allocation peak; allocator peaks omit raw AOTI allocations. On GB10, CUDA,
allocator and process/system values overlap and must not be added. Reserved
allocator memory may remain cached after live tensors are released.

The rebuilt 40-layer capacity-1,024 bundle completed the native memory probe.
ATen live allocation returned exactly to the warmed **32 MiB baseline** after
runtime destruction: post-destroy allocated growth was **zero bytes**. Binding
prefill and decode left the canonical pool's allocated-byte count unchanged,
and both pointer-alias checks passed. For the eight-token prefill plus one
decode, peak ATen allocation was **33.4283 GiB** and peak reservation was
**33.4492 GiB**. These are short-probe allocator results, not total unified
memory or a full-context inference peak; cached reservation remained after
live tensors were released.

All three descriptor-lifetime cases passed: caller mutation, temporary `Spec`
and destruction of a scoped caller `Spec`. The constructor callback was
released, runtime thread/TF32 settings were applied, and the GPU-architecture
guard rejected before weight construction. All three fresh file-preflight
rejection cases also passed with CUDA hidden.

Recompilation preserved all **1,991 recorded outputs bitwise** against the
original bundle, including greedy final states. The rebuilt Python AOTI matrix
still reports 1,339 compiler/eager failures and 501/1,957 bitwise comparisons;
teacher and greedy tokens are unchanged. The installed, moved native runtime
then passed **1,991/1,991 comparisons bitwise** against the rebuilt Python AOTI
reference, with 34 teacher calls, 22 state checkpoints, two greedy cases, one
loader pair and no Python library loaded. The inherited 1,526 HF failures and
`model_accepted: false` remain intact. Correcting handle ownership did not
resolve either numerical gate.

The retained reports are `artifacts/qwen36/step4/memory-fixed-r1.json`,
`ownership-fixed-r1.json`, `files-fixed-r1.json`, `rebuilt-numerics-r2.json`,
and the `python-aoti-fixed-r1/` and `native-aoti-fixed-r1/` summaries under that
same directory. The successful production CLI report is
`artifacts/qwen36/step4/cli-fixed-r1/summary.json`.

The load observer is an optional synchronous constructor callback. It is not
retained after construction. Ordinary CLI/library construction supplies no
observer and emits no phase samples; the memory probe opts into sampling.
`qwen_runtime_contract` separately checks caller-independent `Spec` ownership,
observer release, thread/TF32 setup and preflight ordering. Its `--files-only`
mode checks fresh file-integrity preflight without allocating a GPU weight pool.

The production CLI checker consumes only saved Python AOTI traces and the small
model manifest in its controller. Use the copied/renamed installed runtime
prefix and complete five-file bundle for a deployment check, following the
[runtime installation guide](../environments/README.md#install-and-relocate-the-runtime):

```bash
NATIVE_PREFIX=/absolute/path/to/a/copied/native-prefix
DEPLOY_PACKAGES=/absolute/path/to/a/copied/complete-bundle
CLI_RUN="$PWD/artifacts/qwen36/step4/cli-new"
HF_HUB_OFFLINE=1 "$Q36_PY" -m tools.validation.qwen36.cli \
  --native "$NATIVE_PREFIX/bin/xinfer_aoti" --packages "$DEPLOY_PACKAGES" \
  --reference "$COMPILED_RUN/python" --model-dir "$MODEL_DIR" --output "$CLI_RUN"
```

It runs exactly three model-loading CLI requests: capital-of-France, a one-output
limit, and a 513-token prompt split at 512 with one output. The last reference
uses the saved prompt-end logits, not a later teacher decode. IDs, raw FP32 trace
bytes, stop reason and consumed positions must match exactly. Child processes
run from the fresh output directory with absolute paths and sanitized Python/
library-path variables. `ldd` and the capital request's loader trace check native
attention linkage and absence of Python libraries.

Eighteen rejection probes hide CUDA and require the expected pre-load error
without input/output mutation. They cover invalid IDs/lengths/budgets, retired
flags, missing values, overwrite/path collisions and three model-contract
mutations. Metadata probes copy only the small package/sidecar files and hardlink
the large weight archive. The controller never loads HF/GPU model weights and
checks CUDA remains uninitialized. `cli_passed` is separate from retained HF and
compiler/eager failures; `model_accepted` remains false.

The installed/renamed runtime and renamed complete bundle passed all three
generation cases, with IDs and all FP32 logit bytes exactly matching the saved
Python AOTI reference:

| Case | Prompt tokens | Generated IDs | Consumed positions | Stop | Logit rows |
|---|---:|---|---:|---|---:|
| `capital` | 22 | `[57590, 248046]` | 23 | EOS | 2 |
| `limit_one` | 22 | `[57590]` | 22 | Limit | 1 |
| `multicall_513` | 513 | `[9338]` | 513 | Limit | 1 |

All 18 CUDA-hidden rejection probes passed without input/output mutation.
Native linkage contained no Python library, and the controller's CUDA state
remained uninitialized. The report retains all 1,526 HF and 1,339 compiler/eager
failures with `model_accepted: false`. These results establish production CLI
parity and deployment for the bounded profile; they do not establish benchmark
performance or full-context numerical acceptance.

Run the small CPU loader/adapter regressions without loading the full checkpoint:

```bash
CUDA_VISIBLE_DEVICES='' "$Q36_PY" -m unittest \
  compiler.qwen36.tests.test_model compiler.qwen36.tests.test_adapters
```
