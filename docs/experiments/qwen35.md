# Qwen3.5 experiment record

Historical results retain their original artifact hashes. Commands below use the current source layout.

# Qwen3.5: text-only AOTI / C++ experiment

As of 2026-10-27, the primary target is **Qwen/Qwen3.5-0.8B**. Steps 1-2 pass:
single DeltaNet block export/compile, Python-free C++ execution and extended
state/sequence tests. The existing T=64/T=1 packages handle inputs of 1-512 tokens
through chunk/remainder dispatch. Shapes come from config/package metadata;
the same runner still passes with the existing 4B packages. Keep 4B for later
scaling and additional-head-layout checks. Step 3 now exports, compiles and runs
the full text model in Python. Its state semantics checks pass, but numerical
acceptance remains open: 854 output-state comparisons against HF exceed the
unchanged limits. Step 4 passes native/Python AOTI parity, with a Python-free
C++ generation loop and offline text helpers. Step 5's expanded validation and
three-profile benchmark are implemented and functionally checked. HF numerical
acceptance remains open; final timing runs belong in the user's separate idle
environment. The 0.8B experiment is kept together in one implementation commit.

## Scope and approach

Use one GPU, batch size 1, BF16 weights/activations and FP32 DeltaNet recurrent
state. Start with the upstream PyTorch implementation and small inference/state
adapters, reusing HF modules and weights. Replace selected child modules before
export when needed; avoid a full model fork or a general adapter framework.

Python handles checkpoint loading, offline tokenization/decoding, export and
compilation under `compiler/qwen35/`. C++ under `runtime/qwen35/` loads compiled
AOTI packages and performs all prefill, decode, state management and greedy
token generation. Keep Torch/LibTorch versions matched and preserve the separate
Qwen3-0.6B environment. Initial full-model limits: prompts of 1-512 tokens and a
total context budget of 1,024 tokens. Vision, quantization, serving, batching and
performance kernels are outside the initial correctness milestone.

## Implementation sequence

1. **Retarget the single DeltaNet block probe to 0.8B (complete).** Pin the checkpoint revision
   and verify its weights. Derive dimensions from its config and pass them through
   package metadata to C++; remove the current 4B constants from both sides.
   Keep one implementation usable by both sizes. Start with prefill T=64 and
   decode T=1. Repeat HF eager, adapter, Python AOTI, native C++ and FP32-control
   comparisons on outputs and both state tensors. Preserve the completed 4B
   evidence and use separate output directories for 0.8B.
2. **Establish state and sequence-length handling (complete).** Test short inputs, convolution
   boundaries (3/4/5), chunk boundaries (63/64/65 and 127/128/129), longer decode
   chains, prefill/decode handoff and request reset. Compare different valid chunk
   partitions of the same sequence. Try bounded dynamic prefill; if the HF chunk
   implementation cannot export that cleanly, use fixed 64-token chunks with
   T=1 remainder calls first. Do not advance recurrent state through padding.
3. **Export the full 0.8B text model (implemented; numerical acceptance open).** Combine embedding, all 24 decoder layers,
   final norm and LM head through a thin tensor-only adapter. Expose 18 pairs of
   convolution/recurrent states and six KV-cache pairs. Document state ownership,
   positions, causal masking and update semantics. Compile prefill and decode,
   validate them in Python and record model/shape/dtype/toolchain metadata.
4. **Add the minimal C++ generation loop (complete: native/AOTI parity).** Load both packages, allocate state
   from metadata, execute prompt prefill and repeated decode, apply greedy token
   selection, and stop at configured EOS or limits. Add the offline Python text
   helper and reject invalid inputs/package combinations. Verify a fresh native
   process runs without Python libraries.
5. **Validate end to end and measure the baseline (functional work complete; numerical acceptance and final timings open).** Compare HF eager, Python AOTI
   and C++ with identical teacher-forced tokens and greedy-generation fixtures.
   Check logits, KV/conv/recurrent state, context boundaries, resets and repeated
   requests. Require native/Python AOTI bitwise parity; retain BF16 error metrics
   and focused FP32 controls, investigating failures without silently relaxing
   limits. On an idle GPU, measure compile/load cost, package sizes, prefill
   latency, decode latency/throughput and memory against the same eager workload.
6. **Profile and optimize the working pipeline.** Once full text generation and
   correctness checks pass, refactor the profiling tools for Qwen3 and Qwen3.5.
   Profile prefill and decode separately, then evaluate Inductor options, native
   CUDA Graphs and suitable existing DeltaNet/convolution/attention kernels.
   Prefer small module adapters for custom-op substitution. Validate and measure
   each change independently against the preserved baseline. Finally rerun 4B
   to test scaling and its different DeltaNet head layout.

## Step 5: end-to-end validation and benchmark

Functional work is complete on the shared L20 (2026-10-27). Numerical acceptance
is **not passed**, and no performance baseline is claimed. The user will run
the benchmark in a separate environment. Production model packages and their
precision settings are unchanged; no optimization was introduced in this step.

### Validation result

`validate_end_to_end.py` compares original HF eager with both natural full-prompt
calls and the compiled path's 64/1 partitions, Python AOTI, and a fresh native
process. `reference.py` keeps the independent HF cache/mask handling in one small
helper shared by validation and benchmarking. The native validation driver also
accepts poisoned future KV slots to check that masking isolates them.

The 21 native cases include teacher-forced prompt lengths 1/2/63/64/65/127/128/129/
511/512, eight subsequent decode steps (64 for length 512), the last context
slot, poisoned future KV, reset/repeat, and three ordinary chat prompts. A
separate original-weight FP32 model covers 73 steps: length 512, the context
boundary and the France teacher-forced sequence.

| Check | Result |
| --- | --- |
| C++ vs Python AOTI teacher-forced outputs/state | All 9,947 tensors across 203 steps bitwise equal |
| C++ vs Python AOTI greedy generation | IDs, 22 logit rows and 144 final state tensors bitwise equal |
| AOTI vs HF with matching partitions | 203/203 teacher-forced top-1 predictions; 3/3 greedy sequences equal |
| AOTI vs HF with full-prompt calls | 202/203 top-1 predictions; 2/3 greedy sequences equal |
| Masked future KV, reset/repeat, input preservation, consumed positions, context overflow | Passed |
| Native process Python-library check | Passed |
| Unchanged 5% numerical limits | Failed for conv/recurrent states; logits and K/V pass |

There are 3,466 failing state comparisons against matched-partition HF (3,451
conv and 15 recurrent), and 3,500 against full-prompt HF (3,451 conv and 49
recurrent). These are comparison entries, including repeated fixtures, not
6,966 distinct tensors. The validator retains all metrics and exits **1**.

| Worst error over checked groups | AOTI vs HF chunked | AOTI vs HF full prompt |
| --- | ---: | ---: |
| Logit relative L2 | 3.1356% | 3.2270% |
| Recurrent-state relative L2 | 4.6169% | 5.7811% |
| Recurrent-state peak-relative | 9.1719% | 9.2499% |

The large conv relative errors remain dominated by near-zero reference channels;
the limits have not been relaxed. On the 73 FP32-control steps, worst logit
relative L2 is 2.3753% for AOTI and 2.6459% for HF. These diagnostics neither
close Q35-NUM-001 nor establish general accuracy superiority.

Evidence: `artifacts/qwen35/0.8b/end-to-end/` contains `summary.json`, fixtures,
native dumps, source snapshots/hashes and final HF/AOTI/FP32 context-state dumps.
Step-4 binaries and sources are preserved in `work/qwen35/0.8b/step4-source/`,
because the native validator was rebuilt to support the poison checks.

### Open issue Q35-NUM-002: partition-sensitive greedy decision

For the France prompt, AOTI and matched-partition HF generate `Paris`, while
full-prompt HF generates `Paris.` (surrounding whitespace/special tokens omitted).
At generated token index 1, AOTI chooses `248046` (end of message), and full-prompt
HF chooses `13` (period). HF's top-two margin and its score gap to token 248046
are both **zero**. HF itself exhibits this partition difference, so it cannot
be attributed exclusively to compilation. Arithmetic and Chinese fixtures match
on every path. This remains an exact-greedy acceptance failure; no tie exception
or tolerance change was added. The retained report records token IDs and margins.

The separate fixed-token benchmark also exposes one AOTI/HF difference after
consuming `[248045, 57590]`: AOTI and FP32 choose `11`, HF BF16 chooses `248046`,
with HF score gap 0.0625. C++ and Python AOTI predictions agree. Logit relative
L2 is 2.0491% AOTI/HF, 1.4315% AOTI/FP32 and 1.3691% HF/FP32. This is a different
fixture from the partition tie above, and its precise cause is not isolated.
Evidence is `benchmark-prediction-check.json` under the same artifact root;
the focused reproducer is `work/qwen35/0.8b/check_predictions.py`. Matching one
FP32 token is not sufficient to dismiss a BF16 parity failure.

### Benchmark contract and functional checks

`benchmark.py` runs fresh processes for three profiles, rotating their order
over three rounds by default:

| Profile | Computation |
| --- | --- |
| `hf` | Original HF model with one full-prompt call, then single-token decode |
| `hf_chunked` | Original HF model with the same 64/1 dispatch as AOTI |
| `cpp_aoti` | `runtime/qwen35/benchmark.cpp`, using the generation CLI's `Runtime` |

All profiles use identical prompt and teacher-forced continuation IDs. EOS is
ordinary input data here, so every profile performs the same number of calls.
The timed prediction includes argmax and its host read. Tokenization, model
loading and report writing are outside steady-state request timing. One CPU
thread is used and TF32 is disabled. No CUDA Graphs or custom kernels are enabled.

Report fields separate reset, prefill, reset-plus-prefill (`first_prediction_ms`),
decode per token and total request time. HF lazily allocates state during prefill;
native state allocation is charged to reset. Compare `first_prediction_ms` when
including both costs. CUDA initialization, load and each case's first request
are retained separately from warmup/measured samples; only the first case starts
with a freshly loaded process. HF load includes checkpoint checks/construction
and transfer; native load includes both packages and initial state allocation.
Metadata retains package size and observed export/compile time; this is not a
controlled cold-cache compilation benchmark.

Memory reports distinguish allocator allocated/reserved peaks from sampled
device-wide usage. The latter includes other processes and can miss transient
peaks. Two native loaders retain independent package weight pools. Reports also
save the exact workload, source snapshots, binary/package hashes, environment,
per-worker GPU telemetry and the matching end-to-end acceptance report.
Prediction differences between profiles are recorded; within-profile repeats
must match. Functional success never changes failed numerical acceptance to pass.

The shared-host functional matrix passes all three profiles at lengths
1/22/65/128/512, 64 decode steps, one warmup and two measured repeats. A second
matrix covers lengths 64/511 and all three process orders, with physical GPU UUID
selection. Together these execute 132 requests. Of 65 predictions per case,
AOTI matches chunked HF on 64 at length 1 and all 65 at the other tested lengths.
Full-prompt HF also differs at length 22 (64/65 agreement). All results remain
`provisional_shared_gpu`; their latency values are not a speedup claim.

Evidence: `benchmark-functional/` and `benchmark-order-check/` under the artifact
root. The final script also passes a nine-request, three-profile rerun in
`benchmark-final-smoke/`. An idle-guard variable-shadowing bug found during validation was fixed;
`benchmark-idle-check-v2.json` records the intended busy-GPU rejection after 30
samples. By default the runner requires five consecutive utilization samples
at or below 10% before and after each worker. This checks run boundaries, not
exclusive GPU ownership. `--allow-busy` explicitly marks a functional-only run.

### Run in the benchmark environment

Use the sources from this experiment commit as well as the pinned prepared text directory
`work/qwen35/0.8b/text/` (weights, config, tokenizer and manifest). Alternatively,
run `compiler/qwen35/model.py` there to prepare it. The manifest is verified
when loading. Create an isolated environment with the pinned dependencies in
`requirements.txt` and a CUDA-enabled Torch installation; the development host's
`.pth` shortcut is specific to that host. Optional `fla` and `causal_conv1d` must
remain absent for this baseline. If the target GPU needs a newer Torch/toolchain,
treat that as a new configuration and rerun correctness before interpreting times.

Re-export on the target GPU/toolchain and rebuild against that same Torch's
LibTorch. The existing SM89 packages/binaries are not a portable deployment bundle.
Run these commands from the repository root; use fresh output paths on reruns:

```bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
QWEN_PY=work/qwen35/venv/bin/python
export TORCHINDUCTOR_CACHE_DIR="$PWD/work/qwen35/0.8b/inductor-target"
"$QWEN_PY" -m compiler.qwen35.export decode --output artifacts/qwen35/0.8b/text-target
"$QWEN_PY" -m compiler.qwen35.export prefill --output artifacts/qwen35/0.8b/text-target

QWEN_CMAKE_PREFIX=$("$QWEN_PY" -c 'import torch; print(torch.utils.cmake_prefix_path)')
QWEN_SM=$("$QWEN_PY" -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')
cmake -S . -B work/qwen35/build-target -G Ninja \
  -DCMAKE_PREFIX_PATH="$QWEN_CMAKE_PREFIX" -DTORCH_CUDA_ARCH_LIST="$QWEN_SM" \
  -DCMAKE_BUILD_TYPE=Release -DQWEN_BUILD_QWEN35=ON \
  -DQWEN_BUILD_VALIDATION=ON -DQWEN_BUILD_BENCHMARK=ON
cmake --build work/qwen35/build-target --target qwen35_aoti qwen35_validate qwen35_benchmark -j2

"$QWEN_PY" -m compiler.qwen35.tests.validate_end_to_end \
  --packages artifacts/qwen35/0.8b/text-target \
  --native work/qwen35/build-target/qwen35_validate \
  --output artifacts/qwen35/0.8b/end-to-end-target
```

Inspect the validation report even if it exits 1 for the known numerical issues.
The benchmark copies that failed status into its own report; do not interpret
functional timing execution as numerical acceptance. With an otherwise idle GPU:

```bash
"$QWEN_PY" -m compiler.qwen35.benchmark run --gpu 0 \
  --packages artifacts/qwen35/0.8b/text-target \
  --native work/qwen35/build-target/qwen35_benchmark \
  --validation artifacts/qwen35/0.8b/end-to-end-target/summary.json \
  --output artifacts/qwen35/0.8b/benchmark-target
```

Defaults: prompt lengths 22/64/65/128/511/512, 64 decode steps, two warmups, five
measured repeats and three rounds. `--gpu` selects a physical `nvidia-smi` index
or UUID; each worker exposes only that GPU as CUDA device 0, and telemetry uses
the same UUID. Keep export/validation on that GPU too. For a short harness-only
check, add `--lengths 22 65 --decode-steps 4 --warmup 1 --repeats 2 --rounds 1
--allow-busy` and choose another output directory. Final measurements must omit
`--allow-busy` and retain the whole result directory, not only `summary.json`.

## Step 3: full text model

`text_model.py` extracts the original text weights from the pinned 0.8B
checkpoint, checks the source shard against its Hub SHA256, and loads all 24 HF
decoder layers, embeddings, final norm and tied LM head. There are 752,393,024
text parameters. Vision and MTP weights are excluded. Weights and activations
use BF16; RoPE frequency buffers and DeltaNet recurrent states remain FP32.
The original HF modules perform all model arithmetic. The adapter only supplies
tensor state, positions, a causal mask and the last-token logits interface.

```text
step(input_ids, cache_position, state) -> last_logits, *next_state
```

- Batch size is one. `input_ids` is contiguous CUDA int64 `[1,T]` and
  `cache_position` is contiguous CUDA int64 `[T]`. T is exactly 64 for prefill
  or one for decode. Positions are consecutive append positions starting at zero
  for each request; the caller must enforce the 1,024-token capacity.
- `state` is a list of 48 contiguous CUDA tensors, two per decoder layer in
  layer order. Each of 18 DeltaNet layers receives BF16 conv `[1,6144,4]` and
  FP32 recurrent `[1,16,128,128]` tensors. Each of six full-attention layers
  receives BF16 K/V `[1,2,1024,256]` tensors. State occupies 30.84375 MiB.
- Zero state starts a request. The interface preserves its inputs; returned
  state becomes the next call's input. Conv storage is cloned and K/V updates
  use out-of-place `index_copy`. This deliberately permits allocation and cache
  copies for the initial correctness baseline.
- Full attention uses the original HF SDPA path with the memory-efficient CUDA
  backend and an explicit additive causal mask over all 1,024 cache slots.
  Query position `p` can read only slots `0..p`. DeltaNet receives no padding
  mask. Text positions use identical coordinates on all three RoPE axes.
- Output zero is BF16 last-token logits `[1,248320]`, followed by the 48 next
  states. The flattened AOTI/C++ ABI therefore has 50 inputs and 49 outputs.
  Package metadata records the order, dimensions, dtypes and model/toolchain
  provenance. No custom operator library is required.

`export_text.py` strictly exports and AOT-compiles each fixed entry point into
a `.pt2` package. `validate_text.py` compares the original HF model with its
own StaticCache, the eager adapter, loaded AOTI packages and an independent
FP32 reference. A separate HF DynamicCache run checks fixed-capacity attention
against attention over the valid prefix. Six teacher-forced cases cover 50
steps: decode from zero, 64-token prefill plus decode, a 65-token remainder,
two prefills, a 512-token prefix plus decode, and request reset/repeat.

```bash
# Once, after the existing step-1 environment setup:
work/qwen35/venv/bin/python -m compiler.qwen35.model
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/work/qwen35/0.8b/inductor-text"
work/qwen35/venv/bin/python -m compiler.qwen35.export decode
work/qwen35/venv/bin/python -m compiler.qwen35.export prefill
work/qwen35/venv/bin/python -m compiler.qwen35.tests.validate_model
```

Preparation refuses an existing model directory; export refuses an existing
package; validation refuses an existing result directory. Use fresh `--output`
directories for reruns and `--packages` to select another compiled pair.
The validator exits nonzero when numerical acceptance fails and retains its
report. Full-model native execution, greedy generation and context-limit tests
are covered by the step-4 result below; step 3 alone does not establish those
results or GPU performance.

### Observed result — 2026-10-27

Strict export and AOTI compilation succeed without graph breaks or graph-signature
input/buffer mutations. An initial attempt to construct `SimpleNamespace` inside
the traced adapter was rejected; using an ordinary small `LayerState` class
resolved it without changing any model arithmetic.

| Entry point | Exported nodes | Export time | Compile time | Package bytes |
| --- | ---: | ---: | ---: | ---: |
| Decode T=1 | 3,605 | 6.60 s | 59.87 s | 1,506,903,652 |
| Prefill T=64 | 19,211 | 27.31 s | 149.98 s | 1,509,608,510 |

Both files are compiled AOTI packages, about 1.4 GiB each, with separate weight
pools when both are loaded. Build times are observations on a shared host, not
controlled compiler or inference benchmarks.

| Check | Result |
| --- | --- |
| Eager adapter vs original HF StaticCache | All 2,450 logits/state tensors bitwise equal |
| HF DynamicCache vs StaticCache control | All 441 tensors bitwise equal |
| AOTI vs HF next-token argmax | 50/50 teacher-forced steps agree |
| AOTI logits and all full-attention K/V states | Pass unchanged 5% limits |
| AOTI conv/recurrent states | **Fail:** 850 conv and four recurrent tensors exceed limits |
| Poisoned future KV ignored; past/future KV write regions preserved | Passed |
| Inputs preserved; finite values, shapes/dtypes; request reset/repeat | Passed |

The validator compares AOTI against both HF and the identical eager adapter, so
its report contains 1,708 failure entries for those 854 distinct tensors. It
exits with status 1. Top-1 agreement under teacher forcing does not establish
greedy generation parity or close cache-state acceptance.

Worst AOTI/HF relative L2 is 1.3690% for logits, 1.8532% for keys, 2.4940% for
values and 4.4172% for recurrent states. Recurrent peak-relative error reaches
7.2500%. Conv comparisons use the existing per-channel, four-value metric;
near-zero channels can have very large relative errors. HF itself also has
large per-channel relative errors against FP32. This does not waive the test:
all raw errors and unchanged limits remain recorded. Worst logits relative L2
against the independent FP32 reference is 1.3995% for AOTI and 1.3042% for HF.

The diagnostic `--eager-numerics` option retains native `aten.mean.dim` and
disables FP contraction, as in the Qwen3 experiment. A complete second run still
fails the same 854 tensors; this option is **not** an accepted correctness fix.

```bash
export TORCHINDUCTOR_CACHE_DIR="$PWD/work/qwen35/0.8b/inductor-text-eager"
work/qwen35/venv/bin/python -m compiler.qwen35.export decode --eager-numerics \
  --output artifacts/qwen35/0.8b/text-eager-numerics
work/qwen35/venv/bin/python -m compiler.qwen35.export prefill --eager-numerics \
  --output artifacts/qwen35/0.8b/text-eager-numerics
work/qwen35/venv/bin/python -m compiler.qwen35.tests.validate_model \
  --packages artifacts/qwen35/0.8b/text-eager-numerics \
  --output artifacts/qwen35/0.8b/text-eager-validation
```

### Open issue Q35-NUM-001: lost BF16 rounding boundary

Status: **open**, recorded on 2026-10-27. Preserve this issue while implementing
the native runtime; C++/Python AOTI parity does not resolve HF numerical acceptance.
The affected environment is Torch 2.7.0+cu126, Triton 3.3.0, Transformers 5.20.2
and NVIDIA L20 SM89, using the pinned 0.8B checkpoint above.

Focused layer-0 decode probes locate the first BF16 activation difference at
the DeltaNet output: embeddings and input RMSNorm are exact; 588/1,024 DeltaNet
output elements differ (relative L2 0.2629%). Retaining native sum reductions
makes that probe's recurrent state exact but leaves its output difference.
An intermediate-output probe then identifies a lost rounding boundary before
gated RMSNorm: the BF16 core output agrees, but its FP32 conversion uses the
unrounded producer values. All 2,048 converted values differ; rounding the
compiled FP32 values through BF16 reproduces eager exactly. The discrepancy
persists with `emulate_precision_casts=True` and with pattern matching disabled.
Smaller FP32 differences also occur in exponential/softplus/SiLU operations.
This isolates a contributing cause, not every full-model numerical discrepancy.

Forcing native `aten._to_copy` in the focused probe compiles but its package
fails to load with `[json.exception.type_error.302] type must be string, but is
number` in this Torch 2.7 environment. That experimental workaround is not in
the implementation. The exact compiler pass responsible has not been isolated;
the failed pattern-matcher control is not evidence that a particular graph pass
caused it. No production model fork, graph rewrite or custom kernel has been
added to hide these failures.

The retained `probe_rounding.py` makes this observation reproducible. It uses
the original layer-0 DeltaNet with the normalized embedding of token `The`,
zero state, native mean/sum reductions and disabled FP contraction. It exposes
intermediate outputs only in a diagnostic graph and compares execution before
and after AOTI compilation with `emulate_precision_casts=True`.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/work/qwen35/0.8b/inductor-layer-probe"
work/qwen35/venv/bin/python -m compiler.qwen35.diagnostics.probe_rounding
```

The recorded run in `artifacts/qwen35/0.8b/rounding-probe/` confirms:

| Boundary check | Result |
| --- | --- |
| BF16 input to gated RMSNorm (`reshape_4`) | Exact |
| Conversion back to FP32 (`to_11`) | 2,048 / 2,048 elements differ |
| Maximum absolute difference at that conversion | 0.000015905126929283142 |
| Compiled FP32 values rounded through BF16 vs eager FP32 values | Exact |

`summary.json` records package, checkpoint, source and HF implementation hashes;
`boundary.pt` holds the four compared tensors; `graph.py` and `probe.pt2` retain
the diagnostic graph and compiled package. Exit zero means the known issue was
reproduced, **not** that numerical acceptance passed. Use a fresh `--output` on
reruns. This is evidence of a missing rounding boundary, not proof that every
full-model discrepancy shares one cause or that eager BF16 is closer to FP32.

Next investigation: isolate explicit dtype-conversion lowering/fusion in a small
model-independent reproducer, then test a targeted compiler fix or a pinned newer
toolchain. Any candidate must rerun the complete HF/AOTI/native checks without
relaxing the existing limits. This investigation remains separate from step 4.

Evidence is retained under `artifacts/qwen35/0.8b/text/`, `text-validation/`,
`text-eager-numerics/` and `text-eager-validation/`. The validation reports contain
all per-step metrics, package/checkpoint/source hashes and final HF/AOTI/FP32
state dumps. Focused diagnostic scripts/logs are in `work/qwen35/0.8b/probe_*`
and `probe-*.log`; their packages/value dumps are under the same artifact root.
The initial source snapshot is in `work/qwen35/0.8b/text-initial-source/`.

## Step 4: native text generation

`runtime/qwen35/runtime.h` contains the metadata contract and a small
stateful runtime. `text_main.cpp` supplies the token-ID command line. Both model
packages load once. Each request starts from zero state and absolute position
zero; complete 64-token chunks use prefill and the remaining tokens use decode.
The loop selects the largest logit, returns only newly generated token IDs,
and stops at the package's configured EOS or `--max-new-tokens`.

The caller owns the returned conv/recurrent/KV tensors and passes them into the
next package call. No hidden tokens are padded. The last predicted token is not
fed back after stopping, so consumed positions equal prompt length plus generated
length minus one. The conservative budget check requires prompt length plus the
requested generation limit to fit within 1,024 tokens. Prompts contain 1-512 IDs.

Before loading weights, the CLI validates the request and package metadata:
entry points, state order/shapes/dtypes, compatible checkpoint and compiler
settings, LibTorch version, C++ ABI and file sizes. GPU architecture is checked
before constructing loaders. Metadata and its corresponding `.pt2` files must
stay together; the Python validator additionally verifies package SHA256s.
Existing output/trace files are refused. Optional `--trace` writes raw FP32
logit rows, one vocabulary row per generated token.

`prepare_text.py` applies the checkpoint's official non-thinking chat template
offline and writes whitespace-separated IDs. It explicitly requests a list from
Transformers 5's chat-template API. Its decode command reads generated IDs and
prints text; neither helper performs model inference. The native executable
does not tokenize, import Python, embed an interpreter or call a Python process.

```bash
/data/workspace/abo/.venv/bin/cmake -S . -B work/qwen35/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/data/workspace/abo/.venv/lib/python3.12/site-packages/torch/share/cmake \
  -DQWEN_BUILD_QWEN35=ON -DQWEN_BUILD_VALIDATION=ON
/data/workspace/abo/.venv/bin/cmake --build work/qwen35/build \
  --target qwen35_aoti qwen35_validate -j2

work/qwen35/venv/bin/python -m compiler.qwen35.prepare encode \
  --prompt 'What is the capital of France? Answer briefly.' \
  --output work/qwen35/0.8b/demo/prompt.ids
work/qwen35/build/qwen35_aoti artifacts/qwen35/0.8b/text \
  work/qwen35/0.8b/demo/prompt.ids work/qwen35/0.8b/demo/generated.ids \
  --max-new-tokens 24
work/qwen35/venv/bin/python -m compiler.qwen35.prepare decode \
  --input work/qwen35/0.8b/demo/generated.ids

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 work/qwen35/venv/bin/python \
  -m compiler.qwen35.tests.validate_runtime
```

`runtime/qwen35/tests/text_validate.cpp` is a separate driver using the same
runtime. It tests multiple requests with one loader pair, captures tensors,
checks input preservation and inspects process mappings for Python libraries.
The Python validator compares native tensors bitwise with Python-loaded AOTI,
checks the CLI independently, and exercises request/package rejection paths.
Its report distinguishes native runtime acceptance from the still-open HF
numerical acceptance. Step 5 will extend HF end-to-end comparisons and measure
performance on an idle GPU; this runtime validation is not a benchmark.

### Step 4 result — 2026-10-27

The 18-case matrix uses one loader pair across requests. It includes all six
step-3 teacher-forced cases, native dispatch of 2/63/65/129/511-token requests,
the final legal position 1,023 followed by a rejected overflow, three chat
prompts, a repeated chat, a one-token generation limit, and a test-only EOS
override that stops on the first prediction. Actual model EOS is also reached
in all three ordinary chat fixtures.

| Check | Result |
| --- | --- |
| Teacher-forced C++ vs Python AOTI | All 3,136 tensors across 64 checked steps bitwise equal |
| Greedy C++ vs Python AOTI | All 28 logit rows, generated IDs and 288 final state tensors bitwise equal |
| Request reset/repeat, state handoff, consumed positions, input preservation | Passed |
| Last legal context slot and overflow rejection | Passed |
| EOS and generation-limit stopping | Passed |
| CLI invoked independently with Python environment removed | IDs and logits bitwise equal |
| `ldd` and native `/proc/self/maps` checks | No `libpython` or `libtorch_python` |
| Invalid request/package rejection checks | All 18 passed |

Rejection checks cover empty/oversized prompts, malformed/negative/out-of-range
IDs, invalid or overflowing generation budgets, output overwrite, mixed
checkpoint/state/numerical metadata, wrong entry-point length, Torch version,
state dtype, tensor ABI order and GPU architecture. Invalid requests produce
no output file, and the overwrite check preserves its existing input file.

The standalone CLI's France prompt uses 22 input tokens and returns four tokens,
including configured EOS 248044; it consumes 25 positions. Offline decoded
examples, excluding special tokens and surrounding whitespace:

| Prompt | Native output |
| --- | --- |
| What is the capital of France? Answer briefly. | Paris |
| What is 17 + 25? Answer with only the number. | 42 |
| 用一句话解释什么是编译器。 | 编译器是将源代码翻译成机器码的自动化过程。 |

Evidence is in `artifacts/qwen35/0.8b/runtime-validation-v2/`: `summary.json`,
fixture definitions, native state/logit dumps, CLI and rejection logs, linkage
reports, source/binary/package hashes and decoded results. A separate CLI smoke
run is in `artifacts/qwen35/0.8b/runtime-smoke/`. The initial helper attempt
exposed Transformers 5's dictionary return default; `return_dict=False` fixes
the helper, and the successful matrix uses the corrected integer token lists.
The original model packages and the step-3 failure reports are unchanged.

The report explicitly sets `native_runtime_passed: true` while keeping
`hf_numerical_acceptance: open`. This result does not close Q35-NUM-001 or the
854 step-3 HF state failures. The GPU remained busy; no performance baseline is
claimed. The 0.8B implementation and retained diagnostics belong to one experiment commit.

## Step 2 result: sequence and state handling

### Dynamic export probe and selected path

`probe_dynamic.py` tries strict export with a sample T=32 and bounded dynamic
ranges T=2..512 and T=2..64. Both fail with `UserError: Constraints violated
(tokens)` in Torch 2.7 / Transformers 5.20.2. Guards generated from HF's modulo
padding, reshapes and sequential chunk scan ultimately require `tokens = 32`.
The single-chunk range also fails; a smaller bound alone does not resolve it.
This is a dynamic-shape export limitation. Fixed T=64/T=1 still exports and
compiles as full graphs, and its existing packages are reused without changes.

The minimal baseline uses `sequence.py::run_sequence` and the equivalent C++
function: consume each complete 64-token chunk with prefill, then consume the
remaining 0-63 tokens through decode. For example, T=129 is 64+64+1 and T=63 is
63 decode calls. Each call passes its returned conv/recurrent state to the next;
all token outputs are concatenated. External hidden-token padding is never added.
This trades extra launches on short/tail inputs for a small, explicit contract.
No speedup is claimed; dynamic export or additional buckets can be revisited
during optimization if profiling justifies them.

### Validation

`validate_block.py --sequence-tests` covers 18 cases and 666 checked sequence
steps, dispatching 34 prefill and 918 decode package calls per compiled path:

- Prefix lengths 1/2/3/4/5, 63/64/65, 127/128/129 and 511/512, each followed by
  four decode steps. These cover convolution and chunk boundaries.
- The same 65-token prefix as one request versus 65 single-token requests;
  the same 129-token prefix as one request versus 1+64+64 and 3+61+65 requests.
  Compare all token outputs and both states at the prefix and final boundaries.
- A 512-token prefix followed by 512 decode steps; a repeated 65-token request
  after that long request verifies reset and reuse across different workloads.

HF runs twice: once with the supplied request lengths, and once partitioned
exactly like the compiled path. This separates partition-induced BF16 differences
from adapter and compiler differences. The original checkpoint also runs in FP32.

| Check | Result |
| --- | --- |
| C++ vs Python AOTI | All 1,998 output/state tensors bitwise equal |
| Adapter vs HF with matching partitions | All 1,998 tensors bitwise equal |
| AOTI vs HF supplied request lengths | All unchanged 5% relative limits pass |
| Alternate partitions | All BF16 comparisons pass; FP32 diagnostics retained |
| Input preservation, shapes/dtypes, finite values, reset/repeat | Passed |
| Native Python-free linkage/loading | Passed |
| Metadata and request rejection checks | Seven passed |

Worst errors across all 666 checked steps:

| Tensor | AOTI vs HF relative L2 | AOTI vs HF peak-relative | AOTI vs FP32 relative L2 | HF vs FP32 relative L2 |
| --- | ---: | ---: | ---: | ---: |
| Output | 0.9043% | 2.1834% | 1.1041% | 1.1535% |
| Convolution state | 0.5451% | 0.7519% | 0.3849% | 0.3849% |
| Recurrent state | 0.6470% | 0.6414% | 1.0847% | 1.0847% |

With matching partitions, compiled convolution states are bitwise equal to HF;
worst recurrent-state relative L2 is 0.0005064%. HF itself shows BF16 differences
when partitioning changes, so cross-partition equality is numerical rather than
bitwise. The unchanged per-token/per-channel/per-head limits apply. FP32 controls
are diagnostic, and all metrics remain in `summary.json`.

Native checks reject empty/oversized requests and multi-token decode requests,
in addition to mismatched hidden/state dimensions, wrong dtype and wrong strides.
The updated runner also passes the original 4B matrix (25 calls / 75 tensors and
seven rejections) with the preserved 4B packages.

```bash
# Diagnostic probe: inspect each result's exported field, not just process exit.
work/qwen35/venv/bin/python -m compiler.qwen35.diagnostics.probe_dynamic
/data/workspace/abo/.venv/bin/cmake --build work/qwen35/build --target qwen35_block -j2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 work/qwen35/venv/bin/python \
  -m compiler.qwen35.tests.validate_block --sequence-tests --binary work/qwen35/build/qwen35_block
```

Evidence is under `artifacts/qwen35/0.8b/dynamic-prefill/` (including complete
constraint errors), `artifacts/qwen35/0.8b/sequence-validation/` and
`artifacts/qwen35/4b-sequence-regression/`. Commands require the step-1 setup and
packages below. Use fresh `--output` directories for reruns. Step-1 packages and
results remain unchanged; its runner/source snapshot is in
`work/qwen35/0.8b/step1-source/`.

## Step 1 result: 0.8B single block

The probe uses the original layer-0 DeltaNet weights and HF forward, with a small
`TensorState` adapter. The independent eager reference uses HF `DynamicCache`.
The block includes projections, depthwise convolution, the gated delta rule,
gated normalization and output projection; surrounding MLP/residuals and all
other layers are outside this test.

```text
step(hidden, conv, recurrent) -> output, next_conv, next_recurrent
```

| Tensor | Shape | Dtype |
| --- | --- | --- |
| hidden / output | `[1, T, 1024]` | BF16 |
| conv / next_conv | `[1, 6144, 4]` | BF16 |
| recurrent / next_recurrent | `[1, 16, 128, 128]` | FP32 |

Prefill uses fixed T=64; decode uses T=1. Inputs are contiguous CUDA tensors.
Inputs are preserved and returned state tensors become the next call's inputs.
Zero state begins each request. The adapter clones convolution storage to isolate
HF's in-place update. This functional baseline permits allocations/copies.

- Pinned [checkpoint](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17):
  `2fc06364715b967f1860aea9cf38778875588b17`. The downloaded weight shard's SHA256
  matched the Hub record; the extracted block and config have recorded checksums.
- Torch `2.7.0+cu126`, Triton `3.3.0`, Transformers `5.20.2`, NVIDIA L20 SM89;
  same isolated environment and native toolchain as the original 4B probe.
- Strict export succeeds without graph breaks or graph-signature mutations:
  decode 127 nodes, prefill 994 nodes. Both `.pt2` files contain compiled AOTI
  code. No HF arithmetic changes, custom kernels or numerical overrides beyond
  `emulate_precision_casts=True` and disabled TF32 were needed.

| Entry point | Export time | Compilation time | Package size |
| --- | ---: | ---: | ---: |
| Decode | 0.51 s | 14.28 s | 20.67 MiB |
| Prefill | 1.68 s | 30.32 s | 22.22 MiB |

These are observed build times on a shared host, not inference benchmarks.
Four cases cover 25 calls / 75 output-and-state tensors: prefill plus eight
decode steps, four decode steps from zero state, consecutive prefills followed
by decode, and a repeated request. Inputs are seeded synthetic hidden states.

| Check | Result |
| --- | --- |
| Adapter vs HF DynamicCache | All 75 tensors bitwise equal |
| C++ vs Python AOTI | All 75 tensors bitwise equal |
| AOTI vs HF BF16, unchanged 5% relative limits | Passed |
| State reset/repeat, input preservation, shapes/dtypes, finite values | Passed |
| Python-free native loading (`libpython` and `libtorch_python`) | Passed |
| Native rejection of mismatched hidden/state shapes, wrong dtype/stride | All four passed |

Worst observed numerical errors, using the same per-token/per-channel/per-head
metrics described in the historical result below:

| Tensor | AOTI vs HF relative L2 | AOTI vs HF peak-relative | AOTI vs FP32 relative L2 | HF vs FP32 relative L2 |
| --- | ---: | ---: | ---: | ---: |
| Output | 0.5736% | 1.4881% | 1.0391% | 1.0801% |
| Convolution state | 0% | 0% | 0.3847% | 0.3847% |
| Recurrent state | 0.0000903% | 0.0000735% | 1.1241% | 1.1241% |

FP32 is a diagnostic control using checkpoint values loaded into FP32 and the
same BF16 input fixtures promoted to FP32. Its metrics are retained for both HF
and AOTI; eager parity is not a claim of improved FP32 accuracy.

Evidence: `artifacts/qwen35/0.8b/block/` and `artifacts/qwen35/0.8b/validation/`.
The latter contains `summary.json`, native traces/linkage and rejection logs.
`work/qwen35/0.8b/` holds the extracted weights, compilation cache and logs.
The 4B compatibility run in `artifacts/qwen35/4b-config-regression/` also passes
25 calls / 75 tensors and all four rejections, using the preserved 4B packages.
No full-model generation or inference-performance result is claimed here.

### Reproduce 0.8B

Run from `/data/workspace/develop/qwen3-aoti-poc`. The isolated environment uses
the existing Torch installation read-only through a `.pth` file; the original
`.venv` and ABO packages are not upgraded. Skip environment/build configuration
if already set up. Optional `fla` and `causal_conv1d` packages must be absent so
eager and compiled runs use the documented PyTorch baseline.

```bash
.venv/bin/python -m venv work/qwen35/venv
work/qwen35/venv/bin/python - <<'PY'
from pathlib import Path
site = Path('work/qwen35/venv/lib/python3.12/site-packages')
(site / 'torch-base.pth').write_text('/data/workspace/abo/.venv/lib/python3.12/site-packages\n')
PY
work/qwen35/venv/bin/python -m pip install -r compiler/qwen35/requirements.txt
work/qwen35/venv/bin/python -m pip check

work/qwen35/venv/bin/python -m compiler.qwen35.diagnostics.block
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/work/qwen35/0.8b/inductor"
work/qwen35/venv/bin/python -m compiler.qwen35.diagnostics.export_block decode
work/qwen35/venv/bin/python -m compiler.qwen35.diagnostics.export_block prefill

/data/workspace/abo/.venv/bin/cmake -S . -B work/qwen35/build -G Ninja \
  -DCMAKE_PREFIX_PATH=/data/workspace/abo/.venv/lib/python3.12/site-packages/torch/share/cmake \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DTORCH_CUDA_ARCH_LIST=8.9 -DCMAKE_BUILD_TYPE=Release -DQWEN_BUILD_DELTANET=ON
/data/workspace/abo/.venv/bin/cmake --build work/qwen35/build --target qwen35_block -j2
work/qwen35/venv/bin/python -m compiler.qwen35.tests.validate_block \
  --binary work/qwen35/build/qwen35_block
```

Preparation/export/validation refuse to overwrite their outputs. For another
run, choose fresh `--output` paths and use the matching `--model-dir` and
`--packages` arguments. `diagnostics/block.py --model Qwen/Qwen3.5-4B` selects the pinned 4B
weights and defaults extraction to `work/qwen35/4b/block`; pass that path explicitly
to export/validation and select separate package/result directories.

## Completed 4B single-block probe

The following contract, commands and results describe **4B only**. This probe
uses the real weights of `model.language_model.layers.0.linear_attn`, including its projections,
depthwise convolution, gated delta rule, gated normalization and output
projection. It does not include the surrounding decoder MLP/residuals, other
layers, tokenizer, vision encoder or text generation.

The HF module and its PyTorch kernels are reused without modifying their
arithmetic. `TensorState` implements the small cache interface this block needs.
The independent reference uses HF's `DynamicCache`, not that adapter.

### Contract

```text
step(hidden, conv, recurrent) -> output, next_conv, next_recurrent
```

| Tensor | Shape | Dtype |
| --- | --- | --- |
| hidden / output | `[1, T, 2560]` | BF16 |
| conv / next_conv | `[1, 8192, 4]` | BF16 |
| recurrent / next_recurrent | `[1, 32, 128, 128]` | FP32 |

Prefill has fixed `T=64`; decode has fixed `T=1`. Inputs are contiguous CUDA
tensors. Zero state starts a request; returned states become the next call's
inputs. Inputs are not mutated. The adapter clones convolution storage because
HF's decode convolution updates that storage in place. This deliberately simple
functional interface permits allocations/copies; it is not a CUDA Graph path.

Weights are loaded as BF16, including conversion of the checkpoint's FP32
`A_log` and gated-norm parameters. The FP32 control loads the original checkpoint
values into FP32 parameters and uses the same BF16 input fixtures promoted to
FP32. It measures both weight/activation rounding and compute differences.

### Reproduce 4B

Use the environment and native build above. Explicit arguments reproduce the
original 4B paths; existing results must be preserved by choosing fresh output
directories for another run.

```bash
# Downloads only the shard containing layer 0, verifies its Hub SHA256, and
# extracts the block. Only the block is loaded onto the GPU.
work/qwen35/venv/bin/python -m compiler.qwen35.diagnostics.block \
  --model Qwen/Qwen3.5-4B --output work/qwen35/block

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/work/qwen35/inductor"
work/qwen35/venv/bin/python -m compiler.qwen35.diagnostics.export_block decode \
  --model-dir work/qwen35/block --output artifacts/qwen35/block
work/qwen35/venv/bin/python -m compiler.qwen35.diagnostics.export_block prefill \
  --model-dir work/qwen35/block --output artifacts/qwen35/block

work/qwen35/venv/bin/python -m compiler.qwen35.tests.validate_block \
  --model-dir work/qwen35/block --packages artifacts/qwen35/block \
  --output artifacts/qwen35/validation \
  --binary work/qwen35/build/qwen35_block
```

Export and validation refuse to overwrite their outputs. For another run, use
fresh `--output` directories and pass the corresponding `--packages` directory
to validation. `--model-dir` selects an extracted block directory.

The native runner can also be invoked directly using already prepared fixtures:

```bash
./work/qwen35/build/qwen35_block artifacts/qwen35/block \
  artifacts/qwen35/validation work/qwen35/native-rerun
```

The last directory must not exist. The native runner reads raw BF16 hidden
inputs, loads both AOTI packages once, resets state between cases and saves raw
output/state tensors. It does not invoke or link Python. The validation script
launches it with a minimal environment and checks both `ldd` and loaded mappings.

### Recorded result: 2026-10-23

- Model revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Torch `2.7.0+cu126`, Triton `3.3.0`, Transformers `5.20.2`, C++11 ABI enabled.
- NVIDIA L20, SM89, 14,336 MiB visible allocation; native CUDA toolkit 12.2.
- Strict export succeeded for both entry points, with no graph breaks and no
  graph-signature input/buffer mutations: decode 129 nodes, prefill 996 nodes.
- Both `.pt2` files are compiled AOTInductor packages, approximately 81/83 MiB
  for decode/prefill. They are not merely saved `ExportedProgram` objects.
- `emulate_precision_casts=True`; TF32 disabled. No custom kernels, mean
  overrides, or FP-contraction override were required.

Four cases cover 25 block calls: prefill + eight decode steps, four decode steps
from zero state, two consecutive prefills + decode, and a repeated first request.
Inputs are seeded synthetic hidden states, not captured full-model activations.

| Comparison | Result |
| --- | --- |
| State adapter vs HF DynamicCache | All 75 output/state tensors bitwise equal |
| Native C++ vs Python AOTI | All 75 tensors bitwise equal |
| Reset/repeated request | Bitwise equal |
| Input preservation, shapes, dtypes, finite values | Passed |
| Python-free native loading | Passed |
| AOTI vs HF BF16, 5% limits | Passed |

Worst observed errors across the 25 calls:

| Tensor | AOTI vs HF relative L2 | AOTI vs HF peak-relative | AOTI vs FP32 relative L2 | AOTI vs FP32 peak-relative |
| --- | ---: | ---: | ---: | ---: |
| Output | 0.7270% | 1.5152% | 1.0932% | 2.4579% |
| Convolution state | 0% | 0% | 0.3886% | 0.3886% |
| Recurrent state | 0.0001901% | 0.0002847% | 0.8315% | 1.3514% |

Relative L2 is `norm(actual-reference)/norm(reference)`; peak-relative is
`max(abs(actual-reference))/max(abs(reference))`. Denominators are clamped to
`1e-12`. Metrics are computed per token for output, per channel for convolution
history, and per head after flattening the recurrent 128x128 matrix. The maximum
over those groups is recorded. Both relative metrics must be <=5% for adapter/HF
and AOTI/HF comparisons. FP32 is a diagnostic control, not an eager-parity test.
All passing and failing cases retain their metrics; failures return a nonzero
exit status. Native/AOTI comparisons require exact bits.

The HF prefill function intentionally uses a triangular solver in eager and a
forward-substitution construction during export. Compiled arithmetic is therefore
not assumed to be bitwise equal to eager. No numerical acceptance failure was
observed in these fixtures. The surrounding full model remains unvalidated.

Evidence is in `artifacts/qwen35/block/` (packages, graph dumps and metadata) and
`artifacts/qwen35/validation/` (fixtures, native traces, linkage and summary).
Metadata records checkpoint and source hashes, parameter dtypes, package hashes,
versions, ABI, shapes, compile settings and observed compilation duration.
`work/qwen35/` contains the build, isolated environment, downloaded shard and
logs. Generated data is ignored by Git. The shared GPU was busy: no performance
claim is made from this correctness probe.

References: [pinned model](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a),
[pinned HF implementation](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/qwen3_5/modeling_qwen3_5.py).
