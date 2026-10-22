# Qwen3-0.6B AOTInductor / C++ PoC

Status: step 1 complete (environment, checkpoint verification, eager references). Steps 2-5 remain planned.
Date: 2026-10-22.
Remote host: root@kikoxia-any4.devcloud.woa.com:36000.
Project: /data/workspace/develop/qwen3-aoti-poc.

## Goal and scope

Demonstrate a complete autoregressive generation loop for Qwen/Qwen3-0.6B:
load the original model in Python, export and AOT-compile its computation,
then load the compiled packages and generate tokens in a C++ executable.

The user selected token IDs as the runtime interface. An offline Python helper
will apply the official chat template, tokenize prompts, and decode saved output
IDs. The executable will perform all prefill, decode, cache management, greedy
selection, and stopping without invoking or embedding Python.

Initial scope:
- One visible L20 GPU, batch size 1, BF16 weights and KV cache.
- Dense Qwen3-0.6B, using the existing Hugging Face implementation.
- Prompt lengths 1–512 tokens; fixed KV capacity 1,024 tokens.
- Default generation limit 64 tokens; validate the requested context budget.
- Non-thinking chat template and greedy decoding for reproducible validation.
  Greedy decoding is a diagnostic choice, not Qwen's recommended quality setup.
- Separate prefill and decode AOTI packages, one native generation loop.
- No CUDA Graphs, quantization, concurrent requests, paged attention, HTTP server,
  custom CUDA/Triton kernels, or native tokenizer in this first milestone.

## Confirmed environment

- /data/workspace/abo/.venv: Python 3.12, torch 2.7.0, triton 3.3.0.
- Transformers, tokenizers, safetensors, and huggingface_hub are absent there.
- CMake exists at /data/workspace/abo/.venv/bin/cmake (4.3.4).
- Ninja and GCC 12.3.1 are available.
- The visible NVIDIA L20 exposes 14,336 MiB; it was idle during inspection.
- The existing benchmark already uses AOTIModelPackageLoader.
- Model files already exist at:
  /data/workspace/develop/qwen3-rust-demo/models/qwen3-0.6b
- That directory's manifest records model revision:
  c1899de289a04d12100db370d81485cdf75e47ca

Reuse the model directory read-only. Verify its recorded checksums during setup.
Keep dependencies project-local and leave the ABO environment unchanged. Start
with the existing torch 2.7.0 build exposed to an isolated project environment;
install only the additional pinned dependencies there. Transformers 4.51.3 is
the initial candidate (Qwen3 requires >=4.51.0), subject to the compatibility
probe. Build the executable against the exact Torch installation used to
compile its packages. Record package versions, ABI, GPU architecture, and CUDA.

## Architecture and tensor contract

Offline:
  Hugging Face Qwen3 -> small tensor-only wrapper -> torch.export
  -> AOTInductor -> prefill.pt2 + decode.pt2 + metadata

Native:
  prompt IDs -> prefill -> greedy token selection -> repeated decode -> output IDs

Use the same Python wrapper for both exports. Reuse model layers and weights;
do not copy the Qwen implementation into this repository. A small cache adapter
will bridge flat tensor arguments to the Hugging Face forward API.

Proposed exported signature:
- input_ids: int64 CUDA tensor [1, T]
- cache_position: int64 CUDA tensor [T]
- per-layer K/V tensors: 28 pairs, each BF16 [1, 8, 1024, 128]
- result: logits for the last supplied token, [1, 151936]
- side effect: update only the supplied positions in the caller-owned KV tensors

The C++ runtime allocates the cache once and passes that same storage to both
packages. Construct the causal mask from absolute positions inside the wrapper
and mask unused cache entries. Explicitly verify that export/AOTI preserves the
input-mutation contract and that C++ observes the updates. Check generated code
for full-cache copies or accidental replacement storage; semantic correctness
alone is not sufficient to claim an efficient persistent-cache path.

Prefill: bounded dynamic T=2..512.
Decode: fixed T=1, with the position supplied as tensor data.
A one-token prompt uses the decode package at position zero. This keeps the
T=1 specialization explicit and avoids padding/bucket logic in the PoC.

Prefer native PyTorch SDPA with an explicit mask; verify its export and AOTI
lowering in the chosen versions. An eager/math attention fallback may be used
to isolate a correctness failure, and must be identified in any timing report.

Use two independent loaders initially. Accept potential duplicated immutable
weights for this small model; measure actual memory rather than introducing
weight-sharing machinery. Both loaders must still share the caller's KV tensors.
The planned BF16 cache is 112 MiB, calculated from
2 * 28 * 8 * 1024 * 128 * 2 bytes. Other buffers and weights are additional.

## Proposed small file layout

README.md          Exact setup/export/build/run/validate commands and results
PLAN.md            This design and acceptance criteria
requirements.txt   Pinned additional Python dependencies
model.py           Thin Qwen wrapper and explicit cache adapter
export.py          Load weights, export/compile both graphs, write metadata
prepare.py         Offline prompt tokenization and output-ID decoding
validate.py        Reference comparisons and focused regression cases
main.cpp           Load packages, allocate cache, prefill/decode, stop and time
CMakeLists.txt     Build against the matching LibTorch

Ignored generated directories: .venv/, build/, artifacts/, work/.
Use simple token-ID files and a small metadata file; avoid a general framework,
configuration hierarchy, serialization framework, or duplicated model code.
Keep cache layout and position semantics documented beside the code that uses them.

## Implementation sequence

1. Establish the environment and eager reference.
   Verify model files, pin dependencies, load the local checkpoint in BF16, and
   prepare a few non-thinking prompt fixtures with the official tokenizer.
   Record reference next-token logits and a short greedy sequence.

2. Prove the export/cache contract before building the full application.
   Use a focused cache-update probe if necessary, then run the Qwen wrapper
   eagerly. Compare it with the unmodified HF model. Export and compile decode,
   verify several successive positions, then export prefill and verify handoff
   into decode using the same cache. Test T=1 separately from dynamic T>=2.
   Resolve export failures here; do not spread workarounds through the C++ loop.

3. Implement the minimal C++ executable.
   Use AOTIModelPackageLoader and LibTorch tensors. Load metadata, validate token
   IDs and context limits, allocate zeroed GPU cache, execute prefill, select a
   token using GPU argmax, and call decode until EOS or the generation limit.
   Read EOS IDs from the recorded generation configuration (the local model
   lists 151645 and 151643). Support saving generated IDs and an optional
   validation trace. A scalar synchronization per step is acceptable initially.

4. Validate the three execution paths.
   Compare HF eager, Python-loaded AOTI, and the standalone C++ executable.
   Use teacher-forced identical token continuations for logit comparisons so
   that a sampling divergence does not invalidate all later comparisons.
   Also check end-to-end greedy token sequences and repeated requests.

5. Measure and document the completed PoC.
   Record export/build commands, pinned versions, correctness results, compilation
   time, package sizes, model-load time, inference-only first-token latency,
   steady-state decode milliseconds/token and tokens/second, and peak GPU memory.
   Compare against the same eager workload, dtype, cache and attention setup.
   Separate warmup from measurement, reset its cache, and synchronize CUDA at
   measurement boundaries. Identify host synchronization included in timings.
   Do not assert a performance target or a vLLM comparison at this stage.

## Acceptance criteria

- A fresh C++ process loads the packages and generates without Python execution.
  Confirm the binary does not link libpython/libtorch_python, and run the native
  command independently of the preparation/verification helpers.
- Original HF versus wrapper, wrapper versus AOTI, and Python AOTI versus C++
  comparisons pass with documented BF16-appropriate error criteria.
- Check intermediate logits under identical fed tokens. Report maximum/relative
  error and top-1 agreement. Investigate greedy divergence with logit margins;
  do not conceal mismatches by merely relaxing tolerances.
- Cover prompt lengths 1, 2, an ordinary short prompt, 128, 511, and 512.
- Cover multiple decode steps, EOS, generation limits, cache reset/reuse,
  processing different prompts in succession, and the final legal cache slot.
- Reject invalid token IDs, empty/oversized prompts and invalid context budgets
  before launching model work.
- Verify prefill/decode cache handoff and preserved input-mutation semantics.
- Decode generated IDs offline to show readable text alongside numerical checks.
- README commands reproduce the result, and implementation remains a short
  model adapter plus a straightforward native loop.

## Risks and deliberate tradeoffs

The first technical risk is HF cache adaptation and mutation lowering in torch
2.7.0; the next is dynamic prefill export. If either needs a larger workaround,
document a minimal reproducer and the proposed change before expanding scope.
A version change must keep export-time Torch and native LibTorch matched.

Fixed-capacity attention may do work over unused cache slots. Two loaders may
duplicate weights. Greedy decoding and scalar synchronization prioritize a
readable correctness PoC. These are measured limitations, with optimization
considered only after this milestone is complete.

## References

- Qwen model card:
  https://huggingface.co/Qwen/Qwen3-0.6B
- Qwen configuration:
  https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json
- HF Qwen implementation, proposed initial version:
  https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py
- PyTorch 2.7 AOTInductor native deployment:
  https://docs.pytorch.org/docs/2.7/torch.compiler_aot_inductor.html
