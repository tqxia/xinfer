# Qwen3.5 text-only AOTI / C++

The primary target is **Qwen/Qwen3.5-0.8B**, pinned to
`2fc06364715b967f1860aea9cf38778875588b17`. The earlier 4B single-block probe is
retained for compatibility checks. Vision and MTP are outside this experiment.

The full model exports, compiles and runs in C++. Native/Python AOTI outputs
match bitwise across 203 teacher-forced steps / 9,947 tensors. Numerical
acceptance against HF is **open**: conv/recurrent errors exceed the unchanged
5% limits, and some token decisions differ. Benchmark execution is validated;
final performance measurements require the user's separate idle environment.

## Contract

- One GPU, batch one, prompts of 1–512 tokens, total context capacity 1,024.
- Prefill consumes fixed 64-token chunks; remaining tokens use the T=1 package.
- BF16 weights/activations, FP32 RoPE frequencies and DeltaNet recurrent state.
- The 48 state tensors cover 18 conv/recurrent pairs and six KV pairs. Inputs
  remain unchanged; package outputs replace the previous states.
- Python prepares/compiles and tokenizes/decodes offline. The native process
  loads compiled packages and performs all generation without Python libraries.

## Source layout

`model.py` owns checkpoint preparation and the full-model adapter; `export.py`
builds AOTI packages; `prepare.py` handles token IDs. `reference.py` supplies HF
and Python AOTI runners for checks. `config.py` owns the pinned checkpoints,
execution settings and DeltaNet state shapes. The full-model path does not
import diagnostic modules or require single-block packages.

`diagnostics/block.py` / `diagnostics/export_block.py` retain the isolated
DeltaNet experiment alongside focused issue reproducers. `tests/` holds
acceptance checks and fixtures, including the optional block/C++ checks.
General file/error helpers live in `compiler/common/`; native code is in
`runtime/qwen35/`.

## Environment and build

Use the isolated `work/qwen35/venv` environment and `requirements.txt`; Qwen3.5
needs Transformers 5.20.2, while Qwen3 keeps its separate environment. Match
export-time Torch and native LibTorch. Optional `fla` and `causal_conv1d` must
be absent for this baseline. The development host reuses Torch read-only from
ABO; that host-specific setup is recorded in the experiment record below.

Run all commands from the repository root. Output directories must be fresh.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
QWEN_PY=work/qwen35/venv/bin/python
# Prepare once, or copy the prepared text directory including its manifest.
"$QWEN_PY" -m compiler.qwen35.model
"$QWEN_PY" -m compiler.qwen35.export decode
"$QWEN_PY" -m compiler.qwen35.export prefill

QWEN_CMAKE_PREFIX=$("$QWEN_PY" -c 'import torch; print(torch.utils.cmake_prefix_path)')
QWEN_SM=$("$QWEN_PY" -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')
cmake -S . -B work/qwen35/build -G Ninja \
  -DCMAKE_PREFIX_PATH="$QWEN_CMAKE_PREFIX" -DTORCH_CUDA_ARCH_LIST="$QWEN_SM" \
  -DCMAKE_BUILD_TYPE=Release -DQWEN_BUILD_QWEN35=ON \
  -DQWEN_BUILD_VALIDATION=ON -DQWEN_BUILD_BENCHMARK=ON
cmake --build work/qwen35/build --target qwen35_aoti qwen35_validate qwen35_benchmark -j2
```

## Generate and validate

```bash
"$QWEN_PY" -m compiler.qwen35.prepare encode --prompt "What is 17 + 25?" --output work/prompt.ids
work/qwen35/build/qwen35_aoti artifacts/qwen35/0.8b/text work/prompt.ids work/generated.ids
"$QWEN_PY" -m compiler.qwen35.prepare decode --input work/generated.ids

"$QWEN_PY" -m compiler.qwen35.tests.validate_model
"$QWEN_PY" -m compiler.qwen35.tests.validate_runtime --output artifacts/qwen35/0.8b/runtime-new
"$QWEN_PY" -m compiler.qwen35.tests.validate_end_to_end --output artifacts/qwen35/0.8b/end-to-end-new
```

The model/end-to-end validators exit nonzero for numerical acceptance failures
and retain their reports. Native parity is checked separately; a passing native
test does not close HF acceptance. Use `--help` for package/model/binary paths.

## Benchmark

```bash
"$QWEN_PY" -m compiler.qwen35.benchmark run --gpu 0 \
  --validation artifacts/qwen35/0.8b/end-to-end-new/summary.json \
  --output artifacts/qwen35/0.8b/benchmark-new
```

Profiles are HF full-prompt, HF with matching 64/1 partitions, and native AOTI.
All receive identical fixed tokens; EOS is data during timing. Reports retain
numerical acceptance, reset/prefill/decode timings, memory, workload, provenance
and telemetry. `--allow-busy` is only for functional harness checks and marks
timings provisional. Re-export packages and rebuild on the target GPU/toolchain;
the existing SM89 artifacts are not a portable deployment bundle.

## Open issues and evidence

- **Q35-NUM-001:** lost BF16 rounding before DeltaNet gated normalization.
  `python -m compiler.qwen35.diagnostics.probe_rounding` reproduces the issue;
  exit zero means reproduced, not fixed.
- **Q35-NUM-002:** partition-sensitive greedy punctuation decision; HF itself
  differs between full-prompt and chunked execution. Another fixed-token case
  differs between AOTI and HF BF16; its precise cause remains unisolated.
- Bounded dynamic block prefill fails shape guards in the pinned toolchain.
  `python -m compiler.qwen35.diagnostics.probe_dynamic` records that limitation.

[Detailed experiments, metrics, setup and target-environment instructions](../../docs/experiments/qwen35.md)
preserve all five steps and the earlier 4B result. Generated evidence remains in
`artifacts/qwen35/`; scratch scripts, builds and caches remain ignored in `work/`.

Shared execution helpers are in `runtime/common/generation.h`,
`runtime/common/benchmark.h` and `compiler/common/benchmark.py`. Model-specific
state, package validation and benchmark profiles stay in their model directory.

The organization regression in `artifacts/qwen35/0.8b/reorg-step2-*` reproduces
the retained block, native-runtime and end-to-end results, including every
numerical comparison and known failure.
