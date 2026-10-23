# Qwen3.5-4B: single DeltaNet block

Step 1 of the text-only extension is complete. This probe uses the real weights
of `model.language_model.layers.0.linear_attn`, including its projections,
depthwise convolution, gated delta rule, gated normalization and output
projection. It does not include the surrounding decoder MLP/residuals, other
layers, tokenizer, vision encoder or text generation.

The HF module and its PyTorch kernels are reused without modifying their
arithmetic. `TensorState` implements the small cache interface this block needs.
The independent reference uses HF's `DynamicCache`, not that adapter.

## Contract

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

## Reproduce on the development host

Run from `/data/workspace/develop/qwen3-aoti-poc`. The separate environment uses
the existing Torch installation read-only through a `.pth` file. The original
`.venv` and ABO packages are not upgraded. Requirements are pinned; optional
`fla` and `causal_conv1d` packages must be absent so eager and compiled runs use
the documented PyTorch baseline.

```bash
.venv/bin/python -m venv work/qwen35/venv
work/qwen35/venv/bin/python - <<'PY'
from pathlib import Path
site = Path('work/qwen35/venv/lib/python3.12/site-packages')
(site / 'torch-base.pth').write_text('/data/workspace/abo/.venv/lib/python3.12/site-packages\n')
PY
work/qwen35/venv/bin/python -m pip install -r compiler/qwen35/requirements.txt
work/qwen35/venv/bin/python -m pip check

# Downloads only the shard containing layer 0, verifies its Hub SHA256, and
# extracts the block. Only the block is loaded onto the GPU.
work/qwen35/venv/bin/python compiler/qwen35/block.py

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/work/qwen35/inductor"
work/qwen35/venv/bin/python compiler/qwen35/export_block.py decode
work/qwen35/venv/bin/python compiler/qwen35/export_block.py prefill

/data/workspace/abo/.venv/bin/cmake -S . -B work/qwen35/build -G Ninja \
  -DCMAKE_PREFIX_PATH=/data/workspace/abo/.venv/lib/python3.12/site-packages/torch/share/cmake \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DTORCH_CUDA_ARCH_LIST=8.9 -DCMAKE_BUILD_TYPE=Release -DQWEN_BUILD_DELTANET=ON
/data/workspace/abo/.venv/bin/cmake --build work/qwen35/build --target qwen35_block -j2

work/qwen35/venv/bin/python compiler/qwen35/validate_block.py \
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

## Recorded result: 2026-10-23

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

## Remaining stages

2. Expand state/length tests, especially convolution/chunk boundaries and longer
   decode chains; establish dynamic prefill or fixed-chunk remainder handling.
3. Export the complete text model with 24 DeltaNet state pairs and eight KV pairs.
4. Add the minimal C++ token-generation loop and offline tokenizer helper.
5. Validate and benchmark on the separate 48 GB L20 environment.
6. Profile and evaluate Inductor options, CUDA Graphs and suitable existing kernels.

References: [pinned model](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a),
[pinned HF implementation](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/qwen3_5/modeling_qwen3_5.py).
