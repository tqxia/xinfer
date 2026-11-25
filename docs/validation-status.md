# Validation status and known issues

Updated November 24, 2026. Results apply to the bundles and workloads below.
The [shared numerical policy](development-tools.md#shared-numerical-policy) requires
matched-input logit relative L2 and peak-relative error within 5%, native/Python
bitwise parity and state/runtime contracts. HF state and token differences are
diagnostic. Validate each new export with the [validation tools](development-tools.md#python-validation-tools).

## Current validation

The BF16 models below were tested on L20 with Torch 2.13.0+cu126; the
Qwen3.8-27B-FP8 profile was tested on GB10 with Torch 2.13.0+cu130. All use
dynamic B=1..8 and 1,024-token context capacity per row. The FP8 reference retains
the original HF FP8 modules and the pinned version-4 kernels-community
implementation (`03ae31933313f38fae2b050dbe2731024386c369`).
Checkpoint pins and model contracts are in the
[Qwen3](../compiler/qwen3/README.md), [Qwen3.5](../compiler/qwen35/README.md) and
[Qwen3.8](../compiler/qwen38/README.md) guides. Qwen3.5 uses linear packing and default Inductor options; the
singleton-only GEMV pass does not match its dynamic-batch graphs.

| Model | Target | Batch cases / calls | Failed / required logit comparisons | Worst AOTI/HF relative L2 |
| --- | --- | ---: | ---: | ---: |
| Qwen3-0.6B | L20 | 142 / 583 | **120 / 1,749** | 7.0358% |
| Qwen3.5-0.8B | L20 | 166 / 663 | **68 / 1,989** | 10.9454% |
| Qwen3.8-27B-FP8 | GB10 | 166 / 663 | **1,648 / 1,989** | 45.3763% |

All three profiles passed native bitwise parity for logits and all allocated
state rows, including unused rows. Finite values, positions, ownership, valid writes,
padding, row isolation, reset/resumption, batch-size transitions and overflow
checks passed. Generation, CLI and rejection checks also passed. Native execution
loaded no Python library. **Full-model numerical acceptance remains open.**

Qwen3.8-27B-FP8 dynamic-batch validation completed in a single unattended run
across 6 disk-bounded waves on GB10 against the re-exported `weights.safetensors`
bundle (with sequential single-model GPU staging); the earlier full scalar and
runtime suites ran in partitions on GB10 with fixtures, tensor bytes and
numerical bounds preserved.

Each required batch entry compares one matched-input call at one stage
(eager/HF, AOTI/eager or AOTI/HF), using the worst per-row error. These counts
are not vocabulary elements or individual failed rows.

Additional coverage:

- Qwen3's full B=1 scalar suite passed all 713 required comparisons.
- Qwen3.5's focused scalar chain passed 28 teacher-forced calls and one greedy
  case with zero required failures.
- Qwen3.8-27B-FP8's full scalar suite covered 49 cases, 382 teacher-forced calls
  and three greedy cases. All 50,067 exact comparisons passed; 861/1,178 required
  numerical comparisons failed, counting repeated partition anchors once.
  Native generation, CLI and all 52 rejection checks passed. FP8 projection and
  48-head DeltaNet component/AOTI checks also passed in the earlier component run.
- DeltaNet BF16 component output/carried-state tests passed against HF BF16
  tensor operations. Standalone AOTI/C++ parity passed all 90 BF16/FP32 calls.

## Open numerical issues

- **Qwen3:** attention arithmetic differs between HF SDPA and FlashAttention
  from layer 0 at both B=1 and B=2 with identical Q/K/V inputs; it does not
  explain every batch failure. The 8→1→3→5→8 sequence retains three HF-comparison
  failures despite passing state contracts and native parity.
  [Attention probe][q3-attention]. The focused 98-token larger-context logit
  regression still needs rechecking with the current model interface.
- **Qwen3.5:** eager/HF fails 29/663 comparisons, AOTI/eager 11/663 and AOTI/HF
  28/663. The largest AOTI/HF relative L2 error is in B=5
  `prompt_prefix_teacher` decode. Batch-shape-sensitive BF16 projection rounding
  has been observed, but attribution of the full-model errors remains open.
  Both shrinking/growing batch sequences pass numerical bounds and native parity.
- **Qwen3.8-27B-FP8:** batch eager/HF fails 545/663 comparisons, AOTI/eager
  544/663 and AOTI/HF 559/663. Worst AOTI/HF relative L2 is 45.3763% in B=8
  `length_1024`'s second prefill. Scalar AOTI/HF fails 279/382 teacher-forced
  comparisons, with worst relative L2 43.9296%. These compare execution against
  the original FP8 checkpoint, not quantization loss against unquantized weights.
  FP8-wrapper and explicit-cache controls match HF bitwise when retaining HF
  attention/DeltaNet math. Compiling only HF normalization with `torch.compile`
  (Inductor, cast emulation enabled) produces 6.9–7.5% prefill logit relative L2;
  replacing only DeltaNet produces 8.6–8.7%. Small BF16 arithmetic differences
  amplify through the model. The normalization ablation does not establish a
  compiler correctness bug or explain every production AOTI difference.
- **Q35-NUM-001:** the lost BF16 rounding boundary before gated RMSNorm still
  needs verification on the current toolchain. See the
  [model notes](../compiler/qwen35/README.md#numerical-considerations).

## Validation gaps

- The BF16 dynamic-batch profiles need expanded-context and GB10 validation;
  Qwen3.5-4B also needs re-export and validation with dynamic batching.
- Qwen3.5 needs the complete scalar numerical suite beyond the focused regression.
- Qwen3.8-27B-FP8 needs expanded-context validation; performance and quality
  against unquantized weights are unmeasured.
- Batch performance is unmeasured. The L20 correctness runs used a busy GPU.

## Evidence

Raw reports and retained traces remain in local artifact directories. Per-case
failures, metrics and package identities belong there; keep this file a current
status summary.
GB10 full-state snapshots were removed after comparison; reports, logits, manifests
and failed-attempt evidence remain.

- **Qwen3 bundle:** `artifacts/qwen3-dynamic-batch-v2/`.
  Batch results: `artifacts/validation-dynamic-batch-reuse/`;
  scalar results: `artifacts/validation-dynamic-batch-single/summary.json`;
  runtime results: `artifacts/runtime-dynamic-batch/summary.json`.
- **Qwen3.5 bundle:** `artifacts/qwen35-dynamic-batch-v2/`.
  Batch results: `artifacts/qwen35-batch-validation-v3/`;
  focused scalar results: `artifacts/qwen35-dynamic-scalar-focused-v2/summary.json`;
  runtime results: `artifacts/qwen35-dynamic-runtime-v2/summary.json`.
- **Qwen3.8-27B-FP8 bundle (`weights.safetensors`):** `artifacts/qwen38/27b-fp8/text/`.
  Full 6-wave batch results:
  `artifacts/qwen38/27b-fp8/batch-validation-safetensors/summary.json`
  (logs: `work/qwen38/batch-residency-validation/`);
  prior full scalar and runtime results:
  `artifacts/qwen38/27b-fp8/validation-c1024-540ba06/result.json`;
  numerical diagnostics: `artifacts/qwen38-investigation-20261121/`.
- **DeltaNet component results:** `artifacts/deltanet-batch/`.

[q3-attention]: https://github.com/tqxia/xinfer/blob/06c668b606c58d339dd06ff5c90e144eea1eded2/docs/validation-status.md#qwen3-batched-prefill-attention-divergence--november-18
