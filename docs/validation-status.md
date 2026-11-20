# Validation status and known issues

Updated November 20, 2026. Results apply to the bundles and workloads below.
The [shared BF16 policy](development-tools.md#shared-numerical-policy) requires
matched-input logit relative L2 and peak-relative error within 5%, native/Python
bitwise parity and state/runtime contracts. HF state and token differences are
diagnostic. Validate each new export with the [validation tools](development-tools.md#python-validation-tools).

## Current validation

Both models have been tested on L20 with Torch 2.13.0+cu126, dynamic B=1..8 and
1,024-token context capacity per row. Checkpoint pins and model contracts are in
the [Qwen3](../compiler/qwen3/README.md) and [Qwen3.5](../compiler/qwen35/README.md)
guides. Qwen3.5 uses linear packing and default Inductor options; the
singleton-only GEMV pass does not match its dynamic-batch graphs.

| Model | Batch cases / calls | Failed / required logit comparisons | Worst AOTI/HF relative L2 |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 142 / 583 | **120 / 1,749** | 7.0358% |
| Qwen3.5-0.8B | 166 / 663 | **68 / 1,989** | 10.9454% |

Both passed native bitwise parity for logits and all allocated state rows,
including unused rows. Finite values, positions, ownership, valid writes,
padding, row isolation, reset/resumption, batch-size transitions and overflow
checks passed. Generation, CLI and rejection checks also passed. Native execution
loaded no Python library. **Full-model numerical acceptance remains open.**

Each required batch entry compares one matched-input call at one stage
(eager/HF, AOTI/eager or AOTI/HF), using the worst per-row error. These counts
are not vocabulary elements or individual failed rows.

Additional coverage:

- Qwen3's full B=1 scalar suite passed all 713 required comparisons.
- Qwen3.5's focused scalar chain passed 28 teacher-forced calls and one greedy
  case with zero required failures.
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
- **Q35-NUM-001:** the lost BF16 rounding boundary before gated RMSNorm still
  needs verification on the current toolchain. See the
  [model notes](../compiler/qwen35/README.md#numerical-considerations).

## Validation gaps

- Current dynamic-batch interfaces need expanded-context and GB10 validation;
  Qwen3.5-4B also needs re-export and validation with dynamic batching.
- Qwen3.5 needs the complete scalar numerical suite beyond the focused regression.
- Batch performance is unmeasured. These correctness runs used a busy GPU.

## Evidence

Raw reports and traces remain in local artifact directories. Per-case failures,
metrics and package identities belong there; keep this file a current status summary.

- **Qwen3 bundle:** `artifacts/qwen3-dynamic-batch-v2/`.
  Batch results: `artifacts/validation-dynamic-batch-reuse/`;
  scalar results: `artifacts/validation-dynamic-batch-single/summary.json`;
  runtime results: `artifacts/runtime-dynamic-batch/summary.json`.
- **Qwen3.5 bundle:** `artifacts/qwen35-dynamic-batch-v2/`.
  Batch results: `artifacts/qwen35-batch-validation-v3/`;
  focused scalar results: `artifacts/qwen35-dynamic-scalar-focused-v2/summary.json`;
  runtime results: `artifacts/qwen35-dynamic-runtime-v2/summary.json`.
- **DeltaNet component results:** `artifacts/deltanet-batch/`.

[q3-attention]: https://github.com/tqxia/xinfer/blob/06c668b606c58d339dd06ff5c90e144eea1eded2/docs/validation-status.md#qwen3-batched-prefill-attention-divergence--november-18
