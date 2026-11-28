# Validation status and known issues

Updated November 26, 2026. Results apply to the recorded bundles and workloads
below. Under the [shared numerical policy](development-tools.md#shared-numerical-policy),
matched-input logit relative L2 and peak-relative error must be `<= 5%`,
native/Python parity must be bitwise, and structural/state checks are required;
HF state and token differences are diagnostic.

## Current validation

BF16 dynamic-batch runs below used L20 (`Torch 2.13.0+cu126`); Qwen3.8-27B-FP8
used GB10 (`Torch 2.13.0+cu130`). All used dynamic `B=1..8` and 1,024-token
per-row capacity. The FP8 reference retains the unmodified HF FP8 modules with
pinned `kernels-community/finegrained-fp8` v4
(`03ae31933313f38fae2b050dbe2731024386c369`). The recorded L20 Qwen3.5
dynamic-batch run used linear packing and default Inductor options (predating
dynamic-batch GEMV support).

| Model | Target | Batch cases / calls | Failed / required logit comparisons | Worst AOTI/HF relative L2 |
| --- | --- | ---: | ---: | ---: |
| Qwen3-0.6B | L20 | 142 / 583 | **120 / 1,749** | 7.0358% |
| Qwen3.5-0.8B | L20 | 166 / 663 | **68 / 1,989** | 10.9454% |
| Qwen3.8-27B-FP8 | GB10 | 166 / 663 | **1,648 / 1,989** | 45.3763% |

All three passed native bitwise parity for logits and all allocated state rows
(including unused rows), plus finite values, positions, ownership, valid writes,
padding, row isolation, reset/resumption, batch-size transitions, overflow,
CLI/rejection checks, and zero Python library linkage. **Full-model numerical
acceptance remains open.** (Each required count is one matched-input call at one
stage — `eager/HF`, `AOTI/eager`, or `AOTI/HF` — evaluated by worst per-row
error.)

Qwen3.8-27B-FP8 dynamic-batch validation ran unattended across 6 disk-bounded
waves on GB10 against the `weights.safetensors` bundle with sequential
single-model GPU staging; earlier full scalar and runtime suites ran in
partitions on GB10 with identical bounds.

Additional coverage:

- **Qwen3-0.6B (L20 scalar):** full B=1 suite passed all `713` required comparisons.
- **Qwen3.5-0.8B (L20 focused scalar):** passed 28 teacher-forced calls and 1
  greedy case with zero required failures.
- **Qwen3.8-27B-FP8 (GB10 scalar & components):** 49 cases / 382 teacher-forced
  calls / 3 greedy cases passed all `50,067` exact native checks and all 52
  rejection checks, with `861 / 1,178` required HF numerical failures. FP8
  projection and 48-head DeltaNet component/AOTI checks passed.
- **DeltaNet component tests:** BF16 output/carried-state checks passed against
  HF BF16; standalone AOTI/native parity passed all 90 BF16/FP32 calls.
- **PRO 5000 (`sm_120`, Torch `2.13.0+cu130`, 1,024-token capacity):**
  - `Qwen3.5-0.8B`: full scalar validation passed (`0 / 1,174` required
    failures, worst AOTI/HF relative L2 `3.7865%`, `19,216 / 19,216` bitwise
    native checks) and `validate_runtime` passed (`5,243` teacher tensors, `28`
    greedy logit rows, `288` greedy state tensors, rejection checks).
  - `Qwen3-0.6B`: `validate_runtime` and `250 / 250` bitwise native checks
    passed (worst AOTI/HF relative L2 `4.8870%`), with `1 / 517` required
    failure (`eager_vs_hf` on `forced_tokens` step 2, relative L2 `5.2020%`).
  - Attention/compile (`13/13`), DeltaNet AOTI (`17/17`), native contract tests
    and CPU unit tests (`404 OK`) passed. Dynamic-batch and larger models remain
    untested on PRO 5000.

## Open numerical issues

- **Qwen3:** HF SDPA and FlashAttention differ from layer 0 at B=1 and B=2 on
  identical Q/K/V inputs (not sufficient to explain every batch failure). The
  `8→1→3→5→8` sequence retains 3 HF failures despite passing state contracts and
  native parity ([attention probe][q3-attention]). The focused 98-token
  larger-context logit regression still needs rechecking.
- **Qwen3.5:** batch failures are `29/663` (`eager/HF`), `11/663` (`AOTI/eager`)
  and `28/663` (`AOTI/HF`), worst in B=5 `prompt_prefix_teacher` decode.
  Batch-shape-sensitive BF16 projection rounding has been observed; full
  attribution remains open. Shrinking/growing batch sequences pass both bounds
  and parity.
- **Qwen3.8-27B-FP8:** batch failures are `545/663` (`eager/HF`), `544/663`
  (`AOTI/eager`) and `559/663` (`AOTI/HF`, worst `45.3763%` in B=8 `length_1024`
  second prefill); scalar `AOTI/HF` fails `279/382` (worst `43.9296%`). FP8-wrapper
  and explicit-cache controls match HF bitwise when retaining HF attention/DeltaNet
  math. Compiling only HF normalization with Inductor (cast emulation enabled)
  yields `6.9–7.5%` prefill relative L2; replacing only DeltaNet yields
  `8.6–8.7%`. Small BF16 differences amplify through the 64-layer model; full
  attribution remains open.
- **Q35-NUM-001:** lost BF16 rounding boundary before gated RMSNorm still needs
  rechecking on the current toolchain; see
  [Qwen3.5 numerical notes](../compiler/qwen35/README.md#numerical-considerations).

## Validation gaps

- BF16 dynamic-batch profiles need expanded-context and GB10 validation;
  Qwen3.5-4B needs dynamic-batch re-export and validation.
- Qwen3.5 needs the full scalar numerical suite on L20 beyond the focused subset.
- Qwen3.8-27B-FP8 needs expanded-context validation; performance and quality vs
  unquantized weights are unmeasured.
- Batch performance is unmeasured; L20 correctness runs used a busy GPU.

## Evidence

Raw reports and traces live in local `artifacts/` directories (GB10 full-state
snapshots were removed after comparison; reports, logits and manifests remain):

- **Qwen3:** `artifacts/qwen3-dynamic-batch-v2/` — batch
  `artifacts/validation-dynamic-batch-reuse/`, scalar
  `artifacts/validation-dynamic-batch-single/summary.json`, runtime
  `artifacts/runtime-dynamic-batch/summary.json`.
- **Qwen3.5:** `artifacts/qwen35-dynamic-batch-v2/` — batch
  `artifacts/qwen35-batch-validation-v3/`, focused scalar
  `artifacts/qwen35-dynamic-scalar-focused-v2/summary.json`, runtime
  `artifacts/qwen35-dynamic-runtime-v2/summary.json`.
- **Qwen3.8-27B-FP8:** `artifacts/qwen38/27b-fp8/text/` — 6-wave batch
  `artifacts/qwen38/27b-fp8/batch-validation-safetensors/summary.json` (logs
  `work/qwen38/batch-residency-validation/`), prior scalar/runtime
  `artifacts/qwen38/27b-fp8/validation-c1024-540ba06/result.json`, diagnostics
  `artifacts/qwen38-investigation-20261121/`.
- **DeltaNet components:** `artifacts/deltanet-batch/`.

[q3-attention]: https://github.com/tqxia/xinfer/blob/06c668b606c58d339dd06ff5c90e144eea1eded2/docs/validation-status.md#qwen3-batched-prefill-attention-divergence--november-18
