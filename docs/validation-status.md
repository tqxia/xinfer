# Validation status and known issues

Updated November 19, 2026. Results apply to the recorded packages and workloads;
validate each new export using the [validation tools](development-tools.md#python-validation-tools).
The [shared BF16 policy](development-tools.md#shared-numerical-policy) requires
logit relative L2 and peak-relative error within 5%, native/Python bitwise parity
and state/runtime contracts. HF state and token differences remain diagnostic.

## Tested profiles

The L20 runs use Torch 2.13.0+cu126; the GB10 run uses 2.13.0+cu130. Checkpoint
pins and usage are in the [Qwen3](../compiler/qwen3/README.md) and
[Qwen3.5](../compiler/qwen35/README.md) guides. Capacity is allocated context per row.

| Model / device | Capacity | Result |
| --- | ---: | --- |
| Qwen3-0.6B / L20, ABI 4, B=1 | 1,024 | Scalar suite passed all 713 required comparisons: 122 teacher-forced calls and seven greedy cases. Worst AOTI/HF relative L2: 4.0647%. |
| Qwen3-0.6B / L20, ABI 4, B=1..8 | 1,024 | Native parity and state contracts passed. Numerical acceptance remains open: 120 of 1,749 required comparisons failed. See the batch results below. |
| Qwen3.5-0.8B / L20 and GB10, GEMV profiles | 1,024 | Both passed the shared BF16 policy with zero required failures. Native parity passed; HF state/token diagnostics remain. [November 13 case study](experiments/qwen35-gemv-2026-11-13.md#correctness-and-compilation-caveat). |
| Qwen3.5-4B / L20 | 1,024 / 16,384 | Native execution and parity passed; numerical acceptance remains open with 75 / 209 logit failures respectively. At 16K, worst AOTI/HF errors were 10.62% relative L2 and 8.43% peak-relative. [1K evidence][q35-4b], [16K evidence][q35-scaling]. |
| Qwen3.5-0.8B / L20 | 16,384 / 262,144 | Native parity and full-capacity execution passed. Numerical acceptance has not been rerun under the current policy; the 262K bundle's numerical prompts reached only 513 tokens. [16K evidence][q35-scaling], [262K evidence][default-context]. |

Current Qwen3 ABI 4 bundles still need expanded-context and GB10 validation.
Qwen3.5 expanded contexts and 4B still need GB10 validation. A 1K-context pass
does not establish either. The batching runs used a busy GPU and provide no
performance conclusion.

## Qwen3 batching

The L20 bundle supports B=1..8 with eight allocated cache rows. All **142 cases
and 583 calls** passed Python/C++ AOTI bitwise logit parity and exact comparison
of all 56 final KV tensors, including unused rows. Finite values, positions,
cache ownership, write regions, padding, row isolation, reset/resumption and
overflow checks passed. Native execution loaded no Python library.

Required entries compare matched-input logits across eager/HF, AOTI/eager and
AOTI/HF. Counts below are comparison entries, not tensor elements. Transition
cases are listed separately, matching the current validator's report groups.

| Group | Failed / required entries | Worst AOTI/HF relative L2 |
| --- | ---: | ---: |
| B=1 | 0 / 93 | 3.1203% |
| B=2 | 10 / 216 | 5.9798% |
| B=3 | 9 / 228 | 5.9798% |
| B=4 | 18 / 231 | 6.9150% |
| B=5 | 18 / 234 | 6.2206% |
| B=6 | 17 / 237 | 6.5294% |
| B=7 | 26 / 240 | 6.0066% |
| B=8 | 19 / 246 | 5.9798% |
| Batch transitions | 3 / 24 | 7.0358% |
| **Total** | **120 / 1,749** | **7.0358%** |

The 8→1→3→5→8 sequence retains three HF-comparison failures: two after
shrinking to B=1 and one after returning to B=8. The 1→3→8 sequence passes
all numerical bounds. Both sequences pass state contracts and native parity;
those passes do not resolve the HF divergence.

The scalar CLI/deployment suite also passed three generation fixtures,
56 rejection cases and seven boundary cases across 67 native processes.
Operator, adapter, export and native ownership checks passed, including the
symbolic B=1 range and scalar prefill after a batched call.

## Open numerical issues

- **Qwen3 attention arithmetic:** an appended-prefill probe found the first
  difference in layer 0's attention output at both B=1 and B=2, with identical
  Q/K/V and causal positions. HF used memory-efficient SDPA; the adapter used
  FlashAttention. At 2,560 populated tokens, final-logit relative L2 reached
  4.2084% at B=1 and 6.2050% at B=2. This lies outside the current 1,024-slot
  batch suite. One-token differences also reproduced between unmodified HF
  batch sizes. These findings do not explain every current batch failure.
  [Probe evidence][q3-attention].
- **Qwen3 focused logit regression:** a 98-token case reached 6.21265% AOTI/HF
  relative L2 in retained larger-context bundles. It still needs rechecking
  with the current ABI; the 1K pass does not close it. [Evidence][default-context].
  RoPE/RMSNorm rounding sensitivity is covered in the
  [model notes](../compiler/qwen3/README.md#numerical-considerations).
- **Q35-NUM-001:** a lost BF16 rounding boundary before gated RMSNorm was
  isolated on Torch 2.7. Reproduction on the current stack remains pending;
  it is not an explanation for all full-model discrepancies. See the
  [diagnostic and model notes](../compiler/qwen35/README.md#numerical-considerations).
- **Q35-NUM-002:** full-prompt and partitioned HF execution can select different
  tokens at a tie. Use matched partitions for attribution; independent greedy
  differences remain diagnostic.
- **GEMV FP fusion:** the tested AOTI stack omitted a launch-level fusion option,
  allowing eager and packaged FP32 kernels to differ. Preserve the independent
  controls and dtype-specific checks in the
  [case study](experiments/qwen35-gemv-2026-11-13.md#correctness-and-compilation-caveat).

## Evidence

Qwen3 results cover the batching implementation in `06c668b` and bundle
`artifacts/qwen3-dynamic-batch-v2/`. Retained reports:

- Full batch matrix: `artifacts/validation-dynamic-batch-reuse/`.
- Scalar chain: `artifacts/validation-dynamic-batch-single/summary.json`.
- CLI/deployment: `artifacts/runtime-dynamic-batch/summary.json`.
- Current fixture grouping: `artifacts/validation-batch-groups-smoke/`.
  The grouping preserves the full matrix's fixtures and coverage; its focused
  14-case, 56-call rerun matched the previous metrics and digests. The full GPU
  matrix was not repeated after regrouping.

Update tested profiles, required failures and validation gaps when new evidence
changes them. Keep raw reports and traces in ignored artifact directories.

[q3-attention]: https://github.com/tqxia/xinfer/blob/06c668b606c58d339dd06ff5c90e144eea1eded2/docs/validation-status.md#qwen3-batched-prefill-attention-divergence--november-18
[q35-scaling]: https://github.com/tqxia/xinfer/blob/60e730611cdad7d312374d7d098eaa6e121ee449/docs/experiments/qwen35.md#full-model-context-scaling--2026-11-06
[q35-4b]: https://github.com/tqxia/xinfer/blob/60e730611cdad7d312374d7d098eaa6e121ee449/docs/experiments/qwen35.md#full-4b-l20-validation-2026-11-04
[default-context]: https://github.com/tqxia/xinfer/blob/60e730611cdad7d312374d7d098eaa6e121ee449/docs/experiments/common-runtime.md#model-derived-capacity-and-request-budgets--2026-11-07
