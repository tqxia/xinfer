# Validation status and known issues

This page summarizes recorded evidence, not a new validation run or a guarantee
for every build. Results belong to a checkpoint, compiled capacity, toolchain,
source revision and workload. Re-export and validate changes using the
[current validation tools](development-tools.md#python-validation-tools).

The current acceptance rules are [shared_bf16 v1](development-tools.md#shared-numerical-policy).
Required logit bounds, native/Python bitwise parity and state/runtime contracts
are separate from HF state/token diagnostics. Older reports sometimes required
state bounds or exact HF tokens. Preserve their original verdicts: changing the
policy does not fix arithmetic differences or establish a fresh pass.

## Recorded profiles

The L20 runs below used Torch 2.13.0+cu126; the November 13 GB10 run used
2.13.0+cu130. The model guides identify the pinned
[Qwen3](../compiler/qwen3/README.md) and [Qwen3.5](../compiler/qwen35/README.md)
checkpoints. Capacity is allocated context, not the maximum tokens per call.

| Model / device | Capacity | Evidence date | Result and scope |
| --- | ---: | --- | --- |
| Qwen3-0.6B / L20 | 16,384 | November 6 | The recorded numerical suite passed its HF logit gates and native/Python parity, including 8K prompts and full-capacity checks. Worst direct AOTI/HF logit relative L2 was 4.52264%. [Source][q3-scaling] |
| Qwen3-0.6B / L20 | 40,960 | November 7 | Export, full-capacity native execution and focused native/Python parity passed. A separate focused numerical check failed three required logit comparisons on the final decode of a 98-token case; direct AOTI/HF relative L2 was 6.21265%. [Source][default-context] |
| Qwen3.5-0.8B / L20 | 16,384 | November 6 | Native parity, state contracts and full-capacity execution passed. The historical numerical report failed state/token gates, with no logit-threshold failure. This is not a current-policy rerun. [Source][q35-scaling] |
| Qwen3.5-4B / L20 | 1,024 | November 4 | Full checkpoint export, native parity, generation and shared-weight loading passed. Numerical validation failed, including 75 logit comparisons across stages. [Source][q35-4b] |
| Qwen3.5-4B / L20 | 16,384 | November 6 | Native parity and full-capacity execution passed. Numerical validation failed 209 logit comparisons across the chain; the long-context AOTI/HF maxima were 10.62% relative L2 and 8.43% peak-relative error. [Source][q35-scaling] |
| Qwen3.5-0.8B / L20 | 262,144 | November 7 | Export, full-capacity native execution and focused native parity passed. The historical numerical report failed state/token gates, with no logit-threshold failure. Its longest numerical prompt was 513 tokens; filling the cache was an execution/memory check. [Source][default-context] |
| Qwen3.5-0.8B / GB10 | 1,024 | November 13 | The controlled GEMV candidate passed shared_bf16 v1 with zero required failures and 19,217 exact native/Python tensor records. State/token diagnostics remained. [Case study](experiments/qwen35-gemv-2026-11-13.md#correctness-and-compilation-caveat) |
| Qwen3.5-0.8B / L20 | 1,024 | November 13 | Baseline and GEMV candidates passed shared_bf16 v1; the candidate had 19,216 exact native/Python records. State diagnostics remained. [Case study](experiments/qwen35-gemv-2026-11-13.md#correctness-and-compilation-caveat) |

The November 7 focused Qwen3 failure also reproduced with a retained 2,048-slot
bundle. A different 2,048-slot suite had passed earlier that day. These are
different case sets, not evidence that changing capacity caused or fixed the
failure. The November 6 scaling suites covered prompt boundaries
1/511/512/513/8191/8192, carried state, poisoned future KV, reset/reuse and overflow.

The November 13 GEMV study started from revision
`f6aba28949ec20375d4130ca3209ca325c950da2`; it records the tested implementation
and controlled exports. Its results establish neither 4B acceptance nor
long-context acceptance of the new projections. Earlier GB10 checks used older
packages; current Qwen3 packaging, expanded contexts and 4B still need their own
GB10 validation. The GEMV run does establish bundled-attention execution for
its tested Qwen3.5-0.8B GB10 profile.

## Numerical findings to preserve

- **Qwen3 logit regression:** the focused 98-token case above exceeds the required
  5% bound. Native parity does not close it. Keep that case when checking changes.
- **Qwen3 rounding sensitivity:** earlier probes isolated BF16 RoPE contraction
  and RMSNorm reduction order crossing BF16 rounding midpoints. Matching BF16
  eager bitwise did not consistently improve agreement with an independent FP32
  control. These findings motivate bounded matched-input comparisons, not an
  alternate eager-numerics export profile. See the
  [model notes](../compiler/qwen3/README.md#numerical-considerations).
- **Q35-NUM-001:** a Torch 2.7 layer-0 probe found a lost BF16 rounding boundary
  before gated RMSNorm. It is one identified contributor, not an explanation for
  every discrepancy or a fresh reproduction on Torch 2.13. The maintained
  [diagnostic](../compiler/qwen35/README.md#numerical-considerations) can recheck it.
- **Q35-NUM-002:** full-prompt and partitioned HF execution can choose different
  greedy tokens at a tie. Matching partitions is necessary for attribution;
  independent greedy differences remain diagnostics under the current policy.
- **4B logits:** required logit errors remain unresolved in the recorded full-model
  and long-context runs. Reclassifying state/token findings cannot resolve these.
- **GEMV FP fusion:** eager and packaged FP32 kernels can differ because AOTI does
  not preserve one launch-level fusion option in the tested stack. Preserve the
  independent controls and dtype-specific parity checks described in the
  [case study](experiments/qwen35-gemv-2026-11-13.md#correctness-and-compilation-caveat).

## Memory evidence

Shared-weight loading was checked against embedded-weight packages in fresh
L20 processes on November 2, including CUDA allocations outside PyTorch's
caching allocator. After an eight-token prefill and one decode, median device
use fell from 2,764 to 1,626 MiB for Qwen3-0.6B and from 3,302 to 1,936 MiB for
Qwen3.5-0.8B. The initial sharing run used archive format 1; the format-2 rerun
retained those measurements and native output parity. These are historical
storage-sharing measurements, not current latency claims. [Source][shared-weights]

For the November 6 Qwen3.5 full 16K-context probes:

| Measurement | 0.8B | 4B |
| --- | ---: | ---: |
| Peak PyTorch allocated | 1.706 GiB | 8.599 GiB |
| Peak PyTorch reserved | 1.811 GiB | 8.707 GiB |
| Maximum boundary-sampled total device use | 2.334 GiB | 9.231 GiB |

Both probes filled the context without OOM. Total-device samples include CUDA
context/library memory and can miss temporaries freed inside a compiled call;
they are not exact global peaks. Export and accuracy validation also hold
additional model copies/workspaces and have different memory requirements.
[Source][q35-scaling]

## Evidence and updates

The links below preserve the original reports at the pre-cleanup documentation
revision `60e7306`. Raw packages, profiles and tensor dumps remain in the
recorded ignored artifact locations; deleting a journal from the working tree
does not delete those artifacts. Historical commands require their recorded
source checkout, not current `main`.

When updating this page, identify the tested source/package, model, device,
capacity, workload and policy. Record required acceptance, native parity and
remaining diagnostics separately. A newly passing run supersedes only the
profile and cases it actually covers.

[q3-scaling]: https://github.com/tqxia/xinfer/blob/60e730611cdad7d312374d7d098eaa6e121ee449/docs/experiments/qwen3-refactor-validation.md#context-scaling-validation-2026-11-06
[q35-scaling]: https://github.com/tqxia/xinfer/blob/60e730611cdad7d312374d7d098eaa6e121ee449/docs/experiments/qwen35.md#full-model-context-scaling--2026-11-06
[q35-4b]: https://github.com/tqxia/xinfer/blob/60e730611cdad7d312374d7d098eaa6e121ee449/docs/experiments/qwen35.md#full-4b-l20-validation-2026-11-04
[default-context]: https://github.com/tqxia/xinfer/blob/60e730611cdad7d312374d7d098eaa6e121ee449/docs/experiments/common-runtime.md#model-derived-capacity-and-request-budgets--2026-11-07
[shared-weights]: https://github.com/tqxia/xinfer/blob/60e730611cdad7d312374d7d098eaa6e121ee449/docs/experiments/shared-weights.md
