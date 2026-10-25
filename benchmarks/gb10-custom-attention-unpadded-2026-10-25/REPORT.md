# GB10: remove custom-attention padding and size-oblivious export

New `flash-custom` exports use ordinary contiguous KV caches and leave
`backed_size_oblivious` disabled. The native read-only attention implementation
and its ABI are unchanged. The custom op creates its valid-prefix views at
runtime, so export no longer needs to reason about their layout transition at
length 1,024 or their size-one boundary.

All 185 validated logit rows and all 56 final cache tensors are bitwise identical
to the previous padded custom-op version. The existing numerical limitation
against HF BF16 remains: 17/56 full-cache tensors exceed the unchanged 5% limits.
This remains an experimental numerical profile.

## What changed

| Contract | Previous custom packages | New custom packages |
|---|---|---|
| Logical cache shape | `[1,8,1024,128]` | `[1,8,1024,128]` |
| Strides, in elements | `[1049600,131200,128,1]` | `[1048576,131072,128,1]` |
| Spare storage positions per head | 1 | 0 |
| `backed_size_oblivious` | Enabled | Disabled |
| Valid-length input | CPU uint8 `[L+1]` | CPU uint8 `[L+1]` |
| Custom operator ABI | 1, read-only | 1, read-only |

The native requested cache storage decreases by 98 KiB across 56 tensors, to
112 MiB. This is requested tensor storage, not a claim about allocator reservation
or total device memory. Logical capacity stays at 1,024 tokens.

The direct `flash` path retains its padding and size-oblivious setting. Older
custom packages remain loadable: metadata without `cache_padding` follows the
original padded contract. New metadata explicitly records zero padding. Native
allocation and Python validation follow the package's declared strides, rather
than assuming every custom package uses the current default.

Implementation: [cache allocation](../../compiler/model.py),
[export settings](../../compiler/export.py),
[native package validation](../../runtime/runtime.h), and
[usage guide](../../docs/gb10-custom-attention.md).

## Correctness

- Nine operator cases pass native-Flash bitwise parity, independent FP32 masked
  attention checks, read-only and non-aliasing checks, future-slot poisoning,
  extent-data independence and `torch.library.opcheck`. Non-unit feature strides
  are also covered.
- Nine compiled attention cases pass with contiguous unpadded inputs and
  `backed_size_oblivious=False`. Strict runtime checks include valid cache lengths
  1, 2 and 1,024, initial prefill and appended chunks; complete caches and outputs
  match exactly and addresses remain stable.
- Full native/Python/HF validation covers 26 cases and 185 rows, including changing
  valid lengths through 1, 3, 16, 32, 63, 64, 128, 256, 512, 1023 and 1024.
  Native/Python outputs match bitwise; all top-1 predictions and greedy fixtures
  match HF. Cache update regions, resets, reuse, poisoned slots and overflow pass.
- Hashes of all 26 logit files and the complete final cache dump match the saved
  padded custom baseline.
- The rebuilt native validator also runs the old padded packages through all
  26 cases with strict input checks. Its outputs match the saved padded baseline.
- Native smoke tests pass three fixtures, lengths 1/2/512, generation limits and
  19 rejection cases. New cases reject mismatched padding, unsupported padding,
  wrong strides and mismatched export settings. Python-free native loading passes.

The unchanged full-model validator exits nonzero after saving its complete report
because the same 17 cache tensors exceed the numerical threshold. Structural and
bitwise parity with the previous Flash implementation do not establish acceptance
against HF BF16.

Evidence: [operator tests](unit-validation.json),
[full-model validation](validation-unpadded.json),
[padded comparison](padded-parity.json),
[legacy runtime parity](legacy-parity.json), and
[native smoke tests](native-smoke.json).

## Compiler audit

Both packages record `cache_padding: 0` and
`export_options.backed_size_oblivious: false`. Each rewritten graph retains 28
opaque custom calls and 56 cache writes. Each compiled wrapper passes the original
cache handles to all 28 proxy-executor calls, without full-cache repacking.
The extent encoding and fixed-size mutation inputs remain in place.
See [graph-audit.json](graph-audit.json) and
[generated-code-excerpts.txt](generated-code-excerpts.txt).

## Paired GB10 benchmark

Baseline: the existing padded packages in `artifacts/aoti-flash-custom`.
Candidate: `artifacts/aoti-flash-custom-unpadded`. Both run in the same rebuilt
native binary with the same custom operator library, checkpoint and dependency
versions: PyTorch 2.13.0+cu130, Triton 3.7.1 and BF16 Qwen3-0.6B on NVIDIA GB10.

The protocol uses four fresh-process rounds in alternating order, two warmups,
five measured requests per length/profile/round and 64 forced decode steps.
There are 120 measured requests, with 20 samples per profile/length. Loading and
first-use requests are separate from steady-state timing. CUDA Graph replay is
not used; optional strict AOTI checks are enabled for validation and left at the
normal default for both timed profiles.

All 16 idle checks passed. Active-sample median SM clock was 2,502 MHz for
both profiles. All 65 predictions per request match at every length, and
repeated requests are deterministic.

| Prompt tokens | Padded prefill, ms | Unpadded prefill, ms | Change |
|---:|---:|---:|---:|
| 25 | 7.490 | 7.491 | +0.01% |
| 128 | 8.150 | 8.179 | +0.36% |
| 512 | 11.792 | 11.815 | +0.19% |

| Prompt tokens | Padded decode, ms/token | Unpadded decode, ms/token | Change |
|---:|---:|---:|---:|
| 25 | 7.967 | 7.965 | -0.02% |
| 128 | 8.092 | 8.091 | -0.01% |
| 512 | 8.274 | 8.285 | +0.13% |

Positive changes mean longer latency. All median latency changes are below
0.14% for decode and 0.36% for prefill. Performance is
effectively unchanged in this run; the benefit is removing compiler-specific
requirements, with a small reduction in requested cache storage.


Distributions and raw rounds are saved in [summary.json](summary.json) and the
neighboring files. [environment.json](environment.json) records package, workload,
binary, operator-library and source hashes. Its Git HEAD is the pre-change
`c99bcde`; the measured working sources are committed with this report. Their
hashes were verified against the final implementation.

## Reproduction

Follow the build/export commands in the [guide](../../docs/gb10-custom-attention.md),
using a fresh output directory for the unpadded packages. Then compare with the
preserved padded baseline:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python compiler/benchmark.py compare \
  --baseline-packages artifacts/aoti-flash-custom \
  --candidate-packages artifacts/aoti-flash-custom-unpadded \
  --baseline-label custom_padded --candidate-label custom_unpadded \
  --output artifacts/benchmark-flash-custom-unpadded-new
```

Decode and prefill export/compilation took 29.0 and 38.2 seconds in this run with
compiler-cache reuse. These are build diagnostics, not cold-build benchmarks.

Raw evidence on GB10: `artifacts/aoti-flash-custom-unpadded/`,
`artifacts/custom-attention-unit-unpadded/`,
`artifacts/three-path-flash-custom-unpadded/`,
`artifacts/native-padded-compat-unpadded-change/`,
`artifacts/native-smoke-flash-custom-unpadded/` and
`artifacts/benchmark-flash-custom-unpadded-20261025/`.
