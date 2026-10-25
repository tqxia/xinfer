# GB10: post-export read-only attention operator

The `flash-custom` profile exports ordinary Qwen SDPA, replaces its attention
subgraphs with `xinfer::flash_attention`, re-exports the transformed graph, and
compiles it with AOTInductor. The native custom operator is read-only; all cache
writes remain in the graph. See the [implementation guide](../../docs/gb10-custom-attention.md)
for code references, build commands and deployment requirements.

The integration is validated against the existing direct-ATen `flash` profile:
all 185 saved logit rows and all 56 final cache tensors are bitwise identical.
Decode performance is effectively unchanged. Prefill medians are 1.1–1.6% higher.
The existing BF16 numerical limitation remains, so this is an optional
experimental profile rather than a new numerical-acceptance baseline.

## Paired performance results

Host: NVIDIA GB10, PyTorch 2.13.0+cu130, Triton 3.7.1, BF16 Qwen3-0.6B.
Both profiles use the same rebuilt native benchmark binary and model revision.
The direct-Flash packages were reused without recompilation. Automatic GPU
clocks were retained; active-sample median SM clock was 2,502 MHz for both paths.

Four fresh-process rounds alternate baseline/candidate order. Each profile and
prompt length receives two warmup requests and five measured requests per round,
with 64 forced decode steps per request: 120 measured requests total and 20
samples per profile/length. First-use requests and package loading are measured
separately from these steady-state medians. All 16 before/after idle checks passed.
CUDA Graph replay is not used. AOTI's optional strict input checks were enabled
for correctness validation and left at the normal default for both timing paths.

| Prompt tokens | Direct Flash prefill, ms | Custom op prefill, ms | Change |
|---:|---:|---:|---:|
| 25 | 7.414 | 7.499 | +1.15% |
| 128 | 8.088 | 8.194 | +1.31% |
| 512 | 11.681 | 11.865 | +1.58% |

| Prompt tokens | Direct Flash decode, ms/token | Custom op decode, ms/token | Change |
|---:|---:|---:|---:|
| 25 | 7.969 | 7.976 | +0.10% |
| 128 | 8.091 | 8.087 | -0.05% |
| 512 | 8.275 | 8.278 | +0.03% |

Positive changes mean longer latency. These measurements demonstrate no decode
speedup from the custom-op boundary. All 65 predictions per measured request
match between profiles at each length, and repeats are deterministic. This
comparison isolates the custom-op integration relative to direct Flash; it does
not repeat the older efficient-attention/eager comparison.

[summary.json](summary.json) contains distributions, including p10/p90 and
throughput. Individual round files and telemetry are saved alongside it.
[environment.json](environment.json) records package, executable, custom-library,
workload and source hashes. Its Git HEAD is the pre-change revision `2d0ed9d`;
the measured implementation was uncommitted at run time and is committed with
this report. The captured source hashes were checked against those final sources.

## Correctness and native deployment

- Nine operator cases cover initial prefill, decode and rectangular chunks,
  including `(T,L)=(1,1)`, `(1,1024)`, `(2,3)` and `(510,1022)`. Results match
  native Flash exactly and pass the unchanged isolated FP32-reference criteria.
- Input immutability, output non-aliasing, future-slot poisoning and ignoring the
  CPU extent's contents pass. `torch.library.opcheck` passes, including a case
  with non-unit feature strides.
- Nine compiled small-model cases pass with `AOTI_RUNTIME_CHECK_INPUTS=1`, exact
  outputs, exact complete cache contents and stable cache addresses. Invalid
  lengths and unknown attention masks are rejected.
- Full native/Python/HF validation covers 26 cases and 185 logit rows. Native and
  Python AOTI agree bitwise, all top-1 predictions and greedy fixtures match HF,
  and cache writes, prefix/suffix preservation, reset/reuse, poisoned slots and
  overflow checks pass.
- Direct-Flash comparison checks hashes for all 26 native logit files and the
  complete final cache dump: all are identical.
- Native smoke tests pass three greedy fixtures, prompt lengths 1/2/512,
  generation limits and 15 rejection cases, including an incompatible custom-op
  ABI. The executable loads and runs with a sanitized environment without
  `libpython` or `libtorch_python`.

Evidence: [operator/compiled tests](unit-validation.json),
[full-model validation](validation-custom.json),
[direct-Flash parity](direct-flash-parity.json), and
[native smoke checks](native-smoke.json).

The unchanged full-cache numerical criterion still fails on **17/56 tensors**
against HF BF16. Worst cache relative L2 is 5.4408%; worst peak-relative error is
44.0549%. Worst logit-row relative L2 is 3.8835%. These are the same results as
direct Flash, not a regression introduced by the wrapper. The full numerical
validator intentionally exits nonzero after writing its complete report.

## Compiler and layout audit

Both entry points contain 28 custom attention calls and preserve 56 cache-write
nodes after rewriting. Their generated native wrappers each contain 28 proxy
executor calls. Every custom call receives the original K/V input handles:
there is no full-cache repacking between the update and custom attention.
See [graph-audit.json](graph-audit.json) and
[generated-code-excerpts.txt](generated-code-excerpts.txt).

The initial implementation omitted the layout tag, and Inductor inserted 56
full-cache repacking kernels per model call. Registering
`at::Tag::flexible_layout` removed those copies. The C++ implementation honors
that tag by handling non-unit feature strides when necessary; the normal Qwen
path uses the existing padded views directly. The final benchmark and validation
use this corrected registration.

Native Flash's singleton output strides also required normalization to match
the fake kernel's contiguous output contract. For the tested Qwen shapes,
native output is already logically contiguous, so normalization changes metadata
without copying output data. Neither change alters attention arithmetic.

The persistent cache layout, CPU `[L+1]` extent and experimental
`backed_size_oblivious` export setting are retained. Removing those mechanisms
is a separate experiment. The native operator library remains an explicit
deployment dependency outside the `.pt2` archive.

## Reproduction and raw evidence

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python compiler/benchmark.py compare \
  --baseline-packages artifacts/aoti-flash \
  --candidate-packages artifacts/aoti-flash-custom \
  --baseline-label aoti_flash --candidate-label aoti_flash_custom \
  --output artifacts/benchmark-flash-custom-new
```

Choose an absent output path. Run the build/export/correctness commands in the
[implementation guide](../../docs/gb10-custom-attention.md) before reproducing
from source. Decode and prefill export/compilation took 30.3 and 40.1 seconds in
this run; these are diagnostics with reused compiler caches, not cold-build
benchmarks.

Raw evidence on GB10 is under `artifacts/aoti-flash-custom/`,
`artifacts/custom-attention-unit-v2/`, `artifacts/three-path-flash-custom/`,
`artifacts/native-smoke-flash-custom/` and
`artifacts/benchmark-flash-custom-20261025/`. The earlier copy-inserting probe is
retained separately under `artifacts/aoti-flash-custom-copy-probe/` and
`artifacts/three-path-flash-custom-copy-probe/` for diagnosis.
