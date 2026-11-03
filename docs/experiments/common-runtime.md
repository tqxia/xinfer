# Common native runtime: 2026-11-03

Qwen3 and Qwen3.5 now use one native runner and one executable for each role:
`xinfer_aoti`, `xinfer_benchmark` and `xinfer_validate`. The model-specific
`main.cpp`, benchmark and validation entry points have been removed. Current
commands are in the [development guide](../development-tools.md) and
[environment workflow](../../environments/README.md).

## Ownership and model boundaries

`runtime/common/spec.h` centralizes normalized dimensions, token/request checks,
package-file validation, toolchain checks and actual-GPU compatibility.
`Runner<StatePolicy>` in `runtime/common/runner.h` owns an immutable model spec,
the canonical shared-weight pool, both package loaders, positions and call
counters. It builds token/position/extent inputs, selects decode for one token,
validates logits and delegates returned state to the policy. The GPU guard runs
before the weight pool is constructed. Loader destruction still precedes
destruction of the weights they borrow.

The model headers retain strict package ABI checks and state policies:

- Qwen3 owns 56 persistent KV buffers and receives only logits from a package.
- Qwen3.5 owns 48 state tensors. Its KV outputs must alias the caller buffers;
  conv/recurrent outputs replace their previous tensors. Its gated attention,
  partial RoPE and packaged DeltaNet kernels are unchanged.

New exports put `model_type="qwen3"` or `model_type="qwen35"` in both entry JSON
files. The dispatcher rejects missing-one-entry, mismatched and unknown tags.
Existing untagged format-2 bundles use the explicit model identity already
embedded in `weights.pt`. That read selects a candidate adapter; the normal
archive and package SHA-256 checks still run before CUDA weight allocation.
Model selection does not infer a family from vocabulary size or tensor counts.
This refactor requires rebuilding the executables, but does not require
re-exporting the existing supported Flash/shared-weight bundles.

## Shared frontends and retained traces

The generation CLI has one input/budget contract and refuses existing output or
trace files and path aliases. Both models report consumed positions, stop reason
and load/generation timings. Native benchmarks use one report header and the
existing common measurement loops; reset, prefill, decode and whole-request
boundaries are unchanged.

Both models use one CPU intra-op thread. Single-token input IDs are created
directly on CUDA with `at::full`; multi-token inputs are copied from the host.

The validation binary normalizes the existing manifests into one case execution
loop. State assertions follow each normalized state descriptor: KV storage and
past/future regions are preserved; functional input state remains unchanged.
`stable_kv_storage` applies to both models. The older `stable_storage` flag remains
specific to Qwen3, whose entire state is persistent.

Small trace-layout adapters preserve the established files: Qwen3 FP32 logit
traces and optional KV dumps, and Qwen3.5 interleaved BF16 logits/mixed-dtype state
dumps. This avoids migrating Python numerical fixtures or inflating Qwen3
validation into full-state dumps after every token. There is no second
model-specific execution loop. Forced EOS remains an explicit validation-case
override, separate from the runner's owned spec.

At the refactor commit, `block_main.cpp` used `common/io.h` for shared checks,
JSON and tensor output. Its standalone fixture ABI was preserved for the checks
below. The legacy block runner and its Python validator were subsequently
retired; their last containing commit is `a63eb21`. The fused-GDN validation
target remains available.

## L20 validation

The retained Torch 2.13/CUDA 12.6 Flash bundles were used directly, without
re-exporting their graphs or weights.

| Check | Result |
| --- | --- |
| Native build | Three common executables plus block/fused fixture and ownership-test targets build |
| Spec ownership | Three lifetime/mutation scenarios pass for each model |
| CPU regressions | 50 unit tests pass; 10 also pass with `python -O` |
| Qwen3 native outputs | 26 cases; 27 data files, 229,873,152 bytes, match the retained golden SHA-256 hashes |
| Qwen3.5 native outputs | 35 cases; 41 data files, 3,737,586,688 bytes, match the retained golden SHA-256 hashes |
| Qwen3 CLI | 39 rejection checks, three greedy fixtures, four prompt/budget boundaries and the declared-size control pass; logits match Python AOTI bitwise |
| Qwen3.5 CLI | 53 rejection checks pass; an explicitly tagged two-token run matches its baseline trace byte for byte |
| Both benchmark harnesses | One round per model, lengths 2/128, two forced decode tokens, one warmup and one repeat pass |

The benchmark runs used `--allow-busy` on a shared GPU. They establish harness
functionality and make no latency or throughput claim. Benchmark validation
report package/archive hashes remain mandatory; a stale Qwen3.5 report was
rejected before measurement. The run with the matching report passes its
functional checks and retains `numerical_acceptance=false` from the existing
Flash/HF comparison; it is not a new numerical failure from this refactor.

The refactor changes host-side ownership and dispatch, with no graph, kernel or
weight changes. Full HF numerical validation was not rerun. Existing HF/compiler
numerical acceptance remains open; native output parity does not close it.

Reports and output digests are retained under
`artifacts/l20/common-runtime-20261103/`. Preserve the package and source hashes
with those reports when reproducing the comparison.

The single-token CUDA input path was additionally checked with four retained
cases per model, covering fresh decode, prefill/decode transitions, forced or
mixed token sequences and greedy generation. All output hashes and native
storage/call reports match. These focused checks are retained under
`artifacts/l20/decode-fastpath-20261103/`.
