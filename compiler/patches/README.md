# PyTorch compiler compatibility patches

These guarded workarounds support the project's pinned Torch 2.13 build. They
are applied explicitly by the compiler; importing this package applies no patch.
They do not change model arithmetic or resolve the outstanding HF/eager and
AOTI/eager numerical differences.

| Module | Problem | Applied by |
|---|---|---|
| [`triton_access.py`](triton_access.py) | Exponential traversal of shared Triton read dependencies | `compiler.qwen36.export` |
| [`aoti_compat.py`](aoti_compat.py) | Leaked handles for user-managed AOTI constants | `compiler.common.weights.compile_shared`, shared by model exporters |

Both context managers temporarily change process-global PyTorch internals. Use
them only around sequential compilation, without concurrent or nested independent
compiles. They restore their hooks even when compilation raises. Neither edits
installed PyTorch files. Unsupported versions or source layouts fail explicitly;
do not bypass the guards to make another build run.

## Triton read-dependency traversal

### Problem and local correction

`torch._higher_order_ops.triton_kernel_wrap.analyze_kernel_access` walks Triton
IR to identify which tensor arguments a user kernel reads or writes and whether
an epilogue may be fused into it. Its nested `_find_arg_access_count` expands
shared intermediate nodes once per dependency path. Repeated diamond-shaped
joins can therefore cause exponential work in a graph with only linearly many
nodes. The first complete Qwen3.6 prefill compile spent more than 15 minutes in
this analysis of the grouped FP8 path.

`memoize_triton_reads()` adds a visited-intermediate set to **read traversal
only**. In the guarded implementation, read counts are consumed only through
their keys, so this preserves the same read-reachability set. Write traversal,
write multiplicity, unknown-operation checks, mutation detection and epilogue
eligibility retain their original behavior. In particular, globally deduplicating
write paths could change a count from two to one and enable unsafe fusion.
This local correction does **not** fix exponential expansion of write paths.

The context checks Torch `2.13.0`, the exact analyzer source SHA256 and the
expected AST, including the keys-only read-count consumer. It retains the
original module globals and analyzer wrapper. Analysis caches are cleared on
entry and exit; the original analyzer function is restored on exit. Deployment
does not execute this compiler-time patch.

### Upstream status

Checked **2026-11-09**, against PyTorch main commit
[`e2c2673d00500d268fcb1cc63fa7cc3173876505`](https://github.com/pytorch/pytorch/commit/e2c2673d00500d268fcb1cc63fa7cc3173876505).
The traversal remains present in
[`triton_kernel_wrap.py`](https://github.com/pytorch/pytorch/blob/e2c2673d00500d268fcb1cc63fa7cc3173876505/torch/_higher_order_ops/triton_kernel_wrap.py#L1375-L1413).

- [Issue #193703](https://github.com/pytorch/pytorch/issues/193703) is the exact
  open issue. It contains an internal-IR reproducer and a proposed repair that
  memoizes parameter-count summaries with zero/one/many saturation, preserving
  write multiplicity. Its scope includes writes and is broader than this local
  read-only workaround.
- [PR #202164](https://github.com/pytorch/pytorch/pull/202164) is open and
  unmerged. Its implementation visits each value once per load/store site while
  changing loop-carried pointer analysis. This overlaps the repeated-traversal
  problem but also changes write-count semantics. Review its current state and
  regression coverage before preparing a separate upstream patch.
- [PR #193227](https://github.com/pytorch/pytorch/pull/193227) is open and
  complementary: it caches whole-kernel analysis across consumers and launch
  sites. Caching does not bound the first analysis of a shared dependency graph.

The upstream status above is a source/history check, not a full-model validation
of a new Torch build or an execution test of either open PR.

## AOTI user-managed constant handles

### Problem and local correction

Prefill and decode borrow tensors from one canonical shared weight pool using
`user_managed=True`. The pinned Torch implementation of
`MaybeOwningAtenTensorHandle` creates a shallow C++ tensor-handle copy but does
not assign that copy to its RAII owner. Destruction then leaves the copy and its
reference to the underlying storage alive. Tensor data is shared, not copied;
the leak is a retained reference, which can keep the entire weight pool live.
The original complete Qwen3.6 runtime retained approximately 35.7 GB of live
weight allocations after destruction.

`owned_constant_handles()` creates a temporary copy of
`torch/csrc/inductor/aoti_runtime/utils.h`. Its user-managed branch checks handle
creation and gives the shallow copy to `RAIIAtenTensorHandle`. The caller retains
its own handle and both handles continue to share tensor storage.

A temporary compiler hook prepends this header overlay only to AOT C++ include
paths. The context checks Torch `2.13.0`, git revision
`cf30153c4c131c8164ee7798e5022d810682e2cb`, the exact original header hash and the
expected include-hook structure. It restores the hook and deletes the overlay
on exit. Include order participates in the compiler cache key; preprocessed
header contents also select the corrected precompiled header.

Although Python installs this compiler hook, the ownership correction is compiled
into the package wrapper and works in Python-free C++ inference. **Recompile
both prefill and decode packages** to correct existing bundles. Rebuilding only
the native executable cannot repair wrappers already embedded in `.pt2` files.

### Upstream fix

This exact leak was fixed upstream by
[`d91366cc308b918b3dd3f8937ce6864fb440cadf`](https://github.com/pytorch/pytorch/commit/d91366cc308b918b3dd3f8937ce6864fb440cadf),
dated **2026-08-31**, associated with
[PR #194480](https://github.com/pytorch/pytorch/pull/194480).

The upstream fix removes `MaybeOwningAtenTensorHandle`, makes constant-map entries
RAII-owned and immediately transfers ownership of checked shallow handle copies
to the map. Entries release their handle when replaced or when the container is
destroyed. This preserves caller ownership and zero-copy tensor storage. It does
not change `free_inactive_constant_buffer()` into a mechanism for releasing all
non-folded user-managed constants.

The fix remains in the main commit checked above:
[`model_base.h`](https://github.com/pytorch/pytorch/blob/e2c2673d00500d268fcb1cc63fa7cc3173876505/torch/csrc/inductor/aoti_runtime/model_base.h#L1303)
uses RAII-owned constant-map entries and
[`model_container.h`](https://github.com/pytorch/pytorch/blob/e2c2673d00500d268fcb1cc63fa7cc3173876505/torch/csrc/inductor/aoti_runtime/model_container.h#L756-L764)
retains shallow handles with explicit ownership. The project's pinned build still
has the affected implementation; its version string alone does not establish
that it contains the upstream fix.

## Tests and retained evidence

Run the focused CPU checks from the repository root with the matching pinned
model environment:

```bash
Q36_PY=work/gb10/qwen36-venv/bin/python
CUDA_VISIBLE_DEVICES='' "$Q36_PY" -m unittest -v \
  compiler.patches.tests.test_triton_access \
  compiler.patches.tests.test_aoti_compat
```

The Triton tests compare original and patched access/fusion results, retain
duplicate write paths, cover nested calls, load-derived pointers, TMA and unknown
operations, bound visits on a depth-256 shared read graph, and check guards and
exception cleanup. The overlay tests check AOT-only include precedence, original
header preservation, compiler cache-key changes, guards and cleanup.

The existing shared-weight tests compile and load small real CPU packages,
checking shared storage and handle release through reference counts and weak
TensorImpl references:

```bash
CUDA_VISIBLE_DEVICES='' "$Q36_PY" -m unittest -v \
  compiler.common.tests.test_weights
```

The retained full-model GB10 experiment returned live ATen allocation exactly to
its warmed 32 MiB baseline after runtime destruction. Rebuilt packages matched
the original captured outputs in all 1,991 comparisons bitwise. These observations
establish ownership cleanup and output preservation for that experiment; cached
CUDA reservation and process RSS need not return to their initial values.
They do not establish HF or compiler/eager numerical acceptance.
See the [model workflow](../../docs/qwen36-fp8.md#native-memory-and-deployment-checks)
and [numerical experiment record](../../docs/experiments/qwen36-numerics.md).

## Reviewing or removing a patch

When changing Torch, inspect the installed source and the upstream implementation
before changing any version or hash guard. Retain the existing targeted
regressions. For the Triton patch, verify bounded traversal and unchanged access,
mutation and fusion decisions. For the ownership patch, verify actual compiled
loader lifetime and shared storage; with a build containing the upstream fix,
remove the overlay call from the shared compiler and recompile both packages.
Then rerun affected export and native numerical/lifetime checks on fresh bundles.
Do not use guard bypasses, changed references or relaxed tolerances as migration.
