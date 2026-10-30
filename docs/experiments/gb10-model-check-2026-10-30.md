# GB10 model validation — 2026-10-30

The branch `dev-gb10-20261030` starts from `origin/main` at
`7ca1cf1cde7818a96dc84e193b2ca7edd03270e2`. This run checks the two supported
model families using Qwen3-0.6B and Qwen3.5-0.8B, batch one, BF16, on GB10
(SM121). The [run guide](../gb10.md) contains the environments and commands.
This run checks correctness and deployment; it does not measure performance.

Both models run through the native C++ executables on GB10. Qwen3 passes its
current repository gates. Qwen3.5 passes native/Python functional parity after
two export fixes, but its strict HF numerical acceptance remains open: state
errors exceed the existing limits and one chat fixture differs in punctuation.
Neither model has complete HF cache-state parity.

## Environment and provenance

Both environments use Python 3.12.3, Torch 2.13.0+cu130, CUDA 13.0 and Triton
3.7.1. Qwen3 retains Transformers 4.51.3; Qwen3.5 uses a separate environment
with Transformers 5.20.2, sharing the same Torch/CUDA binaries. Both environments
pass `pip check`. The fresh CMake build is `work/gb10-20261030/build`.
All seven native model/validation/benchmark binaries resolve their libraries
without linking `libpython` or `libtorch_python`.

Qwen3 reuses the verified checkpoint at
`c1899de289a04d12100db370d81485cdf75e47ca`. Qwen3.5 revision
`2fc06364715b967f1860aea9cf38778875588b17` was downloaded on the Mac, checked
against pinned official metadata, transferred to GB10 and checked again.
CPU extraction retained 320 text tensors; the extracted safetensors SHA-256 is
`60be29c75ea732b08a4ee0013d6e89e488f53809b95dd4e1b7713dae113ea45d`.

Raw evidence remains under `artifacts/gb10-20261030/`; logs, helper scripts,
environments and build products remain under `work/gb10-20261030/`.

## Qwen3-0.6B

Fresh decode/prefill packages use the default fused FlashAttention profile.
The three-path validator passes all 26 cases / 185 logit rows. Native and
Python AOTI logits and full cache states are bitwise identical. Against HF,
maximum logit relative L2 is 3.88345% and peak-relative error is 2.59901%;
there are no top-token or greedy-generation differences.

The existing diagnostic HF cache limits are not all met: 17 of 56 cache tensors
exceed a 5% relative metric, with maxima of 5.4408% L2 and 44.0549% peak.
Consequently the report has `passed=true` but `hf_parity_passed=false` and
`hf_reference_limits_passed=false`, following the current main-branch policy.
This should not be described as complete HF cache parity.

The native smoke suite passes three generation fixtures (43 generated tokens),
prompt boundaries 1/2/512, a one-token budget and 22 input/package rejection
cases. Native output includes “The capital of France is Paris.” and
“17 + 25 = 42”. No Qwen3 source change was required.

Evidence:

- `qwen3/three-path/summary.json`
- `qwen3/native-smoke/summary.json`
- `qwen3/result.json` and `qwen3/validation-runs.json`

## Qwen3.5: singleton chunk-workspace specialization

The two issue explanations were expanded on **2026-10-31**. The production
fixes are in commit `9c2fe9b`. They address different compiler stages:

| Issue | First failing stage | Observable result |
|---|---|---|
| Chunk-workspace specialization | AOTInductor lowering/package guards | Valid prompts longer than 64 tokens are rejected |
| Lost KV-cache updates | `torch.export`, before projection packing or AOTI | Attention reads old caches and generation is incorrect |

### How a dynamic prefill became restricted to 64 tokens

[`export_step()`](../../compiler/qwen35/export.py) uses an example with T=32
for prefill but explicitly declares `torch.export.Dim("tokens", min=2, max=512)`
(lines 52–63 at the fix commit). The exported graph retains that range and the
symbolic T<=97 DeltaNet dispatch condition. Nevertheless, the compiled package
contains a guard rejecting T>64. JIT kernels work at T=97, 98 and 512; the
package rejects those inputs before executing them.

The long-prefill branch uses chunks of 64 tokens. Its temporary tensors
originally had a leading chunk-count dimension, for example:

```python
chunks = (tokens + 63) // 64
shape = (chunks, heads, 64, dim)
u = torch.empty(shape, device=q.device, dtype=torch.float32)
```

At the T=32 tracing example, `chunks` has the concrete example value **one**.
Freezing that derived dimension at one would impose:

```text
ceil(T / 64) == 1  =>  T <= 64  (for positive T)
```

That is the restriction observed in the generated wrapper:

```cpp
if (arg0_1_size[1] > 64) {
    // Throws: "expected to be <= 64"
}
```

Both conditional branches must be represented in the compiled program, even
though T=32 selects the short branch at runtime. The workspace problem arises
while lowering the long branch. Changing only the tracing example to T=128
removes the bad guard; its derived chunk count is two.

These controls isolate the problem to the workspace shape representation and
are consistent with singleton layout/stride specialization. **The exact
internal Torch call that introduced the guard was not captured.** This is a
more limited diagnosis than the confirmed stale-table mechanism below.

### Why flat scratch storage fixes it

[`launch_chunked()`](../../compiler/qwen35/kernels/chunked.py), lines 123–144,
now allocates the same storage as one-dimensional tensors:

```python
# Before: the leading dimension can have example value 1.
shape = (chunks, heads, CHUNK_SIZE, dim)

# After: the same number of FP32 elements in a flat workspace.
shape = (chunks * heads * CHUNK_SIZE * dim,)
u, w, corrections = (torch.empty(shape, device=q.device, dtype=torch.float32)
                     for _ in range(3))
```

The same change applies to `inverse`, `prefix` and `incoming`: six buffers in
total. Their Triton producers and consumers already calculate flat pointer
offsets. For example, `_transforms()` computes
`(chunk * H + head) * C * D + row * D + column`; it does not read the Python
workspace's shape or strides. Flattening therefore preserves allocation size,
dtype, element ordering, launch grids and kernel math while removing the
separate size-one dimension from the metadata.

The tracing example remains T=32 and the recurrent/chunked threshold remains
97. No padding or numerical-tolerance change is involved.

### Verification and evidence

After this change, BF16 and FP32 JIT, strict export and AOTI checks pass at
T=1,2,63,64,65,97,98,127,128,129,511,512, including carried state through
97→98→1→2. Repeated outputs are bitwise equal, inputs are unchanged, convolution
history is exact, and outputs have fresh storage. These checks establish
backend/functional parity, not independent HF numerical acceptance.

Evidence:

- `qwen35-kernel-smoke/prefill.wrapper.cpp.txt:1408`: original T<=64 guard
- `qwen35-kernel-smoke-128/`: larger-example control
- `qwen35-kernel-smoke-flat/summary.json`: original T=32 example after the fix

## Qwen3.5: lost KV-cache updates after symbolic control flow

### Where the error first appears

The first complete packages ran but answered the arithmetic prompt with
`122222222222222222222222`. HF answered `42`. The prefill logit relative L2
error was approximately 145%, so this was a correctness failure, not a small
near-tie token difference.

A stagewise trace compared HF, the direct eager adapter, unpacked exported
eager, packed exported eager and AOTI, using identical fed tokens. The direct
adapter agreed closely with HF. The large error was already present in the
unpacked exported graph; packing and native compilation were not the cause.

The first three DeltaNet layers' states matched bitwise. The first full-attention
layer, layer 3, exposed the fault: exported key/value caches remained zero.
All six attention-layer cache pairs remained zero. The graph computed
`index_copy` results but discarded them, read the original caches for attention,
and returned the original caches.

The relevant model path is:

1. [`TextStep.forward()`](../../compiler/qwen35/model.py), lines 147–162,
   creates `TextCache` before executing the layer loop.
2. The first three layers are DeltaNet layers. Their adapter calls
   [`fused()`](../../compiler/qwen35/kernels/recurrent.py), whose prefill path
   uses `torch.cond(tokens <= SHORT_THRESHOLD, ...)` at line 172.
3. Layer 3 is the first full-attention layer. It calls `TextCache.update()`
   after those conditional operations.
4. `TextStep.forward()` returns logits and `cache.tensors()` as explicit tensor
   outputs. The Python cache object is temporary bookkeeping for one forward.

### Tensor creation and Python attribute assignment are separate operations

The original `TextCache.update()` was:

```python
layer.keys = layer.keys.index_copy(2, self.position, key)
layer.values = layer.values.index_copy(2, self.position, value)
return layer.keys, layer.values
```

`index_copy()` creates a new tensor; it leaves the original tensor unchanged.
Assigning `layer.keys` then changes which tensor the Python attribute refers to.
Export must translate that bookkeeping change into graph edges: subsequent
attention and the returned state must consume the newly computed tensors.

The bad graph preserves the tensor computation but loses the effect of the
attribute assignment. Its dataflow is equivalent to:

```python
K1 = index_copy(K0, positions, key)
V1 = index_copy(V0, positions, value)
attention(query, K0, V0)     # Still reads the original caches.
return logits, K0, V0       # Still returns the original caches.
```

The retained `qwen35/packages/prefill.graph.txt` makes this directly observable:

- Lines 702 and 705 compute `index_copy` results and immediately discard them.
- Lines 708 and 715 expand the original `state_6` and `state_7` for attention.
- Line 4198 returns those original cache inputs, including all later K/V pairs.

For a fresh request, those original tensors are zero-filled. The mistake
therefore affects attention during prefill itself, as well as the cache passed
to later decode calls. The confirmed failure is in dynamic prefill: the T=1
decode path selects the short kernel directly and does not execute this
symbolic condition, but generation inherits the corrupted prefill state.

### Confirmed cause: a cached attribute writer points to a stale side-effect table

The follow-up CPU probe on 2026-10-31 identified the specific mechanism in the
installed **Torch 2.13.0+cu130**, with recorded Torch Git version
`cf30153c4c131c8164ee7798e5022d810682e2cb`.

Dynamo has a `SideEffects` table that records Python bookkeeping during
tracing, such as “this object's `keys` attribute currently refers to tensor
K0.” An object's cached `__dict__` proxy holds a reference to that table.
Tracing a higher-order conditional saves/restores side-effect bookkeeping
around branch tracing by cloning and replacing the current table.

The stale reference arises in this sequence:

| Step | Current table | Cached attribute writer | Consequence |
|---|---|---|---|
| Construct the cache before `cond` | A | A | Initial K0/V0 references are recorded correctly |
| Trace `cond` branches and restore bookkeeping | B, a restored clone | A | The proxy keeps its old table reference |
| Assign `layer.keys = K1` afterward | B | Writes K1 into A | Current table B still records K0 |
| Read `layer.keys` afterward | Read from B | — | Export retrieves K0 and wires the old tensor into the graph |

The failing assignments happen **after** the conditional returns. They are
not cache mutations inside a conditional branch.

The installed Torch source establishes each part of this path. The following
paths are relative to
`/home/tqxia/workspace/xinfer/.venv/lib/python3.12/site-packages/torch/_dynamo/`;
line numbers refer to this recorded installation:

| Source | Relevant behavior |
|---|---|
| `variables/user_defined.py:1694–1697`, `get_dict_vt()` | Caches the object's dictionary proxy in `self.dict_vt` |
| `variables/dicts.py:1491–1494`, `SideEffectsProxyDict.__init__()` | Captures `self.side_effects = tx.output.side_effects` |
| `variables/higher_order_ops.py:1516–1541` | Clones the table before subtracing, then replaces `tx.output.side_effects` with the restored clone |
| `side_effects.py:306–324`, `SideEffects.clone()` | Copies the attribute-mutation dictionaries while retaining tracked-variable references |
| `variables/user_defined.py:2937–2944` and `variables/dicts.py:1519–1524` | Attribute assignment uses the cached proxy, whose setter writes through its captured table |
| `variables/user_defined.py:3256–3265` | Attribute lookup checks the current `tx.output.side_effects` table |

The essential mismatch is:

```python
# Cached attribute writer:
self.side_effects.store_instance_dict_attr(self.item, name, value)

# Attribute reader:
result = tx.output.side_effects.load_attr(self, name, deleted_ok=True)
```

CPU instrumentation confirmed that the writer's captured table and the active
table are the same during cache construction but become different for the K/V
assignments after `cond`. The current-table read path is established by the
installed source and is consistent with the original tensors appearing in the
exported graph. This diagnosis applies to the observed installation; it does
not establish which other Torch builds are affected or whether upstream has
already changed the implementation.

### Why updating existing list entries preserves the cache

The fixed [`LayerState` and `TextCache`](../../compiler/qwen35/model.py),
lines 93–139, initialize a list before the condition and subsequently update
its entries:

```python
# LayerState.__init__:
self.kv = [a, b]

# TextCache.update:
new_key = layer.kv[0].index_copy(2, self.position, key)
new_value = layer.kv[1].index_copy(2, self.position, value)
layer.kv[0], layer.kv[1] = new_key, new_value
return new_key, new_value
```

`TextCache.tensors()` also reads those same list entries. Dynamo's
`ListVariable.sq_ass_item_impl()` in `variables/lists.py:1224–1246` changes
`self.items[idx]` directly and records mutation through the **current**
`tx.output.side_effects`. The update therefore avoids the stale attribute
writer. The existing DeltaNet conv/recurrent states already use this form of
list-entry bookkeeping.

Replacing the whole list would still use the failing attribute-store path:

```python
layer.kv = [new_key, new_value]        # Attribute replacement: reproduces failure.
layer.kv[0], layer.kv[1] = new_key, new_value  # Existing list entries: works.
```

In the fixed `qwen35/packages-v2/prefill.graph.txt`, lines 708 and 715 now feed
`index_copy` and `index_copy_1` into attention, and line 4198 returns the updated
tensors. The external tensor ABI, cache ordering, functional `index_copy`
semantics and input immutability are preserved. Only the temporary Python
representation used during tracing changes.

### Reproduction and verification

The maintained [CPU regression](../../compiler/qwen35/tests/test_text_cache.py)
uses the actual production `TextCache`, created before a symbolic condition,
then calls its update through a tiny submodule. It checks returned attention
tensors, final cache outputs and input immutability. The original implementation
fails all four lengths T=2,3,4,6; the fix passes both conditional branches.

```bash
CUDA_VISIBLE_DEVICES='' work/gb10-20261030/qwen35-venv/bin/python \
  -m unittest compiler.qwen35.tests.test_text_cache
```

The separate instrumentation probe uses minimal `Cache`/`State` classes to
compare the two bookkeeping representations and record table identities:

| Probe variant | Tracking observed at update | Eager output sums | Exported output sums |
|---|---|---|---|
| No `cond`, attribute replacement | Captured and current tables agree | 15, 75, 15, 75 | 15, 75, 15, 75 |
| `cond`, attribute replacement | Writer uses old A while B is current | 21, 81, 21, 81 | 0, 0, 0, 0 |
| `cond`, existing list-entry updates | List mutation uses current tracking | 21, 81, 21, 81 | 21, 81, 21, 81 |

The four sums correspond to K/V returned to attention and K/V returned as final
state. The selected conditional branch adds one to its input, so compare eager
and exported sums within each row; the no-condition control intentionally has
different sums. They are compact diagnostic evidence; the maintained regression
uses exact tensor comparisons. CUDA was never initialized during the probe, and
its instrumentation changed only in-memory Python methods in that process.

The ignored helper and evidence are retained on GB10:

```bash
CUDA_VISIBLE_DEVICES='' work/gb10-20261030/qwen35-venv/bin/python \
  work/gb10-20261030/explain_kv_side_effects.py
```

`work/gb10-20261030/explain-kv-side-effects.json` is explicitly a compact
transcription of the captured output. The script prints fresh records; it does
not write that JSON file. Table IDs are process-specific, so compare identity
agreement rather than expecting the recorded numeric IDs to repeat.

The full-model unpacked exported-eager control then matches the direct adapter
bitwise for all three arithmetic-trace logit rows and all 48 prefill state
tensors. All 12 K/V tensors contain nonzero data, and the predicted IDs are
`[19, 17, 248046]`, matching the HF trace for “42” followed by its separator.
The adapter's remaining logit L2 differences from HF are approximately 2.74%,
1.45% and 2.29%; export introduces no additional difference in this control.

The first packages in `qwen35/packages/` and their traces are retained as failed
evidence. Use the rebuilt `qwen35/packages-v2/` pair for validation and inference.

Evidence: `qwen35/arithmetic-trace/` and the discarded `index_copy` nodes in
`qwen35/packages/prefill.graph.txt`; the fixed control is
`qwen35/arithmetic-trace-v2/exported-vs-adapter.json`.

## Validation resource constraint

An unrelated VLLM engine reserves approximately 103 GiB of GB10's unified
memory. Standalone Python and standalone native inference each fit, but the
original validator retained large CPU state snapshots while launching a native
child. That child exhausted the remaining memory. The existing server was left
running.

The ignored `run_qwen35_mapped_validation.py` helper replaces only the validator's
CPU snapshot copies with file-backed tensor storage. Exact bits, shapes, dtypes,
container structure and copy-on-write behavior were tested independently. It
keeps all validation workloads, assertions and numerical limits, and does not
drop the system's global page cache.

This is sufficient for the runtime suite. The larger end-to-end suite still
exhausted memory at its native child, after completing HF greedy and Python
AOTI stages. It therefore executes the same native case manifest first in a
standalone process, then reruns the maintained Python validator. A narrow
interception replaces its one native launch with checksum-verified saved
outputs. Binary, both packages and metadata, validator source, case-manifest
bytes and all native output files are verified. Every original comparison and
acceptance threshold remains active; native execution and Python inference no
longer occupy GPU memory simultaneously. Both failed parent/child attempts are
retained as resource-failure evidence.

## Final Qwen3.5 results

The rebuilt `packages-v2` pair uses the same default 48 packed projection
groups (187→97 linear calls), with optional Inductor tuning disabled. The
prefill graph retains 18 symbolic conditions and both DeltaNet branches with
T=2..512. Both package metadata files match the fixed source hashes.

The complete native runtime validator passes:

- 35 cases; 5,243 teacher-forced tensors bitwise equal to Python AOTI.
- 30 greedy logit rows and 288 final-state tensors bitwise equal.
- All 27 invalid-input/package rejection checks pass.
- CLI generation, repeated requests, dispatch, context limit/overflow and
  Python-free native loading pass.
- Native arithmetic output is `42\n`; capital output is ` Paris.\n`.
- Native Chinese output is `编译器是将源代码翻译成机器码的自动化过程。\n`.
- All three chat fixtures terminate at EOS.

The mapped runner stores 3,707,788,288 bytes in 113 snapshots for this runtime
suite and completes without the previous native-loader OOM. Evidence:
`qwen35/runtime-v2/summary.json` and
`qwen35/runtime-v2-mapped-scratch/runner.json`.

The full end-to-end comparison also completes, with native/Python bitwise
parity across **294 teacher-forced steps / 14,406 tensors**. It covers T=1..512
boundary fixtures, both dispatch branches, carried state, poisoned future K/V,
repeat requests and the 1,024-token context boundary. Independent HF and FP32
reference stages execute successfully.

| Logit comparison | Rows | Maximum relative L2 | Maximum peak-relative error |
|---|---:|---:|---:|
| AOTI vs HF BF16 | 294 | 3.26182% | 2.83019% |
| AOTI vs FP32 | 100 | 2.61545% | 2.81150% |
| HF BF16 vs FP32 | 100 | 2.51527% | 1.20762% |

**Strict HF numerical acceptance still fails.** Under the unchanged 5% limits,
4,510 convolution-state and 15 recurrent-state comparisons fail. No logit or
K/V-cache comparison fails those numerical limits. The teacher-forced trace
has one AOTI/HF top-token difference, at `capital_teacher`, step 1. Its HF
reference margin is 0.125. The independent FP32 control has a top-token
mismatch at that same step, with margin approximately 0.09289.

Arithmetic and Chinese greedy token sequences match HF exactly. The capital
fixture differs: native/AOTI generates ` Paris.\n`, while HF generates
` Paris\n`. The first difference is the period at generated-token index 1;
the reference score gap to that token is 0.125. This is retained as a failure,
even though both answers are meaningful. The final validator exits 1 for these
numerical/token differences, with `native_bitwise=true` and `passed=false`.
Across the 100 independent FP32 control steps, both AOTI-vs-FP32 and
HF-BF16-vs-FP32 have 1,707 conv and four recurrent threshold crossings.
Those controls record state sensitivity even in the HF reference; they do not
replace the BF16 acceptance requirements or establish the cause of every error.
No threshold was relaxed and the earlier open Qwen3.5 numerical issues are not
considered resolved by this GB10 compatibility work.

Completed evidence:

- `qwen35/end-to-end-v2-sequential-r2/summary.json`
- `qwen35/end-to-end-v2-sequential-r2-mapped-scratch/runner.json`
- `qwen35/end-to-end-v2-native-receipt.json`
- `qwen35/end-to-end-v2-native-standalone/`
- `qwen35/result.json`

The original ordinary runtime attempt and mapped end-to-end attempt failed at
native loading under memory pressure; neither is an acceptance result. The
first sequential wrapper attempt also stopped before model execution because
argument abbreviation confused its `--native-receipt` option with the forwarded
validator's `--native` option. The ignored runner disables abbreviation, and
`sequential-r2` is the completed run reported above.
