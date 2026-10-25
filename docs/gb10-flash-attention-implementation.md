# FlashAttention on GB10: three implementation issues explained

> Historical implementation: the direct-ATen exporter described here has been
> removed. `--attention-backend flash` now uses the [custom operator and graph
> pass](gb10-custom-attention.md). These explanations and diagnostic results are
> preserved; source links below are pinned to the original tested revision.

Documented on 2026-10-25. This describes the implementation in `add91ea` and the
validation recorded in `98da53f`, on branch `dev-gb10`. Code line references below
refer to those revisions. The tested environment is PyTorch 2.13.0+cu130,
Triton 3.7.1, Transformers 4.51.3, and NVIDIA GB10.

Moving attention to PyTorch's native FlashAttention required three changes:
expressing causality correctly for cached inference, keeping the KV layout
consistent at full capacity, and avoiding AOTI's size-one specialization of
dynamic mutation inputs. This note explains the problem, the evidence and the
code for each change.

These implementation issues are resolved in the tested environment. Numerical
acceptance is a separate, open issue: 17/56 full-context cache tensors exceed the
unchanged 5% limits, versus 9/56 for the fused baseline. The optional Flash profile
remains experimental. See the [benchmark and validation report](../benchmarks/gb10-flash-2026-10-24/REPORT.md)
for that distinction and the performance results.

## The lengths used throughout this note

We need to distinguish the number of queries from the number of populated keys:

| Symbol | Meaning |
|---|---|
| `T` | Number of new tokens supplied in this call; one during ordinary decode |
| `P` | Number of tokens already in the cache before this call |
| `L = P + T` | Valid cache length after writing this call's new K/V |
| `C = 1024` | Logical cache capacity |
| `D = 128` | Per-head feature dimension |

Qwen3-0.6B has 16 query heads and 8 KV heads. Queries have shape
`[1, 16, T, 128]`; each full cache tensor has logical shape `[1, 8, 1024, 128]`.
Attention should see K/V views of shape `[1, 8, L, 128]`.

| Call | `P` | `T` | `L` |
|---|---:|---:|---:|
| Initial one-token prompt | 0 | 1 | 1 |
| Initial 25-token prompt | 0 | 25 | 25 |
| First decode call after that prompt | 25 | 1 | 26 |
| Append two tokens after three cached tokens | 3 | 2 | 5 |

The initial one-token prompt has `T=1` **and** `L=1`. Most decode calls have
`T=1` but `L>1`. That distinction is central to issue 3.

## Issue 1: preserving causal alignment after removing the dense mask

### What the original mask did

The original wrapper compared every key position with each query's absolute
position. It therefore handled initial prefill, appended chunks and decode with
the same rule: a query can see itself and all earlier keys.

The original mask construction is retained in the efficient-attention branch of
[QwenStep.forward](https://github.com/tqxia/xinfer/blob/98da53fcabde17422bb4ed349a822a5b45b75e1e/compiler/model.py#L76). With the Flash path, the wrapper
omits that dense mask and relies on the attention operator's causal semantics.
Those semantics must still match the absolute-position rule.

### Why a single public `is_causal=True` setting is insufficient

Public `torch.nn.functional.scaled_dot_product_attention` uses upper-left
alignment for a non-square causal matrix. For query row `i`, it permits keys
`j <= i`.

Our queries describe the final `T` tokens of an `L`-token valid prefix. Their
absolute positions are `P + i`, so the required condition is:

```text
j <= P + i
  = L - T + i
```

For `P=3`, `T=2`, `L=5`, the distinction is visible directly. Here `1` means a
key is visible to the query:

```text
                              key positions
                              0  1  2  3  4

Upper-left alignment:
query row 0                   1  0  0  0  0
query row 1                   1  1  0  0  0

Required lower-right alignment:
query at absolute position 3  1  1  1  1  0
query at absolute position 4  1  1  1  1  1
```

For initial prefill, `T=L`, so both alignments agree. That makes testing only
initial prompts insufficient. For decode with `T=1, L=128`, upper-left alignment
would expose only key zero, while the query must see all 128 valid keys.

Public SDPA with `is_causal=False` is correct for single-token decode when K/V
already contain only valid keys. Initial prefill can use `is_causal=True`.
Multi-token appends need the offset-aware rule above.

### What the implementation uses

[flash_attention_forward](https://github.com/tqxia/xinfer/blob/98da53fcabde17422bb4ed349a822a5b45b75e1e/compiler/model.py#L15) calls the native PyTorch
FlashAttention operator directly:

```python
output = torch.ops.aten._scaled_dot_product_flash_attention.default(
    query, key, value, 0.0, True,
    return_debug_mask=False, scale=scaling)[0]
```

For **this native operator in the pinned Torch build**, the causal `True` argument
means lower-right alignment. This is the same operator used by
`torch.nn.attention.bias.CausalBias._dispatch` for its Flash lower-right path.
Its boolean semantics must not be confused with the public SDPA argument.

Calling this operator also avoids the shape-equality branch in the causal-bias
dispatcher, which distinguishes square and rectangular attention. The native
operator provides one causal rule for all three call types:

| Call | Why lower-right alignment is correct |
|---|---|
| Initial prefill | `P=0`; lower-right and ordinary triangular causality coincide |
| Single-token decode | The sole query is the final key position and sees the entire valid prefix |
| Appended chunk | Each query sees the old prefix and the appropriate part of the new chunk |

The adapter retains eight KV heads and uses the native operator's GQA support.
It avoids expanding K/V from eight heads to sixteen heads. Qwen's original
projections, Q/K normalization, RoPE and output projection remain in use.

[load_model](https://github.com/tqxia/xinfer/blob/98da53fcabde17422bb4ed349a822a5b45b75e1e/compiler/model.py#L96) registers the adapter, and
[_prefix_causal_mask](https://github.com/tqxia/xinfer/blob/98da53fcabde17422bb4ed349a822a5b45b75e1e/compiler/model.py#L33) prevents HF from constructing a
dense mask for this path. The adapter explicitly rejects padding/dense masks,
nonzero dropout and sliding-window attention; those cases are outside its current
contract. The runtime remains batch one with a contiguous populated prefix.

### How we verified it

The [attention test](https://github.com/tqxia/xinfer/blob/98da53fcabde17422bb4ed349a822a5b45b75e1e/compiler/tests/flash_attention.py#L44) constructs the
required mask independently:

```python
positions = torch.arange(nq, device='cuda') + nk - nq
mask = torch.arange(nk, device='cuda')[None, :] <= positions[:, None]
```

It compares Flash output with FP32 math SDPA using that explicit mask and GQA.
The nine cases include square prefill, single-token decode, and rectangular
chunks such as `T=2, L=3` and `T=510, L=1022`. It also changes future storage to
large positive/negative values and requires the Flash result to remain bitwise
unchanged.

The full-model [boundary cases](https://github.com/tqxia/xinfer/blob/98da53fcabde17422bb4ed349a822a5b45b75e1e/compiler/validate_cpp.py#L113) include a
two-token append immediately after a one-token prompt, plus chunks approaching
the full cache. These check that the operator's alignment agrees with the model's
absolute positions and actual cache updates.

This was a semantic requirement we handled during integration, rather than a
claim that we observed a particular wrong-token run from the naive mask setting.
The direct operator is an internal PyTorch interface; toolchain upgrades require
re-export and the same validation.

## Issue 2: the cache layout changed at the full-capacity boundary

### A shorter prefix keeps the original strides

A contiguous tensor with shape `[1, 8, 1024, 128]` has these strides, in elements:

```text
[1048576, 131072, 128, 1]
```

Taking `cache[:, :, :L, :]` changes its logical sequence length, but does not move
the next head's data. The head stride remains `1024 * 128 = 131072`.

For example, an `L=128` prefix has shape `[1, 8, 128, 128]`. A compact tensor of
that shape would have head stride `128 * 128 = 16384`. Our view instead skips the
unused capacity before reaching the next head, so it is non-contiguous.

When `L` reaches 1024, the same view covers the full original tensor and becomes
contiguous. Thus a dimension that we wanted to vary over one compiled range also
changed a layout property at the range's endpoint.

### What export rejected

The attention probe traced with a shorter prefix and produced this guard:

```text
(128 * L) != 131072
```

That condition excludes exactly `L=1024`. Export rejected the declared range
because not every supported value satisfied the generated guard. The original
message is preserved in [the diagnostic excerpts](gb10-flash-attention-diagnostics.txt).

This was a compiler layout-specialization issue. It did not mean that
FlashAttention could not compute a 1,024-key attention operation.

### What the padding changes

[allocate_cache](https://github.com/tqxia/xinfer/blob/98da53fcabde17422bb4ed349a822a5b45b75e1e/compiler/model.py#L110) allocates one extra storage position
per head for the Flash path, then exposes the same logical 1,024-position cache:

```python
storage_length = CACHE_LENGTH + (attention_backend == "flash")
shape = (1, config.num_key_value_heads, storage_length, config.head_dim)
return tuple(
    torch.zeros(shape, dtype=torch.bfloat16, device="cuda")[:, :, :CACHE_LENGTH, :]
    for _ in range(2 * config.num_hidden_layers)
)
```

The Flash cache strides become:

```text
[1049600, 131200, 128, 1]
             ^
             1025 * 128 elements between adjacent heads
```

Even the maximum valid prefix, `L=1024`, still has a gap before the next head.
The problematic contiguous/non-contiguous transition is gone. Attention still
sees at most 1,024 keys; the padding position is never a valid token.

The native runtime checks these strides in [Spec](https://github.com/tqxia/xinfer/blob/98da53fcabde17422bb4ed349a822a5b45b75e1e/runtime/runtime.h#L94) and
allocates matching buffers with `at::empty_strided` in
[Runtime's constructor](https://github.com/tqxia/xinfer/blob/98da53fcabde17422bb4ed349a822a5b45b75e1e/runtime/runtime.h#L138). It retains the same storage
across calls. No `.contiguous()` copy is introduced on the inference path.

Across 56 cache tensors, Python's full padded allocation adds 112 KiB. Native
`empty_strided` needs only the gaps between heads, not the trailing padding after
the last head, so its additional storage is 98 KiB. The usable capacity stays
1,024 tokens.

### What this fixes, and what it does not

Padding resolves the full-capacity layout transition. It does not, by itself,
resolve AOTI's separate size-one guard for dynamically shaped mutation inputs.
That is issue 3.

The full-model validator fills the cache in chunks and exercises both final
slots. The [native checks](https://github.com/tqxia/xinfer/blob/98da53fcabde17422bb4ed349a822a5b45b75e1e/runtime/tests/validate_native.cpp#L45) verify stable
addresses and unchanged cache regions outside each write. It then checks that
[overflow is rejected without mutation](https://github.com/tqxia/xinfer/blob/98da53fcabde17422bb4ed349a822a5b45b75e1e/runtime/tests/validate_native.cpp#L72).
Poisoned-future tests also verify that unused positions do not influence attention.

## Issue 3: AOTI rejected a one-token dynamic cache input

### The first interface made mutation inputs dynamic

The initial design passed valid-prefix K/V views directly to the compiled model:

```python
# Initial proposal, simplified. L is known by the caller.
k_input = full_k[:, :, :L, :]
v_input = full_v[:, :, :L, :]
valid_keys = torch.export.Dim("valid_keys", min=1, max=1024)
```

Each cache input had shape `[1, 8, L, 128]`. Inside the compiled function,
`TensorCache.update()` used `index_copy_()` to write new K/V into those inputs.
Their shapes were dynamic and their contents were mutated.

The required range included `L=1` for a one-token initial prompt. Ordinary decode
with `T=1` and a longer cache was not the failing boundary.

### Export's range and the generated runtime contract disagreed

The minimal mutation probe reported an exported symbolic range of `[1, 1024]`.
After AOTInductor compilation, the generated C++ wrapper contained:

```cpp
if (arg1_1_size[2] < 2) {
    // Report: expected it to be >= 2, but got: 1
    throw ...;
}
```

This is an excerpt with the exception construction shortened. `arg1_1` is the
first KV argument in that small probe. The full-model error identified
`input_handles[2]`, because input IDs and cache positions precede its first KV input.

Initial functional tests at length one happened to produce correct results.
Enabling `AOTI_RUNTIME_CHECK_INPUTS=1` exposed that the input violated the generated
contract. Correct output from one unchecked execution does not establish that
the compiled package supports that input.

The [saved diagnostic excerpts](gb10-flash-attention-diagnostics.txt) distinguish
what we established:

| Probe | Result |
|---|---|
| Read-only FlashAttention with size-one export support | Lengths 1, 2 and 1024 worked |
| Dynamic KV inputs plus `index_copy_` mutation | Native guard required KV length at least 2 |
| Dynamic KV inputs plus indexed assignment | The same guard remained |
| Fixed-size mutation inputs plus a shape-only extent and size-one export support | Lengths 1, 2 and 1024 passed strict checks |

This isolates the interaction with mutation in the tested build. We did not
identify the exact internal PyTorch compiler pass that narrowed the range, so
this note does not attribute it to a proven bug in one specific lowering.

### The final interface separates storage from the valid length

The compiled function now receives:

```text
input_ids:       CUDA int64 [1, T]
cache_position:  CUDA int64 [T]
K0, V0, ...:     CUDA BF16  [1, 8, 1024, 128], fixed sizes and padded strides
kv_extent:      CPU uint8  [L + 1], contents unused
```

The K/V mutation inputs no longer change shape. The changing length comes from
the extra tensor's shape metadata. Attention prefix views are created inside the
compiled model after writing the fixed-size buffers.

The `+1` encodes the length without making the dynamic input dimension equal to
one:

| Valid keys `L` | Extent shape `[L+1]` | Internal attention K/V shape |
|---:|---:|---|
| 1 | `[2]` | `[1, 8, 1, 128]` |
| 25 | `[26]` | `[1, 8, 25, 128]` |
| 1024 | `[1025]` | `[1, 8, 1024, 128]` |

The extra one is only an encoding in CPU metadata. It does not add a key, pad the
query, or change the attention result.

### Follow the length through the code

**Allocate the CPU buffer once.** In
[Runtime's constructor](https://github.com/tqxia/xinfer/blob/98da53fcabde17422bb4ed349a822a5b45b75e1e/runtime/runtime.h#L145):

```cpp
extent_ = at::empty(
    {spec.cache_length + 1},
    at::TensorOptions().device(at::kCPU).dtype(at::kByte));
```

This reserves 1,025 bytes. Its values are uninitialized because the compiled
model does not read them.

**Pass a view with the appropriate shape.** In
[Runtime::step](https://github.com/tqxia/xinfer/blob/98da53fcabde17422bb4ed349a822a5b45b75e1e/runtime/runtime.h#L157), `position_` is `P`, `count` is `T`, and
their sum is `L`:

```cpp
inputs.insert(inputs.end(), cache_.begin(), cache_.end());
if (spec_.prefix_attention)
    inputs.push_back(extent_.narrow(0, 0, position_ + count + 1));
```

`narrow()` creates a tensor view sharing the CPU allocation. The native loop does
not allocate a new 1,025-byte buffer for every token. The separate
[Python helper](https://github.com/tqxia/xinfer/blob/98da53fcabde17422bb4ed349a822a5b45b75e1e/compiler/model.py#L40) constructs equivalent inputs for export
and validation; it is not the native benchmark loop.

**Recover `L` from the shape.** In
[QwenStep.forward](https://github.com/tqxia/xinfer/blob/98da53fcabde17422bb4ed349a822a5b45b75e1e/compiler/model.py#L88), the cache adapter receives:

```python
kv_extent.shape[0] - 1
```

**Write full buffers, then return prefix views.** The relevant operations in
[TensorCache.update](https://github.com/tqxia/xinfer/blob/98da53fcabde17422bb4ed349a822a5b45b75e1e/compiler/model.py#L60) are:

```python
k.index_copy_(2, position, key)
v.index_copy_(2, position, value)
if self.valid_length is not None:
    return k[:, :, :self.valid_length, :], v[:, :, :self.valid_length, :]
```

The writes use the original absolute positions. The returned views share the
same storage but expose only `L` keys to FlashAttention.

**Export the extent, rather than the mutation input size, as dynamic.** In
[compiler/export.py](https://github.com/tqxia/xinfer/blob/98da53fcabde17422bb4ed349a822a5b45b75e1e/compiler/export.py#L57):

```python
inputs += (cache_extent(32),)
extent = torch.export.Dim("kv_extent", min=2, max=CACHE_LENGTH + 1)
shapes = ({} if args.kind == "decode" else {1: seq},
          {} if args.kind == "decode" else {0: seq},
          tuple({} for _ in inputs[2]), {0: extent})
```

The empty dictionaries for the cache tensors keep all their input dimensions
fixed. `{0: extent}` makes the extra CPU tensor's length dynamic. The example
length 32 is only a tracing example; the declared range is 2 through 1025 for the
extent, representing valid cache lengths 1 through 1024.

**Allow size-one internal views.** The export path also enables
[this option](https://github.com/tqxia/xinfer/blob/98da53fcabde17422bb4ed349a822a5b45b75e1e/compiler/export.py#L38):

```python
from torch.fx.experimental import _config as symbolic_config
symbolic_config.backed_size_oblivious = True
```

This experimental option helps preserve size-one intermediate prefix views.
The option alone did not solve the dynamic mutation-input case; the interface
change above was also necessary. Runtime lower-bound checks remain enabled in
validation. We did not disable a failing guard or change an error threshold.

### What the generated program actually reads

The final decode wrapper obtains the symbolic extent from tensor metadata:

```cpp
auto arg370_1_size = arg370_1.sizes();
int64_t s79 = arg370_1_size[0];
```

Later, it constructs the attention view shape:

```cpp
const int64_t int_array_14[] = {1L, 8L, (-1L) + s79, 128L};
```

The FlashAttention call receives these views of the original cache tensors.
The generated input check now requires extent length at least two, which is
correct even when the actual valid cache length is one. See the
[generated-code excerpts](../benchmarks/gb10-flash-2026-10-24/generated-code-excerpts.txt).

This shape encoding fits the existing tensor-input AOTI interface and supplies
a symbolic size without extracting tensor contents. C++ already knows the token
count; reading `.sizes()` is a host metadata operation. No GPU scalar `.item()`,
new GPU synchronization, or KV-data copy is introduced by this mechanism. The
generation loop's existing argmax/host-token synchronization is still present.

### How we verified the final contract

The final full-model run used `AOTI_RUNTIME_CHECK_INPUTS=1`. Its 26 cases cover
185 logit rows, including the initial one-token prompt, length-two prefill,
appended chunks, changing valid lengths, and the final legal cache slot.

The [extra chunk sequence](https://github.com/tqxia/xinfer/blob/98da53fcabde17422bb4ed349a822a5b45b75e1e/compiler/validate_cpp.py#L113) reaches valid lengths
1, 3, 16, 32, 63, 64, 128, 256, 512, 1023 and 1024, both with clean storage and
poisoned future slots. Native CLI [smoke checks](https://github.com/tqxia/xinfer/blob/98da53fcabde17422bb4ed349a822a5b45b75e1e/compiler/tests/smoke_cpp.py#L110)
also exercise prompt lengths 1, 2 and 512 with subsequent decode calls. Native
and Python AOTI outputs agree bitwise for the tested cases.

[Spec](https://github.com/tqxia/xinfer/blob/98da53fcabde17422bb4ed349a822a5b45b75e1e/runtime/runtime.h#L82) checks matching export options, valid-length
range and extent metadata before loading a compatible pair of Flash packages.
The implementation depends on the pinned Torch build's internal/experimental
interfaces; the same boundary validation is required after a toolchain change.

## Evidence and reading order

Start with [TensorCache.update](https://github.com/tqxia/xinfer/blob/98da53fcabde17422bb4ed349a822a5b45b75e1e/compiler/model.py#L60), then follow
[Runtime::step](https://github.com/tqxia/xinfer/blob/98da53fcabde17422bb4ed349a822a5b45b75e1e/runtime/runtime.h#L157) and the
[export shape declaration](https://github.com/tqxia/xinfer/blob/98da53fcabde17422bb4ed349a822a5b45b75e1e/compiler/export.py#L53).

The three fixes address distinct requirements:

| Mechanism | Requirement it satisfies |
|---|---|
| Native lower-right causal FlashAttention | Correct visibility for initial prefill, decode and appended chunks |
| One spare storage position per head | No layout transition at the 1,024-token boundary |
| Fixed-size mutation inputs and `[L+1]` extent | Valid length can change without a dynamic size-one mutation input |
| Size-one export support | Internal attention views can have one valid key |

The [diagnostic excerpts](gb10-flash-attention-diagnostics.txt) preserve the
original guard failures and successful small-probe results. The
[benchmark directory](../benchmarks/gb10-flash-2026-10-24/REPORT.md) preserves full
validation summaries, generated-code evidence and the measured performance.
Reproduction commands are in [GB10.md](../GB10.md#optional-flashattention-path).

The shape and causal checks pass. The separate full-cache BF16 numerical
criterion still fails, so a nonzero exit from the full numerical validator must
be read alongside its saved report rather than treated as an unresolved shape
guard or as complete numerical acceptance.
