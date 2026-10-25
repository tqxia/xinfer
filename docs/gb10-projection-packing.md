# Projection packing on GB10

The optional `--projection-packing` export flag packs independent Qwen3-0.6B
BF16 projections into larger linear operations. The default is `none`.
The [pass](../compiler/passes/projections.py) runs after `torch.export` and the
attention rewrite, then the module is re-exported before AOTInductor compilation.
It uses ordinary ATen linear operations and slices, with no new runtime operator.

The [GB10 experiment](../benchmarks/gb10-projection-packing-2026-10-25/REPORT.md)
found that packing both groups slows decode by 3.2–3.5%. QKV-only packing has
little effect on decode. Both tested full-model profiles introduce top-token
differences against HF; QKV-only also introduces logit tolerance failures.
Packing therefore remains experimental and disabled by default.

| Mode | Packed weight shape per layer | Linear calls per decode token, including LM head |
|---|---|---:|
| `none` | Original weights | 197 |
| `qkv` | `[4096,1024]`, rows ordered Q / K / V | 141 |
| `gate-up` | `[6144,1024]`, rows ordered gate / up | 169 |
| `all` | Both groups | 113 |

Packing concatenates weights once during export, storing them as frozen buffers.
It removes unused original projection weights from the transformed module, so
the compiled package does not also retain those unpacked weights. The source
model and checkpoint are not modified. No weight concatenation runs during
inference, and the total number of weight elements is unchanged.

Q/K/V outputs are views of widths 2048/1024/1024. Gate/up outputs are views of
width 3072 each. Q/K normalization, RoPE, SiLU, multiplication, output/down
projections and cache updates retain their existing graph operations. There is
no extra activation or quantization fusion in this pass. BF16 output boundaries
are preserved, but different matrix shapes can select different vendor kernels
and reduction orders, so bitwise parity is not assumed.

The pass is restricted to the known bias-free Qwen shapes. It checks every
group's layer index, shared activation input, weight shape/dtype/device and
single-consumer weight use before changing the graph. It rejects incomplete
groups, bias, shared weight consumers, or intervening side effects. Exported
tensor-metadata assertions remain in the graph and can be crossed because they
do not mutate tensors or change arithmetic state. Dead-code elimination retains
all cache writes.

## Run and validate

Use the existing GB10 environment and native build, from the repository root:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_COMPILE_THREADS=2

.venv/bin/python compiler/tests/projection_packing.py \
  --output artifacts/projection-packing-unit-new
.venv/bin/python compiler/export.py decode --attention-backend flash \
  --projection-packing all --output artifacts/aoti-flash-packed
.venv/bin/python compiler/export.py prefill --attention-backend flash \
  --projection-packing all --output artifacts/aoti-flash-packed
AOTI_RUNTIME_CHECK_INPUTS=1 .venv/bin/python compiler/validate_cpp.py \
  --packages artifacts/aoti-flash-packed --output artifacts/three-path-flash-packed \
  --collect-logit-errors --collect-top1-errors
.venv/bin/python compiler/benchmark.py compare \
  --baseline-packages artifacts/aoti-flash --baseline-label aoti_flash \
  --candidate-packages artifacts/aoti-flash-packed --candidate-label aoti_packed \
  --output artifacts/benchmark-projection-packing-new
```

Use fresh artifact directories for reruns. The unit suite covers all three
packing modes, decode and dynamic prefill lengths 2/25/128/512, exact cache writes,
source-weight immutability, frozen packed weights, absence of runtime packing,
and rejection of unsupported graph patterns. Full-model validation retains the
existing error limits. `--collect-logit-errors` records tolerance failures across
the suite. `--collect-top1-errors` also records top-token differences with their
HF score gaps. Both are diagnostic options: the report remains failed and the
process exits nonzero when any collected comparison fails. Native/Python parity,
cache invariants and generation-fixture checks remain strict.

Metadata records the packing mode and transformation counts in each package;
`*.packing.json` and graph dumps expose the rewrite. Packing changes only internal
constants and computation. The existing runtime inputs, cache ABI, attention
operator and native binaries stay compatible. Prefill and decode may select
different packing modes; the existing model/revision/input-contract checks still
apply. Decode performance and prefill performance must be measured separately.
