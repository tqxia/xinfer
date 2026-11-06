"""Deterministic DeltaNet fixtures and HF numerical references for focused tests."""

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    causal_conv1d_fn,
    causal_conv1d_update,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

from compiler.common.tests.tensors import bits_equal, cpu_copy, row_errors
from compiler.qwen35.kernels.chunked import CHUNK_SIZE
from compiler.qwen35.kernels.recurrent import SHORT_THRESHOLD, fused, long, short

LENGTHS = (1, 2, 3, 4, 5, 31, 32, 33, 63, 64, 65, 127, 128, 129, 511, 512)
PARTITIONS = (1, 2, 3, 4, 5, 31, 32, 33, 64, 65, 272)
KEY_HEADS, HEAD_DIM, CONV_WIDTH = 16, 128, 4
STATE_FIELDS = ("conv", "state", "weight", "A_log", "dt_bias")
# Predeclared normalized error limits; FP32 reference uses the same input values.
LIMITS = {"bf16": 0.01, "fp32": 0.0001}
# Nearly orthogonal Q/K can make a whole output row tiny. Retain normalized
# metrics, but permit FP32 absolute roundoff below 1e-7 (see experiment record).
ABS_LIMITS = {"bf16": 0.0, "fp32": 1e-7}


def geometry(value_heads):
    """The two supported DeltaNet profiles share key heads, dimensions and taps."""
    if value_heads not in (16, 32):
        raise ValueError("Value heads must be 16 (0.8B) or 32 (4B)")
    return dict(
        key_heads=KEY_HEADS,
        value_heads=value_heads,
        head_dim=HEAD_DIM,
        conv_width=CONV_WIDTH,
    )


class TensorState:
    """The small part of HF's cache protocol used by this one block.

    Zero tensors represent a fresh request. Clone convolution storage because
    HF's decode convolution mutates it; the exported interface is functional.
    """

    def __init__(self, conv, recurrent):
        self.layers = [self]
        self.record_past = False
        self.conv_states = [conv.clone()]
        self.recurrent_states = [recurrent]

    def has_previous_state(self, layer_idx, state_idx=0):
        return True

    def update_conv_state(self, value, layer_idx, conv_kernel_size):
        combined = torch.cat((self.conv_states[0], value), dim=-1)
        self.conv_states[0] = combined[..., -conv_kernel_size:].contiguous()
        return combined

    def update_recurrent_state(self, value, layer_idx):
        self.recurrent_states[0] = value
        return value


class BlockStep(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden, conv, recurrent):
        cache = TensorState(conv, recurrent)
        output = self.block(hidden, cache_params=cache)
        return output, cache.conv_states[0], cache.recurrent_states[0]


class Fused(torch.nn.Module):
    def forward(self, qkv, b, a, conv, state, weight, A_log, dt_bias):
        return fused(qkv, b, a, conv, state, weight, A_log, dt_bias)


def reference(qkv, b, a, conv, state, weight, A_log, dt_bias, *, chunked=False):
    """HF preparation with either recurrent or chunked prefill arithmetic."""
    tokens = qkv.shape[1]
    if tokens == 1:
        next_conv = conv.clone()
        mixed = causal_conv1d_update(qkv.transpose(1, 2), next_conv, weight, None, "silu")
    else:
        combined = torch.cat((conv, qkv.transpose(1, 2)), dim=-1)
        next_conv = combined[..., -4:].contiguous()
        mixed = causal_conv1d_fn(combined, weight, None, activation="silu")[..., -tokens:]
    value_heads, dim = state.shape[1:3]
    key_heads = (qkv.shape[-1] // dim - value_heads) // 2
    q, k, v = mixed.transpose(1, 2).split(
        (key_heads * dim, key_heads * dim, value_heads * dim), dim=-1
    )
    q, k = (x.reshape(1, tokens, key_heads, dim) for x in (q, k))
    v = v.reshape(1, tokens, value_heads, dim)
    # Match the original HF module's grouped Q/K expansion before recurrence.
    if value_heads != key_heads:
        q, k = (x.repeat_interleave(value_heads // key_heads, dim=2) for x in (q, k))
    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    implementation = (
        torch_chunk_gated_delta_rule if chunked and tokens > 1 else torch_recurrent_gated_delta_rule
    )
    options = dict(chunk_size=CHUNK_SIZE) if chunked and tokens > 1 else {}
    core, final = implementation(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        **options,
    )
    return core, next_conv, final


def make_cases(value_heads=16):
    geometry(value_heads)
    channels = (2 * KEY_HEADS + value_heads) * HEAD_DIM
    cases = []
    for dtype_name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):

        def case(name, lengths=(1,), nonzero=True):
            return dict(
                name=f"{dtype_name}-{name}",
                dtype=dtype_name,
                steps=[
                    (
                        torch.randn(1, t, channels).to(dtype),
                        torch.randn(1, t, value_heads).to(dtype),
                        torch.randn(1, t, value_heads).to(dtype),
                    )
                    for t in lengths
                ],
                conv=(torch.randn(1, channels, CONV_WIDTH) * 0.3).to(dtype)
                if nonzero
                else torch.zeros(1, channels, CONV_WIDTH, dtype=dtype),
                state=torch.randn(1, value_heads, HEAD_DIM, HEAD_DIM) * 0.1
                if nonzero
                else torch.zeros(1, value_heads, HEAD_DIM, HEAD_DIM),
                weight=(torch.randn(channels, CONV_WIDTH) * 0.2).to(dtype),
                A_log=torch.full((value_heads,), -2.0, dtype=dtype),
                dt_bias=(torch.randn(value_heads) * 0.1).to(dtype),
            )

        for tokens in LENGTHS:
            for nonzero in (False, True):
                cases.append(
                    case(f"t{tokens}-{'state' if nonzero else 'zero'}", (tokens,), nonzero)
                )
        whole = case("whole", (512,))
        stream = [x.split(PARTITIONS, dim=1) for x in whole["steps"][0]]
        split = dict(whole, name=f"{dtype_name}-split", steps=list(zip(*stream)))
        cases.extend((whole, split, dict(split, name=f"{dtype_name}-split-repeat")))
        carry = case("carry", (128,) + (1,) * 64)
        cases.extend((carry, dict(carry, name=f"{dtype_name}-repeat")))
        gates = case("gate-extremes", (1,) * 8)
        for i, (_, b, a) in enumerate(gates["steps"]):
            b.copy_(torch.linspace(-80, 80, value_heads))
            a.fill_((-80, -20, -1, 0, 19, 20, 21, 80)[i])
        cases.append(gates)
        # Reuse whole's seeded data so adding this case leaves old fixtures unchanged.
        prefill_gates = dict(
            whole,
            name=f"{dtype_name}-prefill-gate-extremes",
            steps=[
                tuple(x[:, start:end].clone() for x in whole["steps"][0])
                for start, end in ((0, 65), (65, 66))
            ],
        )
        extremes = torch.tensor([-80, -20, -1, 0, 19, 20, 21, 80], dtype=dtype)
        for _, b, a in prefill_gates["steps"]:
            b.copy_(torch.linspace(-80, 80, value_heads))
            a.copy_(extremes[torch.arange(a.shape[1]) % len(extremes)].view(1, -1, 1))
        cases.append(prefill_gates)
        zero = case("zero-qkv")
        zero["conv"].zero_()
        zero["steps"][0][0].zero_()
        cases.append(zero)
        # Cancellation and non-exact products expose missing convolution rounding.
        cancellation = case("conv-cancellation", (5,))
        cancellation["conv"].copy_(torch.tensor([0.99609375, -1.0078125, 0.50390625, -0.49609375]))
        cancellation["weight"].copy_(
            torch.tensor([0.333984375, 0.333984375, -0.66796875, -0.66796875])
        )
        cancellation["steps"][0][0].fill_(0.498046875)
        cases.append(cancellation)

    # Slice after the original seeded fixtures so the 0.8B cases stay unchanged.
    for dtype in ("bf16", "fp32"):
        for state in ("zero", "state"):
            source = next(c for c in cases if c["name"] == f"{dtype}-t512-{state}")
            for tokens in (SHORT_THRESHOLD - 1, SHORT_THRESHOLD, SHORT_THRESHOLD + 1):
                cases.append(
                    dict(
                        source,
                        name=f"{dtype}-t{tokens}-{state}",
                        steps=[tuple(x[:, :tokens].clone() for x in source["steps"][0])],
                    )
                )

    return cases


def dispatch_cases(cases):
    """Extend frozen original fixtures by slicing; consume no additional RNG."""
    extra = []
    for dtype in ("bf16", "fp32"):
        source = next(c for c in cases if c["name"] == f"{dtype}-whole")
        lengths = (
            SHORT_THRESHOLD - 1,
            SHORT_THRESHOLD + 1,
            SHORT_THRESHOLD,
            SHORT_THRESHOLD + 2,
            1,
        )
        lengths += (512 - sum(lengths),)
        assert min(lengths) > 0 and sum(lengths) == 512
        data = [x.split(lengths, dim=1) for x in source["steps"][0]]
        alternate = dict(source, name=f"{dtype}-cond-alternate", steps=list(zip(*data)))
        extra.extend((alternate, dict(alternate, name=f"{dtype}-cond-alternate-repeat")))
        # Original carry already covers long prefill followed by 64 decode calls.
        carry = next(c for c in cases if c["name"] == f"{dtype}-carry")
        extra.append(
            dict(
                carry,
                name=f"{dtype}-cond-short-carry",
                steps=[
                    tuple(x[:, : SHORT_THRESHOLD - 1].clone() for x in carry["steps"][0]),
                    *carry["steps"][1:],
                ],
            )
        )
    return cases + extra


def inputs_for(case):
    """Canonical contiguous input strides, including singleton dimensions."""
    return tuple(
        x.to(device="cuda", copy=True, memory_format=torch.contiguous_format)
        for x in (*case["steps"][0], *(case[key] for key in STATE_FIELDS))
    )


def selected(*inputs):
    """Explicit branch control, independent of torch.cond lowering."""
    branch = short if inputs[0].shape[1] <= SHORT_THRESHOLD else long
    return branch(*(x.contiguous() for x in inputs))


def run_cases(call, cases, fp32=False):
    results = {}
    for case in cases:
        # .contiguous() leaves unused B=1 strides unchanged. Force the exact
        # packed strides expected by AOTI guards.
        convert = lambda x: x.to(
            device="cuda",
            dtype=torch.float32 if fp32 else x.dtype,
            copy=True,
            memory_format=torch.contiguous_format,
        )
        conv, state, weight, A_log, dt_bias = (
            convert(case[k]) for k in ("conv", "state", "weight", "A_log", "dt_bias")
        )
        for step, data in enumerate(case["steps"]):
            inputs = (*map(convert, data), conv, state, weight, A_log, dt_bias)
            before = [x.clone() for x in inputs]
            values = call(*inputs)
            assert len(values) == 3 and values[0].shape == (
                1,
                inputs[0].shape[1],
                state.shape[1],
                state.shape[3],
            )
            assert values[0].dtype == inputs[0].dtype and values[1].dtype == conv.dtype
            assert values[1].shape == conv.shape and values[2].shape == state.shape
            assert values[2].dtype == torch.float32
            assert all(bits_equal(a, b) for a, b in zip(inputs, before)), "Input mutation"
            assert not any(torch._C._is_alias_of(x, y) for x in values for y in inputs), (
                "Output aliases input"
            )
            results[case["name"], step] = cpu_copy(values)
            conv, state = values[1:]
    return results


def compare(report, label, actual, expected, exact_tensors=()):
    for key, values in actual.items():
        dtype = key[0].split("-")[0]
        for i, (a, b) in enumerate(zip(values, expected[key])):
            width = (128, 4, 128 * 128)[i]
            error = row_errors(a.float().reshape(-1, width), b.float().reshape(-1, width))
            row = dict(
                comparison=label,
                case=key[0],
                step=key[1],
                tensor=("core", "conv", "recurrent")[i],
                bitwise=bits_equal(a, b),
                **error,
            )
            report["comparisons"].append(row)
            failed = (
                max(error["relative_l2"], error["peak_relative"]) > LIMITS[dtype]
                and error["max_abs"] > ABS_LIMITS[dtype]
            )
            if (
                failed
                or (i in exact_tensors and not row["bitwise"])
                or (i == 1 and not torch.equal(a.float(), b.float()))
            ):
                report["failures"].append(row)
