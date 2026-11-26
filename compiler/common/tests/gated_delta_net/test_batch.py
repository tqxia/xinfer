"""Independent-row DeltaNet numerics, padding and in-place carried-state contracts."""

import unittest

import torch

from compiler.common.tests.tensors import bits_equal, row_errors
from compiler.common.kernels.gated_delta_net.deltanet import (
    batched_deltanet,
    load_deltanet_ops,
)
from compiler.common.tests.gated_delta_net.fixtures import (
    ABS_LIMITS,
    BOUNDARY_LENGTH,
    LIMITS,
    reference,
)


def batch_inputs(batch, tokens, dtype, *, value_heads=16, counts=None, nonzero=True):
    """Distinct deterministic rows, including inactive and partial-chunk prefixes."""
    rng = torch.Generator().manual_seed(3500 + batch + tokens)
    channels = (32 + value_heads) * 128

    def random(shape, scale=1, kind=dtype):
        return (torch.randn(shape, generator=rng) * scale).to(device="cuda", dtype=kind)

    if counts is None:
        counts = [tokens, max(0, tokens - 1), 0, 1, 63, 64, 65, 97][:batch]
        counts = [min(count, tokens) for count in counts]
    conv = random((batch, channels, 4), 0.3)
    state = random((batch, value_heads, 128, 128), 0.1, torch.float32)
    if not nonzero:
        conv.zero_()
        state.zero_()
    return (
        random((batch, tokens, channels)),
        random((batch, tokens, value_heads)),
        random((batch, tokens, value_heads)),
        conv,
        state,
        random((channels, 4), 0.2),
        torch.full((value_heads,), -2.0, device="cuda", dtype=dtype),
        random((value_heads,), 0.1),
        torch.tensor(counts, dtype=torch.int32),
    )


def row_inputs(inputs, row, tokens=None):
    values = [x[row : row + 1] for x in inputs[:5]]
    if tokens is not None:
        values[:3] = [x[:, :tokens] for x in values[:3]]
    # AOTI guards also check strides on size-one dimensions.
    return tuple(x.clone(memory_format=torch.contiguous_format) for x in values) + tuple(
        inputs[5:8]
    )


def call_inplace(inputs):
    """Invoke batched_deltanet on cloned state buffers and return (core, conv, state)."""
    qkv, b, a, conv, state, weight, A_log, dt_bias, query_lengths = inputs
    conv_copy, state_copy = conv.clone(), state.clone()
    core = batched_deltanet(
        qkv, b, a, conv_copy, state_copy, weight, A_log, dt_bias, query_lengths=query_lengths
    )
    return core, conv_copy, state_copy


def independent_reference(inputs):
    """Run each valid prefix through HF at the original input dtype."""
    outputs, conv, state = [], [], []
    for row, count in enumerate(inputs[-1].tolist()):
        values = row_inputs(inputs, row, count)
        output = values[0].new_zeros((1, inputs[0].shape[1], inputs[4].shape[1], 128))
        if count:
            core, next_conv, final = reference(
                *values, chunked=inputs[0].shape[1] > BOUNDARY_LENGTH
            )
            output[:, :count] = core
        else:
            next_conv, final = values[3:5]
        outputs.append(output)
        conv.append(next_conv)
        state.append(final)
    return tuple(torch.cat(values) for values in (outputs, conv, state))


def assert_numerics(test, actual, expected, dtype):
    name = "bf16" if dtype == torch.bfloat16 else "fp32"
    for field, a, b, width in zip(("core", "conv", "recurrent"), actual, expected, (128, 4, 16384)):
        with test.subTest(field=field):
            errors = row_errors(a.float().reshape(-1, width), b.float().reshape(-1, width))
            test.assertTrue(
                max(errors["relative_l2"], errors["peak_relative"]) <= LIMITS[name]
                or errors["max_abs"] <= ABS_LIMITS[name],
                errors,
            )
            if field == "conv":
                test.assertTrue(torch.equal(a.float(), b.float()))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class BatchedDeltaNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        load_deltanet_ops()

    @torch.inference_mode()
    def test_same_dtype_hf_reference(self):
        for dtype in (torch.bfloat16, torch.float32):
            for batch in (1, 2, 4, 8):
                for tokens in (1, 97, 98, 512):
                    with self.subTest(dtype=dtype, batch=batch, tokens=tokens):
                        inputs = batch_inputs(batch, tokens, dtype, nonzero=tokens != 98)
                        actual = call_inplace(inputs)
                        assert_numerics(self, actual, independent_reference(inputs), dtype)

    @torch.inference_mode()
    def test_every_batch_size_and_component_boundaries(self):
        for batch in range(1, 9):
            for tokens in (1, 2, 3, 4, 63, 64, 65, 97, 98, 127, 128, 129, 511, 512):
                with self.subTest(batch=batch, tokens=tokens):
                    inputs = batch_inputs(batch, tokens, torch.bfloat16)
                    before = tuple(x.clone() for x in inputs)
                    conv_ptr, state_ptr = inputs[3].data_ptr(), inputs[4].data_ptr()
                    core = batched_deltanet(*inputs[:8], query_lengths=inputs[8])
                    actual = (core, inputs[3], inputs[4])
                    self.assertEqual(inputs[3].data_ptr(), conv_ptr)
                    self.assertEqual(inputs[4].data_ptr(), state_ptr)
                    for index in (0, 1, 2, 5, 6, 7, 8):
                        self.assertTrue(bits_equal(inputs[index], before[index]))
                    self.assertFalse(any(torch._C._is_alias_of(core, b) for b in inputs))
                    for row, count in enumerate(inputs[-1].tolist()):
                        single = row_inputs(before, row)
                        single_core = batched_deltanet(
                            *single, query_lengths=inputs[8][row : row + 1]
                        )
                        expected = (single_core, single[3], single[4])
                        self.assertTrue(
                            all(bits_equal(a[row : row + 1], b) for a, b in zip(actual, expected))
                        )
                        self.assertTrue(torch.count_nonzero(actual[0][row, count:]) == 0)
                        if not count:
                            self.assertTrue(bits_equal(actual[1][row], before[3][row]))
                            self.assertTrue(bits_equal(actual[2][row], before[4][row]))

    @torch.inference_mode()
    def test_padding_row_isolation_and_grouped_heads(self):
        for heads in (16, 32, 48):
            for tokens in (1, 97, 98, 512):
                with self.subTest(heads=heads, tokens=tokens):
                    inputs = batch_inputs(
                        4, tokens, torch.bfloat16, value_heads=heads, counts=[0, tokens, 1, 0]
                    )
                    expected = call_inplace(inputs)
                    poisoned = tuple(x.clone() for x in inputs)
                    for row, count in enumerate(inputs[-1].tolist()):
                        for tensor in poisoned[:3]:
                            tensor[row, count:] = float("nan")
                    actual = call_inplace(poisoned)
                    self.assertTrue(all(bits_equal(a, b) for a, b in zip(actual, expected)))
                    poisoned[0][1, :tokens].add_(1)
                    changed = call_inplace(poisoned)
                    for row in (0, 2, 3):
                        self.assertTrue(
                            all(bits_equal(a[row], b[row]) for a, b in zip(changed, expected))
                        )
                    assert_numerics(self, expected, independent_reference(inputs), torch.bfloat16)

    @torch.inference_mode()
    def test_carried_state_across_steps_and_inactive_rows(self):
        conv = state = None
        conv_ptr = state_ptr = None
        for tokens, counts in (
            (98, [98, 3, 64, 0]),
            (1, [1, 0, 1, 1]),
            (97, [0, 97, 2, 1]),
            (129, [65, 0, 129, 0]),
        ):
            inputs = list(batch_inputs(4, tokens, torch.bfloat16, counts=counts))
            if conv is not None:
                inputs[3], inputs[4] = conv, state
            else:
                conv, state = inputs[3], inputs[4]
                conv_ptr, state_ptr = conv.data_ptr(), state.data_ptr()
            before = tuple(x.clone() for x in inputs)
            with self.subTest(tokens=tokens):
                core = batched_deltanet(*inputs[:8], query_lengths=inputs[8])
                self.assertEqual(conv.data_ptr(), conv_ptr)
                self.assertEqual(state.data_ptr(), state_ptr)
                assert_numerics(
                    self, (core, conv, state), independent_reference(before), torch.bfloat16
                )

    @torch.inference_mode()
    def test_fp32_decay_parameters_with_bf16_inputs(self):
        for tokens in (1, 97, 98):
            with self.subTest(tokens=tokens):
                inputs = list(batch_inputs(4, tokens, torch.bfloat16))
                inputs[6] = torch.linspace(-2.5, 0.5, 16, device="cuda", dtype=torch.float32)
                inputs[7] = torch.linspace(-0.2, 0.2, 16, device="cuda", dtype=torch.float32)
                actual = call_inplace(inputs)
                assert_numerics(self, actual, independent_reference(inputs), torch.bfloat16)

    @torch.inference_mode()
    def test_distinct_value_heads_use_grouped_qk(self):
        torch.manual_seed(67)
        for value_heads in (32, 48):
            channels = (32 + value_heads) * 128
            for tokens in (1, 98):
                with self.subTest(value_heads=value_heads, tokens=tokens):
                    qkv = torch.randn(1, tokens, channels, device="cuda")
                    # Distinct gates and states prevent the repeated key heads from
                    # concealing an incorrect value-head mapping or shared state.
                    b = (
                        torch.linspace(-2, 2, value_heads, device="cuda")
                        .expand(1, tokens, -1)
                        .clone()
                    )
                    a = (
                        torch.linspace(-1, 1, value_heads, device="cuda")
                        .expand(1, tokens, -1)
                        .clone()
                    )
                    conv = torch.randn(1, channels, 4, device="cuda") * 0.1
                    state = torch.randn(1, value_heads, 128, 128, device="cuda") * 0.1
                    weight = torch.randn(channels, 4, device="cuda") * 0.2
                    A_log = torch.linspace(-3, -1, value_heads, device="cuda")
                    dt_bias = torch.linspace(-0.2, 0.2, value_heads, device="cuda")
                    sample = (qkv, b, a, conv, state, weight, A_log, dt_bias)
                    actual_conv, actual_state = conv.clone(), state.clone()
                    conv_ptr, state_ptr = actual_conv.data_ptr(), actual_state.data_ptr()
                    core = batched_deltanet(
                        qkv, b, a, actual_conv, actual_state, weight, A_log, dt_bias
                    )
                    self.assertEqual(actual_conv.data_ptr(), conv_ptr)
                    self.assertEqual(actual_state.data_ptr(), state_ptr)
                    expected = reference(*sample, chunked=tokens > 1)
                    for actual_tensor, expected_tensor, width in (
                        (core, expected[0], 128),
                        (actual_state, expected[2], 128 * 128),
                    ):
                        error = row_errors(
                            actual_tensor.float().reshape(-1, width),
                            expected_tensor.float().reshape(-1, width),
                        )
                        self.assertLessEqual(error["relative_l2"], 0.0001)
                        self.assertLessEqual(error["peak_relative"], 0.0001)
                    self.assertTrue(bits_equal(actual_conv, expected[1]))


if __name__ == "__main__":
    unittest.main()
