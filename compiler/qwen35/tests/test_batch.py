"""Independent-row DeltaNet numerics, padding and carried-state contracts."""

import unittest

import torch

from compiler.common.tests.tensors import bits_equal, row_errors
from compiler.qwen35.config import configure
from compiler.qwen35.kernels.recurrent import SHORT_THRESHOLD, fused, long, short
from compiler.qwen35.tests.delta import ABS_LIMITS, LIMITS, reference


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
        torch.tensor(counts, device="cuda", dtype=torch.int32),
    )


def row_inputs(inputs, row, tokens=None):
    values = [x[row : row + 1] for x in inputs[:5]]
    if tokens is not None:
        values[:3] = [x[:, :tokens] for x in values[:3]]
    # AOTI guards also check strides on size-one dimensions.
    return tuple(x.clone(memory_format=torch.contiguous_format) for x in values) + tuple(
        inputs[5:8]
    )


def independent_reference(inputs):
    """Run each valid prefix through HF at the original input dtype."""
    outputs, conv, state = [], [], []
    for row, count in enumerate(inputs[-1].tolist()):
        values = row_inputs(inputs, row, count)
        output = values[0].new_zeros((1, inputs[0].shape[1], inputs[4].shape[1], 128))
        if count:
            core, next_conv, final = reference(
                *values, chunked=inputs[0].shape[1] > SHORT_THRESHOLD
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
        configure()

    @torch.inference_mode()
    def test_same_dtype_hf_reference(self):
        for dtype in (torch.bfloat16, torch.float32):
            for batch in (1, 2, 4, 8):
                for tokens in (1, 97, 98, 512):
                    with self.subTest(dtype=dtype, batch=batch, tokens=tokens):
                        inputs = batch_inputs(batch, tokens, dtype, nonzero=tokens != 98)
                        actual = fused(*inputs)
                        assert_numerics(self, actual, independent_reference(inputs), dtype)

    @torch.inference_mode()
    def test_every_batch_size_and_component_boundaries(self):
        for batch in range(1, 9):
            for tokens in (1, 2, 3, 4, 63, 64, 65, 97, 98, 127, 128, 129, 511, 512):
                with self.subTest(batch=batch, tokens=tokens):
                    inputs = batch_inputs(batch, tokens, torch.bfloat16)
                    before = tuple(x.clone() for x in inputs)
                    actual = fused(*inputs)
                    self.assertTrue(all(bits_equal(a, b) for a, b in zip(inputs, before)))
                    self.assertFalse(
                        any(torch._C._is_alias_of(a, b) for a in actual for b in inputs)
                    )
                    branch = short if tokens <= SHORT_THRESHOLD else long
                    for row, count in enumerate(inputs[-1].tolist()):
                        expected = branch(*row_inputs(inputs, row), inputs[-1][row : row + 1])
                        self.assertTrue(
                            all(bits_equal(a[row : row + 1], b) for a, b in zip(actual, expected))
                        )
                        self.assertTrue(torch.count_nonzero(actual[0][row, count:]) == 0)
                        if not count:
                            self.assertTrue(bits_equal(actual[1][row], inputs[3][row]))
                            self.assertTrue(bits_equal(actual[2][row], inputs[4][row]))

    @torch.inference_mode()
    def test_padding_row_isolation_and_grouped_heads(self):
        for heads in (16, 32):
            for tokens in (1, 97, 98, 512):
                with self.subTest(heads=heads, tokens=tokens):
                    inputs = batch_inputs(
                        4, tokens, torch.bfloat16, value_heads=heads, counts=[0, tokens, 1, 0]
                    )
                    expected = fused(*inputs)
                    poisoned = tuple(x.clone() for x in inputs)
                    for row, count in enumerate(inputs[-1].tolist()):
                        for tensor in poisoned[:3]:
                            tensor[row, count:] = float("nan")
                    actual = fused(*poisoned)
                    self.assertTrue(all(bits_equal(a, b) for a, b in zip(actual, expected)))
                    poisoned[0][1, :tokens].add_(1)
                    changed = fused(*poisoned)
                    for row in (0, 2, 3):
                        self.assertTrue(
                            all(bits_equal(a[row], b[row]) for a, b in zip(changed, expected))
                        )
                    assert_numerics(self, expected, independent_reference(inputs), torch.bfloat16)

    @torch.inference_mode()
    def test_carried_state_across_dispatch_and_inactive_rows(self):
        states = None
        for tokens, counts in (
            (98, [98, 3, 64, 0]),
            (1, [1, 0, 1, 1]),
            (97, [0, 97, 2, 1]),
            (129, [65, 0, 129, 0]),
        ):
            inputs = list(batch_inputs(4, tokens, torch.bfloat16, counts=counts))
            if states is not None:
                inputs[3:5] = states
            with self.subTest(tokens=tokens):
                actual = fused(*inputs)
                assert_numerics(self, actual, independent_reference(inputs), torch.bfloat16)
                states = actual[1:]

    @torch.inference_mode()
    def test_fp32_decay_parameters_with_bf16_inputs(self):
        for tokens in (1, 97, 98):
            with self.subTest(tokens=tokens):
                inputs = list(batch_inputs(4, tokens, torch.bfloat16))
                inputs[6] = torch.linspace(-2.5, 0.5, 16, device="cuda", dtype=torch.float32)
                inputs[7] = torch.linspace(-0.2, 0.2, 16, device="cuda", dtype=torch.float32)
                actual = fused(*inputs)
                assert_numerics(self, actual, independent_reference(inputs), torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
