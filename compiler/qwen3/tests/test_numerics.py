"""CPU regression checks for the Qwen3 numerical acceptance contract."""

import unittest

import torch

from compiler.qwen3.tests.numerics import compare, error_metrics, token_differences


class NumericalPolicyTest(unittest.TestCase):
    def test_top1_drift_with_bounded_logits_is_diagnostic(self):
        hf = torch.tensor([[1.0, 0.99]])
        compiled = torch.tensor([[0.99, 1.0]])
        compare(compiled, hf)
        difference, = token_differences(compiled, hf)
        self.assertEqual((difference['hf_id'], difference['compiled_id']), (0, 1))
        self.assertGreater(difference['hf_selected_token_gap'], 0)

    def test_large_logit_drift_still_fails_with_matching_top1(self):
        hf = torch.tensor([[1.0, 0.5]])
        with self.assertRaises(AssertionError):
            compare(hf * 1.06, hf)

    def test_peak_relative_bound_is_enforced_independently(self):
        hf = torch.ones(1, 128)
        compiled = hf.clone()
        compiled[0, 0] += 0.06
        self.assertLess(error_metrics(compiled, hf)['max_relative_l2'], 0.05)
        with self.assertRaises(AssertionError):
            compare(compiled, hf)

    def test_nonfinite_values_cannot_be_reported_as_accepted_drift(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value), self.assertRaises(AssertionError):
                error_metrics(torch.tensor([[value, 1.0]]), torch.ones(1, 2))

    def test_native_parity_retains_bitwise_signed_zero_check(self):
        with self.assertRaises(AssertionError):
            compare(torch.tensor([[0.0, 1.0]]), torch.tensor([[-0.0, 1.0]]), exact=True)


if __name__ == '__main__':
    unittest.main()
