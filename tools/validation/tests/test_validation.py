"""CPU checks for shared numerical measurements and exact file comparisons."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from compiler.common.artifacts import ROOT
from tools.validation.common.validation import (
    bits_equal,
    compare_file,
    compare_tensors,
    record_failure,
    tensor_metrics,
    token_differences,
)
from tools.validation.common.chain import Policy


class ValidationMeasurements(unittest.TestCase):
    def test_rows_are_bounded_individually(self):
        reference = torch.tensor([[1.0, 0.0], [100.0, 0.0]])
        actual = reference.clone()
        actual[0, 1] = 0.1
        metrics = tensor_metrics(actual, reference)
        self.assertAlmostEqual(metrics["relative_l2"], 0.1, places=6)
        self.assertAlmostEqual(metrics["peak_relative"], 0.1, places=6)
        with self.assertRaises(ValueError):
            compare_tensors(actual, reference)

    def test_bitwise_contract_keeps_shape_dtype_and_signed_zero(self):
        positive = torch.tensor([[0.0, 1.0]])
        negative = torch.tensor([[-0.0, 1.0]])
        self.assertFalse(bits_equal(positive, negative))
        self.assertFalse(bits_equal(positive, positive.reshape(2, 1)))
        self.assertFalse(bits_equal(positive, positive.double()))
        compare_tensors(positive, negative)
        with self.assertRaises(ValueError):
            compare_tensors(positive, negative, exact=True)

    def test_canonical_logits_and_token_diagnostics(self):
        reference = torch.tensor([[1.0, 0.99], [3.0, 1.0]])
        actual = torch.tensor([[0.99, 1.0], [3.0, 1.0]])
        compare_tensors(actual, reference)
        metrics = tensor_metrics(actual, reference, logits=True)
        self.assertEqual(metrics["rows"], 2)
        self.assertEqual(metrics["top1_agreement"], 1)
        (difference,) = token_differences(actual, reference)
        self.assertEqual(
            (difference["row"], difference["reference_id"], difference["actual_id"]), (0, 0, 1)
        )
        self.assertGreater(difference["reference_selected_token_gap"], 0)
        self.assertGreater(difference["actual_top2_margin"], 0)

    def test_large_logit_drift_fails_with_matching_top1(self):
        reference = torch.tensor([[1.0, 0.5]])
        with self.assertRaises(ValueError):
            compare_tensors(reference * 1.06, reference)

    def test_peak_relative_bound_is_enforced_independently(self):
        reference = torch.ones(1, 128)
        actual = reference.clone()
        actual[0, 0] += 0.06
        self.assertLess(tensor_metrics(actual, reference)["relative_l2"], 0.05)
        with self.assertRaises(ValueError):
            compare_tensors(actual, reference)

    def test_bf16_token_margins_use_float32_diagnostics(self):
        reference = torch.tensor([[1.0, 0.001]], dtype=torch.bfloat16)
        actual = reference.flip(-1)
        (difference,) = token_differences(actual, reference)
        expected_gap = (reference.float()[0, 0] - reference.float()[0, 1]).item()
        self.assertEqual(difference["reference_top2_margin"], expected_gap)
        self.assertEqual(difference["reference_selected_token_gap"], expected_gap)
        self.assertEqual(difference["actual_top2_margin"], expected_gap)
        self.assertNotEqual(expected_gap, (reference[0, 0] - reference[0, 1]).item())

    def test_invalid_measurements_fail_before_acceptance(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                tensor_metrics(torch.tensor([[value, 1.0]]), torch.ones(1, 2))
        with self.assertRaises(ValueError):
            tensor_metrics(torch.ones(2, 1), torch.ones(1, 2))

    def test_native_file_requires_exact_bytes_and_length(self):
        tensors = (torch.tensor([0.0, 1.0]), torch.tensor([2.0], dtype=torch.bfloat16))
        data = b"".join(t.view(torch.uint8).numpy().tobytes() for t in tensors)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "native.bin"
            path.write_bytes(data)
            compare_file(path, iter(tensors))
            for changed in (bytes([data[0] ^ 1]) + data[1:], data[:-1], data + b"x"):
                with self.subTest(data=changed), self.assertRaises(ValueError):
                    path.write_bytes(changed)
                    compare_file(path, iter(tensors))

    def test_shared_checks_remain_active_in_optimized_python(self):
        # Run a separate CPU-only suite because model-test imports deliberately
        # reject optimized interpreters before their assertion-based checks run.
        code = r"""
import tempfile
import unittest
from pathlib import Path
import torch
from tools.validation.common.validation import (
    compare_file, compare_tensors, cpu_copy, tensor_metrics, token_differences,
)

class OptimizedValidation(unittest.TestCase):
    def test_tensor_acceptance(self):
        reference = torch.tensor([[0.0, 1.0]])
        compare_tensors(reference, reference.clone(), exact=True)
        compare_tensors(reference * 1.01, reference)
        with self.assertRaises(ValueError):
            compare_tensors(torch.tensor([[-0.0, 1.0]]), reference, exact=True)
        with self.assertRaises(ValueError):
            compare_tensors(reference * 1.06, reference)
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            tensor_metrics(torch.ones(2, 1), torch.ones(1, 2))
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                tensor_metrics(torch.tensor([[invalid, 1.0]]), reference)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                cpu_copy((torch.tensor([invalid]),))
        with self.assertRaisesRegex(ValueError, "at least two logits"):
            token_differences(torch.ones(1, 1), torch.ones(1, 1))

    def test_file_acceptance(self):
        tensor = torch.tensor([0.0, 1.0])
        data = tensor.view(torch.uint8).numpy().tobytes()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "native.bin"
            path.write_bytes(data)
            compare_file(path, (tensor,))
            for changed in (bytes([data[0] ^ 1]) + data[1:], data[:-1], data + b"x"):
                with self.subTest(data=changed), self.assertRaises(ValueError):
                    path.write_bytes(changed)
                    compare_file(path, (tensor,))

unittest.main()
"""
        env = {key: value for key, value in os.environ.items() if key != "PYTHONOPTIMIZE"}
        for flag in ("-O", "-OO"):
            with self.subTest(flag=flag):
                result = subprocess.run(
                    [sys.executable, flag, "-c", code],
                    cwd=ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_required_and_diagnostic_comparisons_share_measurements(self):
        metrics = tensor_metrics(torch.ones(1, 2) * 1.06, torch.ones(1, 2))
        failures = []
        self.assertFalse(record_failure(failures, metrics, required=False, tensor="state"))
        self.assertEqual(failures, [])
        self.assertTrue(record_failure(failures, metrics, tensor="state"))
        self.assertEqual(failures, [dict(tensor="state", **metrics)])
        # Kernel tolerances are explicit and can differ from the model default.
        self.assertFalse(record_failure([], metrics, limit=0.1))

    def test_state_grouping_preserves_recurrent_heads(self):
        reference = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]], [[100.0, 0.0], [0.0, 0.0]]]])
        actual = reference.clone()
        actual[0, 0, 1, :] = 0.2
        policy = Policy()
        rows = tuple(policy.rows(t, {"kind": "recurrent"}) for t in (actual, reference))
        error = tensor_metrics(actual, reference, rows=rows)
        self.assertAlmostEqual(error["relative_l2"], 2**0.5 * 0.2, places=6)
        self.assertFalse(error["bitwise_equal"])


if __name__ == "__main__":
    unittest.main()
