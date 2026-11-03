"""CPU checks for common measurements and the models' distinct acceptance policies."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from compiler.common.artifacts import ROOT
from compiler.common.tests.validation import (
    bits_equal,
    compare_file,
    compare_tensors,
    record_failure,
    tensor_metrics,
    token_differences,
)
from compiler.qwen3.tests.numerics import (
    model_passed as qwen3_passed,
    reference_limits_passed,
)
from compiler.qwen35.tests.common import (
    compare as compare_states,
    end_to_end_passed,
    metrics as state_metrics,
    model_passed,
)


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
        metrics = tensor_metrics(actual, reference, logits=True)
        self.assertEqual(metrics["rows"], 2)
        self.assertEqual(metrics["top1_agreement"], 1)
        (difference,) = token_differences(actual, reference)
        self.assertEqual(
            (difference["row"], difference["reference_id"], difference["actual_id"]), (0, 0, 1)
        )
        self.assertGreater(difference["reference_selected_token_gap"], 0)
        self.assertGreater(difference["actual_top2_margin"], 0)

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
from compiler.common.tests.validation import (
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
        error = state_metrics(actual, reference, 2)
        self.assertAlmostEqual(error["relative_l2"], 2**0.5 * 0.2, places=6)
        self.assertFalse(error["bitwise_equal"])


class ModelAcceptancePolicies(unittest.TestCase):
    def qwen3_report(self):
        return dict(
            failures=[], hf_cache_differences=[], hf_token_differences=[], hf_greedy_differences=[]
        )

    def qwen35_report(self):
        return dict(comparisons=[], failures=[], top1_differences=[], greedy_failures=[])

    def test_same_state_error_is_diagnostic_for_qwen3_and_required_for_qwen35(self):
        reference = torch.ones(1, 2)
        actual = reference * 1.06
        q3 = self.qwen3_report()
        record_failure(
            q3["hf_cache_differences"], tensor_metrics(actual, reference), tensor="cache"
        )
        self.assertTrue(qwen3_passed(q3))
        self.assertFalse(reference_limits_passed(q3))
        q35 = self.qwen35_report()
        compare_states(
            q35,
            "aoti_vs_hf",
            {("case", 0): (reference, actual)},
            {("case", 0): (reference, reference)},
            [{"name": "conv", "kind": "conv"}],
        )
        self.assertFalse(model_passed(q35))
        self.assertFalse(end_to_end_passed(q35))

    def test_top1_drift_preserves_stage_specific_acceptance(self):
        reference = torch.tensor([[1.0, 0.99]])
        actual = torch.tensor([[0.99, 1.0]])
        q3 = self.qwen3_report()
        q3["hf_token_differences"] = token_differences(actual, reference)
        self.assertTrue(qwen3_passed(q3))
        self.assertFalse(reference_limits_passed(q3))
        q35 = self.qwen35_report()
        compare_states(
            q35, "aoti_vs_hf_full_prompt", {("case", 0): (actual,)}, {("case", 0): (reference,)}, []
        )
        self.assertTrue(model_passed(q35))
        self.assertFalse(end_to_end_passed(q35))
        diagnostic = self.qwen35_report()
        compare_states(
            diagnostic,
            "aoti_vs_fp32",
            {("case", 0): (actual,)},
            {("case", 0): (reference,)},
            [],
            enforce=False,
        )
        self.assertTrue(end_to_end_passed(diagnostic))

    def test_logit_failure_and_greedy_policy_remain_required(self):
        q3 = self.qwen3_report()
        q3["failures"].append({"tensor": "logits"})
        self.assertFalse(qwen3_passed(q3))
        q35 = self.qwen35_report()
        q35["greedy_failures"].append({"case": "chat"})
        self.assertTrue(model_passed(q35))
        self.assertFalse(end_to_end_passed(q35))


if __name__ == "__main__":
    unittest.main()
