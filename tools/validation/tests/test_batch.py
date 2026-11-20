"""Shared batch validation preserves reference callbacks and comparison coverage."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch, sentinel

import torch

from compiler.common.artifacts import read_json
from tools.validation.common import batch


class BatchValidation(unittest.TestCase):
    def test_reference_is_reused_across_cases(self):
        logits = [torch.ones(1, 4, dtype=torch.bfloat16)]
        state = [torch.ones(1, 1, 1, 2, dtype=torch.bfloat16)]
        reference = Mock(return_value=(logits, state))
        cases = [
            dict(name=name, batch_steps=[[[1]]], state_steps=[0]) for name in ("prompts", "reset")
        ]
        runtime = SimpleNamespace(position=1, positions=[1] + [0] * 7)
        meta = dict(state=[])
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "expected").mkdir()
            args = SimpleNamespace(output=output)
            with (
                patch.object(batch, "collect_explicit", return_value=(logits, state)),
                patch.object(batch, "capture_state", return_value=state),
            ):
                batch.validate_python_group(
                    args,
                    "b1",
                    ["prompt"],
                    cases,
                    meta,
                    sentinel.hf,
                    sentinel.eager,
                    runtime,
                    collect_reference=reference,
                )
            self.assertEqual(reference.call_count, 2)
            self.assertTrue(all(call.args[0] is sentinel.hf for call in reference.call_args_list))
            report = read_json(output / "b1/report.json")
            self.assertEqual(len(report["cases"]), 2)
            self.assertEqual(report["structural_checks"], ["reset"])
            self.assertEqual(report["failures"], [])

    def test_comparison_rejects_missing_entries(self):
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            batch.compare_stage({}, "fixture", "eager/hf", [], [torch.zeros(1)], logits=True)


if __name__ == "__main__":
    unittest.main()
