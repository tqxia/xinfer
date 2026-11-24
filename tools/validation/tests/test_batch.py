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
    def test_stages_run_sequentially_reuse_models_and_preserve_dotted_case_names(self):
        logits = [torch.ones(1, 4, dtype=torch.bfloat16)]
        state = [torch.ones(1, 1, 1, 2, dtype=torch.bfloat16)]
        events = []

        class StageHandle:
            def __init__(self, label):
                self.label = label
                self.position = 1
                self.positions = [1] + [0] * 7

            def close(self):
                events.append(f"close:{self.label}")

        def make_loader(label):
            def loader():
                events.append(f"load:{label}")
                return StageHandle(label)

            return loader

        def collect_ref(ref, _steps):
            events.append(f"run:{ref.label}")
            return logits, state

        def collect_exp(runtime, _case):
            events.append(f"run:{runtime.label}")
            return logits, state

        groups = {
            "b1": (
                ["prompt"],
                [
                    dict(name=name, batch_steps=[[[1]]], state_steps=[0])
                    for name in ("prompts", "reset", "case_1.5")
                ],
            )
        }
        meta = dict(state=[])
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "expected").mkdir()
            args = SimpleNamespace(output=output)
            with (
                patch.object(batch, "collect_explicit", side_effect=collect_exp),
                patch.object(batch, "capture_state", return_value=state),
            ):
                result = batch.validate_python_stages(
                    args,
                    groups,
                    meta,
                    load_reference=make_loader("hf"),
                    collect_reference=collect_ref,
                    load_eager=make_loader("eager"),
                    load_aoti=make_loader("aoti"),
                )
            self.assertEqual(result, ["b1"])
            self.assertEqual(
                events,
                [
                    "load:hf",
                    "run:hf",
                    "run:hf",
                    "run:hf",
                    "close:hf",
                    "load:eager",
                    "run:eager",
                    "run:eager",
                    "run:eager",
                    "close:eager",
                    "load:aoti",
                    "run:aoti",
                    "run:aoti",
                    "run:aoti",
                    "close:aoti",
                ],
            )
            self.assertFalse((output / "_staging").exists())
            self.assertTrue((output / "expected/b1_case_1.5.0.states").is_file())
            self.assertTrue((output / "expected/b1_case_1.5.f32").is_file())
            report = read_json(output / "b1/report.json")
            self.assertEqual(len(report["cases"]), 3)
            self.assertEqual(report["structural_checks"], ["reset"])
            self.assertEqual(report["failures"], [])

    def test_stage_failure_closes_loaded_runtime_and_cleans_staging(self):
        logits = [torch.ones(1, 4, dtype=torch.bfloat16)]
        state = [torch.ones(1, 1, 1, 2, dtype=torch.bfloat16)]
        groups = {
            "b1": (
                ["prompt"],
                [dict(name="prompts", batch_steps=[[[1]]], state_steps=[0])],
            )
        }
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "expected").mkdir()
            args = SimpleNamespace(output=output)
            eager = Mock()
            aoti = Mock()
            with (
                patch.object(batch, "collect_explicit", side_effect=RuntimeError("eager boom")),
                self.assertRaisesRegex(RuntimeError, "eager boom"),
            ):
                batch.validate_python_stages(
                    args,
                    groups,
                    dict(state=[]),
                    load_reference=lambda: sentinel.hf,
                    collect_reference=lambda _ref, _steps: (logits, state),
                    load_eager=lambda: eager,
                    load_aoti=lambda: aoti,
                )
            eager.close.assert_called_once_with()
            aoti.close.assert_not_called()
            self.assertFalse((output / "_staging").exists())

    def test_comparison_rejects_missing_entries(self):
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            batch.compare_stage({}, "fixture", "eager/hf", [], [torch.zeros(1)], logits=True)


if __name__ == "__main__":
    unittest.main()
