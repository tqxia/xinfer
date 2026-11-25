"""Shared batch validation preserves reference callbacks and comparison coverage."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch, sentinel

import torch

import weakref

from compiler.common.artifacts import read_json, write_json
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

    def test_stage_reference_is_released_before_next_stage_loads(self):
        logits = [torch.ones(1, 4, dtype=torch.bfloat16)]
        state = [torch.ones(1, 1, 1, 2, dtype=torch.bfloat16)]
        groups = {
            "b1": (
                ["prompt"],
                [dict(name="prompts", batch_steps=[[[1]]], state_steps=[0])],
            )
        }
        refs = {}

        class Tracked:
            def __init__(self, label):
                self.label = label
                self.position = 1
                self.positions = [1] + [0] * 7
                # Reference cycle requires gc.collect() AFTER dropping caller binding.
                self.cycle = self

            def close(self):
                pass

        def load_ref():
            obj = Tracked("hf")
            refs["hf"] = weakref.ref(obj)
            return obj

        def load_eager():
            self.assertIsNone(refs["hf"](), "HF model still alive when loading eager")
            obj = Tracked("eager")
            refs["eager"] = weakref.ref(obj)
            return obj

        def load_aoti():
            self.assertIsNone(refs["eager"](), "Eager model still alive when loading AOTI")
            obj = Tracked("aoti")
            refs["aoti"] = weakref.ref(obj)
            return obj

        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "expected").mkdir()
            args = SimpleNamespace(output=output)
            with (
                patch.object(
                    batch, "collect_explicit", new=lambda _runtime, _case: (logits, state)
                ),
                patch.object(batch, "capture_state", new=lambda *_args: state),
            ):
                batch.validate_python_stages(
                    args,
                    groups,
                    dict(state=[]),
                    load_reference=load_ref,
                    collect_reference=lambda _ref, _steps: (logits, state),
                    load_eager=load_eager,
                    load_aoti=load_aoti,
                )
            self.assertIsNone(refs["aoti"](), "AOTI runtime still alive after stage exit")

    def test_saved_tensors_roundtrip_mixed_dtypes_and_empty_tensors(self):
        original = [
            torch.tensor([[1.25, -2.5, 3.0]], dtype=torch.bfloat16),
            torch.tensor([[[0.5, -1.5, 2.25]]], dtype=torch.float32),
            torch.empty((1, 0, 2), dtype=torch.bfloat16),
            torch.zeros((2, 4), dtype=torch.float32),
        ]
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "mixed.bin"
            saved = batch._SavedTensors(path, original)
            self.assertEqual(len(saved), len(original))
            restored = list(saved)
            self.assertEqual(len(restored), len(original))
            for actual, expected in zip(restored, original):
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, expected.dtype)
                if expected.numel():
                    self.assertTrue(torch.equal(actual, expected))
            saved.unlink()
            self.assertFalse(path.exists())

    def test_group_waves_keeps_small_models_single_wave_and_partitions_large_models(self):
        small_specs = [dict(shape=[8, 8, 1024, 128], dtype="bfloat16") for _ in range(8)]
        self.assertEqual(len(batch.group_waves(small_specs)), 1)

        large_specs = [
            *[dict(shape=[8, 48, 128, 128], dtype="float32") for _ in range(48)],
            *[dict(shape=[8, 4, 1024, 256], dtype="bfloat16") for _ in range(32)],
        ]
        waves = batch.group_waves(large_specs)
        self.assertGreater(len(waves), 1)
        self.assertEqual(
            [group for wave in waves for group in wave],
            [name for name, _ in batch._GROUP_ROW_ESTIMATES],
        )

    def test_run_batch_waves_merges_manifests_reports_and_cleans_expected_states(self):
        large_specs = [dict(shape=[8, 48, 128, 128], dtype="float32") for _ in range(48)]
        meta = dict(state=large_specs)
        seen_waves = []

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "expected").mkdir()
            args = SimpleNamespace(output=root, packages=root / "pkg", native=root / "exe")

            def fake_python(_args, _meta, *, selected_groups=None):
                self.assertIsNotNone(selected_groups)
                seen_waves.append(list(selected_groups))
                for group in selected_groups:
                    group_dir = root / group
                    group_dir.mkdir()
                    (root / "expected" / f"{group}_prompts.0.states").write_bytes(b"state")
                    (root / "expected" / f"{group}_prompts.f32").write_bytes(b"logits")
                    write_json(
                        group_dir / "cases.json",
                        dict(trace_format=2, cases=[dict(name="prompts", state_steps=[0])]),
                    )
                    write_json(
                        group_dir / "report.json",
                        dict(
                            cases=[dict(name="prompts", positions=[4] + [0] * 7)],
                            failures=[dict(case="prompts")] if group == "b2" else [],
                        ),
                    )
                return list(selected_groups)

            def fake_native(_cmd, **_kwargs):
                cases_file = Path(_cmd[2])
                native_dir = Path(_cmd[3])
                manifest = read_json(cases_file)
                native_dir.mkdir()
                write_json(
                    native_dir / "native.json",
                    dict(
                        loader_pairs=1,
                        no_python_library=True,
                        attention_library=True,
                        cases=[
                            dict(
                                name=case["name"],
                                positions=[4] + [0] * 7,
                                bitwise_parity=True,
                            )
                            for case in manifest["cases"]
                        ],
                    ),
                )

            with patch.object(batch.subprocess, "run", side_effect=fake_native):
                results = batch.run_batch_waves(
                    args, meta, fake_python, batch.validate_native, max_stage_bytes=20 * 1024**3
                )

            expected_groups = [name for name, _ in batch._GROUP_ROW_ESTIMATES]
            self.assertGreater(len(seen_waves), 1)
            self.assertEqual([g for w in seen_waves for g in w], expected_groups)
            self.assertEqual(list(results), expected_groups)
            self.assertEqual(results["b2"]["required_failures"], 1)
            self.assertEqual(results["b1"]["required_failures"], 0)
            self.assertEqual(read_json(root / "summary.json"), results)
            merged_cases = read_json(root / "cases.json")["cases"]
            merged_native = read_json(root / "native/native.json")
            self.assertEqual(
                [c["name"] for c in merged_cases], [f"{g}_prompts" for g in expected_groups]
            )
            self.assertEqual(
                [c["name"] for c in merged_native["cases"]],
                [f"{g}_prompts" for g in expected_groups],
            )
            for group in expected_groups:
                self.assertFalse((root / "expected" / f"{group}_prompts.0.states").exists())
                self.assertTrue((root / "expected" / f"{group}_prompts.f32").is_file())


if __name__ == "__main__":
    unittest.main()
