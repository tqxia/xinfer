"""CPU provenance and numerical-policy checks for the functional AOTI gate."""

import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from compiler.common.artifacts import MODEL_MANIFEST_NAME, json_sha256, sha256, write_json
from tools.validation.common.chain import new_report
from tools.validation.common.runner import Trace
from tools.validation.qwen36.aoti import compare_output, reference_info
from tools.validation.qwen36.eager import fixtures


class AOTIValidation(unittest.TestCase):
    def fixture(self, directory, *, native=False):
        model, reference = directory / "model", directory / "reference"
        model.mkdir()
        (reference / "traces").mkdir(parents=True)
        manifest = dict(model="Qwen/test", revision="pinned", files={"weights": "a" * 64})
        write_json(model / MODEL_MANIFEST_NAME, manifest)
        groups, chats = fixtures([1, 2], [3], "matrix", 1024)
        protocol = dict(
            identity=dict(
                manifest_sha256=sha256(model / MODEL_MANIFEST_NAME),
                layers=40,
                capacity=1024,
                profile="matrix",
            ),
            groups=groups,
            chats=chats,
            state=[],
            eos_token_ids=[0],
            input_sha256=json_sha256(dict(groups=groups, chats=chats)),
        )
        pair = dict(
            prefill=dict(token_range=[2, 512], package_sha256="b" * 64),
            decode=dict(
                token_range=[1, 1],
                package_sha256="c" * 64,
                checkpoint=dict(manifest=manifest),
                cache_length=1024,
                state=[],
                eos_token_ids=[0],
            ),
        )
        summary = dict(
            model_type="qwen36",
            stage="adapted",
            execution_passed=True,
            comparison_passed=False,
            failures=[dict(tensor="known_HF_failure")],
            teacher_steps=34,
            state_checkpoints=22,
            greedy_cases=2,
        )
        if native:
            summary.update(
                stage="python_aoti",
                functional_passed=False,
                package_sha256={kind: item["package_sha256"] for kind, item in pair.items()},
                hf_numerical_reference=dict(passed=False, failures=1369, summary_sha256="d" * 64),
            )
        for case in [case for group in groups for case in group] + chats:
            # This helper preflights provenance/coverage; execution decodes the traces.
            (reference / "traces" / f"{case['name']}.pt").write_bytes(b"fixture")
        write_json(reference / "summary.json", summary)
        write_json(reference / "protocol.json", protocol)
        return model, reference, pair, protocol, summary

    def test_completed_adapted_reference_preserves_known_hf_failure(self):
        with TemporaryDirectory() as temporary:
            model, reference, pair, protocol, _ = self.fixture(Path(temporary))
            actual, _, hf = reference_info(reference, "python", pair, model)
            self.assertEqual(actual, protocol)
            self.assertFalse(hf["passed"])
            self.assertEqual(hf["failures"], 1)
            self.assertEqual(hf["summary_sha256"], sha256(reference / "summary.json"))

    def test_rejects_wrong_failed_execution_or_incomplete_reference(self):
        mutations = (
            lambda report: report.update(stage="hf"),
            lambda report: report.update(execution_passed=False),
            lambda report: report.update(error=dict(message="incomplete")),
            lambda report: report.update(teacher_steps=31),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), TemporaryDirectory() as temporary:
                model, reference, pair, _, summary = self.fixture(Path(temporary))
                mutation(summary)
                write_json(reference / "summary.json", summary)
                with self.assertRaises(ValueError):
                    reference_info(reference, "python", pair, model)

    def test_rejects_checkpoint_capacity_and_input_identity_changes(self):
        for change in ("manifest", "capacity", "input"):
            with self.subTest(change=change), TemporaryDirectory() as temporary:
                model, reference, pair, protocol, _ = self.fixture(Path(temporary))
                if change == "manifest":
                    pair = copy.deepcopy(pair)
                    pair["decode"]["checkpoint"]["manifest"]["revision"] = "other"
                elif change == "capacity":
                    protocol["identity"]["capacity"] = 2048
                else:
                    protocol["groups"][0][0]["steps"][0] = [99]
                write_json(reference / "protocol.json", protocol)
                with self.assertRaises(ValueError):
                    reference_info(reference, "python", pair, model)

    def test_native_reference_binds_packages_without_hiding_prior_failures(self):
        with TemporaryDirectory() as temporary:
            model, reference, pair, _, _ = self.fixture(Path(temporary), native=True)
            _, source, hf = reference_info(reference, "native", pair, model)
            self.assertFalse(source["functional_passed"])
            self.assertFalse(hf["passed"])
            self.assertEqual(hf["failures"], 1369)
            pair["decode"]["package_sha256"] = "e" * 64
            with self.assertRaisesRegex(ValueError, "packages differ"):
                reference_info(reference, "native", pair, model)

    def test_compiler_float_gate_and_native_raw_bit_gate_are_distinct(self):
        raw = torch.tensor([[1.0, -0.0]], dtype=torch.bfloat16)
        expected = Trace(logits={("case", 0): raw.float()}, raw_logits={("case", 0): raw})
        close = Trace(logits={("case", 0): torch.tensor([[1.01, -0.0]])})
        numerical = new_report("qwen36")
        compare_output(numerical, close, expected, [], exact=False)
        self.assertFalse(numerical["failures"])
        self.assertFalse(numerical["comparisons"][0]["bitwise_equal"])
        changed_zero = raw.clone()
        changed_zero[0, 1] = 0.0
        actual = Trace(logits={("case", 0): raw.float()}, raw_logits={("case", 0): changed_zero})
        native = new_report("qwen36")
        compare_output(native, actual, expected, [], exact=True)
        self.assertEqual([row["tensor"] for row in native["failures"]], ["raw_logits"])

    def test_greedy_failure_does_not_compare_subsequent_different_histories(self):
        rows = [torch.tensor([[1.0, 0.0]]) for _ in range(3)]
        actual = Trace(
            greedy={
                "chat": dict(
                    ids=[1, 7, 8],
                    rows=[*rows[:2], rows[2] * 100],
                    position=4,
                    state=[torch.tensor([100.0])],
                )
            }
        )
        expected = Trace(
            greedy={"chat": dict(ids=[1, 2, 3], rows=rows, position=4, state=[torch.tensor([0.0])])}
        )
        report = new_report("qwen36")
        compare_output(report, actual, expected, [], exact=False)
        self.assertEqual([row["tensor"] for row in report["failures"]], ["greedy_tokens"])
        self.assertEqual([row["step"] for row in report["comparisons"]], [0, 1])


if __name__ == "__main__":
    unittest.main()
