"""CPU checks for bounded eager fixtures and matched-trajectory comparisons."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from compiler.common.tests.tensors import bits_equal
from compiler.common.artifacts import write_json
from tools.validation.common.chain import new_report
from tools.validation.common.runner import Trace
from tools.validation.qwen36.eager import (
    compare_case,
    fixtures,
    load_trace,
    load_reference_protocol,
    request_guards,
    save_trace,
)


class EagerValidation(unittest.TestCase):
    def test_reference_requires_a_completed_passing_hf_stage(self):
        identity = dict(manifest_sha256="checkpoint", layers=4, capacity=1024)
        protocol = dict(identity=identity, groups=[], chats=[])
        summary = dict(
            model_type="qwen36", stage="hf", execution_passed=True, passed=True, failures=[]
        )
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_json(directory / "protocol.json", protocol)
            with self.assertRaisesRegex(ValueError, "missing summary"):
                load_reference_protocol(directory, identity)
            write_json(directory / "summary.json", summary)
            self.assertEqual(load_reference_protocol(directory, identity), protocol)
            for update in (
                dict(stage="adapted"),
                dict(stage="fp8"),
                dict(passed=False),
                dict(execution_passed=False),
                dict(passed=1),
                dict(model_type="qwen35"),
                dict(failures=[dict(tensor="logits")]),
            ):
                with self.subTest(update=update):
                    write_json(directory / "summary.json", summary | update)
                    with self.assertRaisesRegex(ValueError, "completed passing.*HF stage"):
                        load_reference_protocol(directory, identity)
            write_json(directory / "summary.json", summary)
            with self.assertRaisesRegex(ValueError, "profile/checkpoint differs"):
                load_reference_protocol(directory, identity | dict(layers=40))

    def test_matrix_partitions_reset_controls_and_budget(self):
        groups, chats = fixtures([11, 12, 13], [21, 22], "matrix", 1024)
        cases = {case["name"]: case for group in groups for case in group}
        self.assertEqual(len(cases), 11)
        self.assertEqual(sum(len(case["steps"]) for case in cases.values()), 34)
        for length in (1, 2, 96, 97, 98, 511, 512):
            self.assertEqual(list(map(len, cases[f"length_{length}"]["steps"])), [length, 1, 1])
        self.assertEqual(list(map(len, cases["multicall_513"]["steps"])), [512, 1, 1, 1])
        self.assertEqual(list(map(len, cases["nonzero_prefill"]["steps"])), [96, 98, 1])
        self.assertEqual(cases["poison_98"]["same_as"], "length_98")
        self.assertEqual(cases["repeat_98"]["same_as"], "length_98")
        self.assertEqual(cases["poison_98"]["poison_kv"], 3)
        self.assertLessEqual(max(map(len, groups)), 3)
        self.assertEqual([case["limit"] for case in chats], [3, 1])
        with self.assertRaises(ValueError):
            fixtures([11], [21], "matrix", 512)

    def test_smoke_does_not_fill_large_capacity(self):
        groups, _ = fixtures([11, 12], [21], "smoke", 262144)
        self.assertEqual(
            sum(len(step) for group in groups for case in group for step in case["steps"]), 4
        )

    def test_trace_roundtrip_preserves_raw_bits_and_hook_coverage(self):
        raw = torch.tensor([[-0.0, 0.0, 1.0]], dtype=torch.bfloat16)
        trace = Trace(
            logits={("case", 0): raw.float()},
            raw_logits={("case", 0): raw},
            states={("case", 0): (raw,)},
            positions={("case", 0): 1},
            compact_states=True,
        )
        blocks = {("case", 0, 3): raw[None, :]}
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.pt"
            save_trace(path, trace, blocks)
            loaded, captured = load_trace(path)
        self.assertTrue(loaded.compact_states)
        self.assertTrue(bits_equal(loaded.raw_logits["case", 0], raw))
        self.assertEqual(loaded.positions, trace.positions)
        self.assertEqual(captured.keys(), blocks.keys())
        self.assertTrue(bits_equal(captured["case", 0, 3], blocks["case", 0, 3]))

    def test_greedy_divergence_keeps_failure_but_excludes_later_histories(self):
        def row(token):
            result = torch.zeros(1, 10)
            result[0, token] = 1
            return result

        expected = Trace(
            greedy={
                "chat": dict(ids=[1, 2, 3], rows=[row(1), row(2), row(3)], state=[], position=5)
            }
        )
        actual = Trace(
            greedy={
                "chat": dict(
                    ids=[1, 4, 9], rows=[row(1), row(4), row(9) * 1000], state=[], position=5
                )
            }
        )
        report = new_report("qwen36")
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.pt"
            save_trace(path, expected, {})
            compare_case(report, "adapted", actual, {}, path, [])
        self.assertEqual(len(report["greedy_failures"]), 1)
        self.assertTrue(report["failures"])
        numerical = [row for row in report["comparisons"] if row["tensor"] == "logits"]
        self.assertEqual([row["step"] for row in numerical], [0, 1])

    def test_rejection_probe_restores_position_and_detects_state_mutation(self):
        class Runtime:
            capacity, position = 1024, 7
            state = [torch.ones(2)]

            def step(self, tokens):
                raise ValueError("Rejected before execution")

        runtime = Runtime()
        proof = request_guards(runtime)
        self.assertEqual(runtime.position, 7)
        self.assertFalse(proof["full_context_window_executed"])
        self.assertTrue(proof["rejected_requests_preserve_state_bytes_and_pointers"])

        def broken(tokens):
            runtime.state[0].zero_()
            raise ValueError("Rejected after mutation")

        runtime.step = broken
        with self.assertRaisesRegex(ValueError, "state bytes"):
            request_guards(runtime)


if __name__ == "__main__":
    unittest.main()
