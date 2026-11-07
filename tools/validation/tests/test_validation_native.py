"""CPU checks for native trace formats, coverage and exact byte boundaries."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from compiler.common.tests.tensors import bits_equal
from tools.validation.common.native import run_native


def raw(tensor):
    return bytes(tensor.contiguous().view(torch.uint8).flatten().tolist())


class NativeTraceContracts(unittest.TestCase):
    def fixture(self, model, *, damage=None, change_report=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        output, packages = root / "output", root / "packages"
        output.mkdir()
        packages.mkdir()
        if model == "qwen3":
            specs = [
                dict(layer=0, kind=kind, shape=[1, 1, 4, 2], dtype="bfloat16", update="inplace")
                for kind in ("key", "value")
            ]
        else:
            specs = [
                dict(layer=0, kind="conv", shape=[1, 2, 2], dtype="bfloat16", update="replace"),
                dict(
                    layer=0, kind="recurrent", shape=[1, 1, 2, 2], dtype="float32", update="replace"
                ),
            ]
        cases = [
            dict(name="teacher", steps=[[1, 0], [1]], order=1),
            dict(
                name="context",
                steps=[[1, 0], [0, 1]],
                dump_cache=True,
                check_overflow=True,
                order=2,
            ),
        ]
        chats = [dict(name="chat", prompt=[1, 0], limit=3, order=0)]
        metadata = dict(decode=dict(model_type=model, state=specs, vocab_size=2, eos_token_ids=[0]))
        # The first value contains FP32 bits that a BF16 round trip would discard.
        logits = torch.tensor([[[1 + 2**-23, 0.25]], [[-0.0, 2.5]]], dtype=torch.float32)
        chat_logits = torch.tensor([[[0.0, 2.0]], [[3.0, 0.0]]], dtype=torch.float32)

        def states(step):
            values = []
            for index, spec in enumerate(specs):
                count = 1
                for dimension in spec["shape"]:
                    count *= dimension
                value = torch.arange(count, dtype=torch.float32).reshape(spec["shape"])
                values.append((value + step + index).to(getattr(torch, spec["dtype"])))
            return tuple(values)

        def native(command, environment, log):
            destination = Path(command[3])
            destination.mkdir()
            rows = {}
            for case in cases:
                lengths = [len(tokens) for tokens in case["steps"]]
                rows[case["name"]] = dict(
                    name=case["name"],
                    rows=len(lengths),
                    position=sum(lengths),
                    prefill_calls=sum(length > 1 for length in lengths),
                    decode_calls=lengths.count(1),
                    stable_kv_storage=True,
                    write_region_checks=True,
                    reset_check=True,
                    kv_write_regions_preserved=True,
                    non_kv_inputs_unchanged=True,
                    overflow_rejected_without_mutation=True,
                )
                if model == "qwen3":
                    (destination / f"{case['name']}.f32").write_bytes(raw(logits))
                    if case.get("dump_cache"):
                        (destination / f"{case['name']}.kv.bf16").write_bytes(
                            b"".join(raw(tensor) for tensor in states(1))
                        )
                else:
                    (destination / f"{case['name']}.states").write_bytes(
                        b"".join(
                            raw(logits[step].bfloat16())
                            + b"".join(raw(tensor) for tensor in states(step))
                            for step in range(len(lengths))
                        )
                    )
            rows["chat"] = dict(
                name="chat",
                ids=[1, 0],
                rows=2,
                position=3,
                prefill_calls=1,
                decode_calls=1,
                stop="eos",
            )
            (destination / "chat.f32").write_bytes(raw(chat_logits))
            if model == "qwen35":
                (destination / "chat.states").write_bytes(
                    b"".join(raw(tensor) for tensor in states(9))
                )
            order = json.loads(Path(command[2]).read_text())["cases"]
            report = dict(
                loader_pairs=1,
                no_python_library=True,
                attention_library=True,
                cases=[rows[case["name"]] for case in order],
            )
            if change_report:
                change_report(report)
            (destination / "native.json").write_text(json.dumps(report))
            if damage:
                name, transform = damage
                path = destination / name
                path.write_bytes(transform(path.read_bytes()))
            log.write_text("Native wire fixture\n")

        with patch("tools.validation.common.native.run", side_effect=native) as run:
            trace, report = run_native(
                root / "xinfer_validate", packages, output, cases, chats, metadata
            )
        run.assert_called_once()
        return trace, report, logits, chat_logits, states

    def test_qwen3_keeps_float32_bits_and_only_selected_final_cache(self):
        trace, report, logits, chat_logits, states = self.fixture("qwen3")
        self.assertEqual([row["name"] for row in report["cases"]], ["chat", "teacher", "context"])
        self.assertEqual(
            set(trace.logits), {("teacher", 0), ("teacher", 1), ("context", 0), ("context", 1)}
        )
        self.assertEqual(set(trace.states), {("context", 1)})
        self.assertEqual(trace.logits["teacher", 0].dtype, torch.float32)
        self.assertTrue(bits_equal(trace.logits["teacher", 0], logits[0]))
        self.assertFalse(bits_equal(logits[0], logits[0].bfloat16().float()))
        for actual, expected in zip(trace.states["context", 1], states(1)):
            self.assertTrue(bits_equal(actual, expected))
        self.assertEqual(len(trace.states["context", 1]), 2)
        self.assertEqual(trace.greedy["chat"]["state"], ())
        self.assertEqual(trace.greedy["chat"]["ids"], [1, 0])
        self.assertTrue(bits_equal(trace.greedy["chat"]["rows"][0], chat_logits[0]))

    def test_qwen35_decodes_every_teacher_state_and_greedy_final_state(self):
        trace, _, logits, _, states = self.fixture("qwen35")
        self.assertEqual(set(trace.states), set(trace.logits))
        self.assertEqual(len(trace.states), 4)
        for key, values in trace.states.items():
            self.assertEqual(len(values), 2)
            self.assertEqual([value.dtype for value in values], [torch.bfloat16, torch.float32])
            self.assertTrue(bits_equal(trace.logits[key], logits[key[1]].bfloat16().float()))
            self.assertEqual(trace.logits[key].dtype, torch.float32)
            for actual, expected in zip(values, states(key[1])):
                self.assertTrue(bits_equal(actual, expected))
        for actual, expected in zip(trace.greedy["chat"]["state"], states(9)):
            self.assertTrue(bits_equal(actual, expected))

    def test_truncated_and_extra_trace_or_cache_bytes_are_rejected(self):
        for model, name in (
            ("qwen3", "teacher.f32"),
            ("qwen3", "context.kv.bf16"),
            ("qwen35", "teacher.states"),
            ("qwen35", "chat.states"),
        ):
            for label, transform in (
                ("truncated", lambda data: data[:-1]),
                ("extra", lambda data: data + b"x"),
            ):
                with self.subTest(model=model, file=name, damage=label):
                    with self.assertRaisesRegex(ValueError, "Invalid native trace size"):
                        self.fixture(model, damage=(name, transform))

    def test_native_case_coverage_must_match_the_manifest(self):
        changes = (
            ("missing", lambda report: report["cases"].pop()),
            ("extra", lambda report: report["cases"].append(dict(report["cases"][0]))),
            ("reordered", lambda report: report["cases"].reverse()),
        )
        for model in ("qwen3", "qwen35"):
            for name, change in changes:
                with self.subTest(model=model, mutation=name):
                    with self.assertRaisesRegex(ValueError, "Native case coverage differs"):
                        self.fixture(model, change_report=change)


class SelectedNativeTrace(unittest.TestCase):
    def test_selected_prefix_state_and_every_raw_logit_are_parsed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, packages = root / "output", root / "packages"
            output.mkdir()
            packages.mkdir()
            specs = [
                dict(
                    layer=0, kind="key", shape=[1, 1, 2048, 2], dtype="bfloat16", update="inplace"
                ),
                dict(layer=0, kind="conv", shape=[1, 2, 2], dtype="bfloat16", update="replace"),
            ]
            metadata = dict(
                decode=dict(
                    model_type="qwen35",
                    state=specs,
                    vocab_size=2,
                    eos_token_ids=[0],
                    cache_length=2048,
                ),
                prefill=dict(token_range=[2, 512]),
            )
            cases = [dict(name="teacher", steps=[[1, 0], [1]])]
            logits = torch.tensor([[[1.0, -0.0]], [[0.25, 2.0]]], dtype=torch.bfloat16)
            key = torch.arange(6).reshape(1, 1, 3, 2).bfloat16()
            conv = torch.ones(1, 2, 2, dtype=torch.bfloat16)

            def native(command, environment, log):
                manifest = json.loads(Path(command[2]).read_text())
                self.assertEqual(manifest["trace_format"], 2)
                self.assertEqual(manifest["cases"][0]["state_steps"], [1])
                destination = Path(command[3])
                destination.mkdir()
                (destination / "teacher.states").write_bytes(raw(logits))
                (destination / "teacher.1.states").write_bytes(raw(key) + raw(conv))
                row = dict(
                    name="teacher",
                    rows=2,
                    position=3,
                    prefill_calls=1,
                    decode_calls=1,
                    stable_kv_storage=True,
                    write_region_checks=True,
                    reset_check=True,
                    kv_write_regions_preserved=True,
                    non_kv_inputs_unchanged=True,
                    state_steps=[1],
                    state_prefix=True,
                )
                (destination / "native.json").write_text(
                    json.dumps(
                        dict(
                            loader_pairs=1,
                            no_python_library=True,
                            attention_library=True,
                            cases=[row],
                        )
                    )
                )
                log.write_text("Selected trace fixture")

            with patch("tools.validation.common.native.run", side_effect=native):
                trace, _ = run_native(
                    root / "native",
                    packages,
                    output,
                    cases,
                    [],
                    metadata,
                    state_keys={("teacher", 1)},
                )
            self.assertTrue(trace.compact_states)
            self.assertEqual(set(trace.states), {("teacher", 1)})
            self.assertEqual(trace.positions, {("teacher", 0): 2, ("teacher", 1): 3})
            self.assertEqual(set(trace.logits), {("teacher", 0), ("teacher", 1)})
            self.assertTrue(bits_equal(trace.raw_logits["teacher", 0], logits[0]))
            self.assertTrue(bits_equal(trace.states["teacher", 1][0], key))
            self.assertTrue(bits_equal(trace.states["teacher", 1][1], conv))


if __name__ == "__main__":
    unittest.main()
