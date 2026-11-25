"""Qwen3 builds live HF references and checks the HF generation API on CPU."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch, sentinel

import torch

from compiler.common.artifacts import MODEL_MANIFEST_NAME
from tools.tests.test_packages import metadata
from tools.validation.common.prompts import chat_prompts
from tools.validation.qwen3 import model


class LiveHFReferences(unittest.TestCase):
    def test_batch_fixtures_cover_supported_rows_and_capacity(self):
        from tools.validation.common import batch as validate_batch

        tokenizer = Mock()
        tokenizer.encode.return_value = [11, 12, 13]
        for capacity in (512, 1024):
            with patch.object(
                validate_batch,
                "encode_prompt",
                side_effect=lambda _, text: list(range(len(text))),
            ):
                groups = validate_batch.case_groups(tokenizer, capacity)
            self.assertEqual(list(groups), [f"b{b}" for b in range(1, 9)] + ["transitions"])
            for group, (texts, cases) in groups.items():
                batch = len(texts)
                names = [case["name"] for case in cases]
                self.assertEqual(len(names), len(set(names)))
                for case in cases:
                    positions = [0] * batch
                    for rows in case["batch_steps"]:
                        self.assertTrue(1 <= len(rows) <= batch)
                        self.assertTrue(any(rows))
                        for row, tokens in enumerate(rows):
                            self.assertLessEqual(len(tokens), 512)
                            positions[row] += len(tokens)
                            self.assertGreater(positions[row], 0)
                            self.assertLessEqual(positions[row], capacity)
                    self.assertEqual(case["state_steps"], [len(case["batch_steps"]) - 1])
                if group == "transitions":
                    self.assertEqual(names, ["changing_batch", "growing_batch"])
                    self.assertEqual(
                        [[len(rows) for rows in case["batch_steps"]] for case in cases],
                        [[8, 1, 3, 5, 8], [1, 3, 8]],
                    )
                    continue
                self.assertEqual(group, f"b{batch}")
                self.assertEqual(names[0], "prompts")
                self.assertTrue({"reset", "poison", "padding"}.issubset(names))
                self.assertNotIn("changing_batch", names)
                self.assertNotIn("growing_batch", names)
                self.assertEqual("perturb" in names, batch > 1)
                self.assertEqual(any(name.startswith("lengths_") for name in names), batch > 1)
                self.assertEqual("staggered_resume" in names, batch > 2)
                if batch > 1:
                    perturb = next(c for c in cases if c["name"] == "perturb")
                    self.assertEqual(perturb["batch_steps"][0][1:], cases[0]["batch_steps"][0][1:])
                    self.assertEqual(perturb["batch_steps"][1:], cases[0]["batch_steps"][1:])
                if batch > 2:
                    staggered = next(c for c in cases if c["name"] == "staggered_resume")
                    self.assertEqual(
                        [[len(row) for row in step] for step in staggered["batch_steps"][-batch:]],
                        [[int(row == active) for row in range(batch)] for active in range(batch)],
                    )
            if capacity == 1024:
                all_cases = [case for _, cases in groups.values() for case in cases]
                self.assertEqual(len(all_cases), 142)
                self.assertEqual(sum(len(case["batch_steps"]) for case in all_cases), 583)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / MODEL_MANIFEST_NAME).write_text(
            json.dumps(dict(model="Qwen/Qwen3-0.6B", revision="pinned"))
        )
        bundle = metadata()
        bundle.update(torch_version=torch.__version__, vocab_size=64, eos_token_ids=[1])
        self.prompts = chat_prompts("qwen3")
        self.prompt_ids = {
            item["text"]: [10 + index, 20 + index] for index, item in enumerate(self.prompts)
        }
        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = lambda messages, **options: list(
            self.prompt_ids[messages[0]["content"]]
        )
        with patch.object(model, "load_tokenizer", return_value=tokenizer):
            self.suite = model.Suite(self.root, bundle)

        self.original = SimpleNamespace(
            config=SimpleNamespace(vocab_size=64),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            generation_config=SimpleNamespace(
                bos_token_id=9, pad_token_id=0, do_sample=True, top_p=0.7
            ),
        )
        self.original.generate = Mock(side_effect=self.generate)

        class CPURuntime(model.HFRuntime):
            state = ()

            def reset(runtime, fill=0):
                runtime.position = 0
                runtime.step_index = 0

            def step(runtime, tokens):
                if runtime.position == 0:
                    runtime.sequence = LiveHFReferences.sequence(tokens)
                token = runtime.sequence[min(runtime.step_index, len(runtime.sequence) - 1)]
                runtime.position += len(tokens)
                runtime.step_index += 1
                logits = torch.zeros(1, 64, dtype=torch.bfloat16)
                logits[0, token] = 10
                return logits

        self.runtime = CPURuntime(self.original, [], capacity=self.suite.capacity)
        self.cache = self.enterContext(
            patch.object(model, "StaticCache", return_value=sentinel.fresh_cache)
        )

    @staticmethod
    def sequence(prompt):
        return [40 + prompt[0] % 3, 45 + prompt[0] % 3, 1]

    def generate(self, **arguments):
        ids = arguments["input_ids"]
        suffix = torch.tensor([self.sequence(ids[0].tolist())], dtype=torch.int64)
        return torch.cat((ids, suffix), dim=1)

    def test_live_fixtures_need_no_saved_references_and_preserve_request_cases(self):
        cases, chats = self.suite.fixtures(self.runtime)
        teachers = {case["fixture"]: case for case in cases if case.get("fixture")}
        self.assertEqual(list(teachers), [item["name"] for item in self.prompts])
        for order, item in enumerate(self.prompts, start=1):
            prompt = self.prompt_ids[item["text"]]
            generated = self.sequence(prompt)
            teacher = teachers[item["name"]]
            self.assertEqual(teacher["name"], item["name"] + "_teacher")
            self.assertEqual(teacher["steps"], [prompt, *[[token] for token in generated[:-1]]])
            self.assertEqual(teacher["order"], order)
        self.assertEqual(
            [case["name"] for case in chats],
            [
                "capital_initial",
                "capital_after_full",
                "arithmetic_after_capital",
                "chinese_after_arithmetic",
                "capital_repeat",
                "limit_one",
                "limit_three",
            ],
        )
        self.assertEqual([case["order"] for case in chats], [0, 18, 19, 20, 21, 22, 23])
        self.assertEqual([case["limit"] for case in chats], [64, 64, 64, 64, 64, 1, 3])
        for case in chats:
            self.assertEqual(case["expected_ids"], self.sequence(case["prompt"])[: case["limit"]])
        self.assertEqual(self.original.generate.call_count, len(self.prompts))
        self.assertEqual({path.name for path in self.root.iterdir()}, {MODEL_MANIFEST_NAME})
        boundary = next(case for case in cases if case["name"] == "prefix_boundaries")
        self.assertTrue(boundary["check_overflow"])
        self.assertEqual(sum(map(len, boundary["steps"])), 1024)

    def test_hf_capacity_uses_explicit_or_state_bounds_before_model_default(self):
        specs = [dict(shape=[1, 8, 1024, 128], update="inplace")]
        with patch.object(
            model, "model_context_limit", side_effect=AssertionError("Unexpected model fallback")
        ) as ceiling:
            self.assertEqual(model.HFRuntime(self.original, specs, capacity=2048).capacity, 2048)
            self.assertEqual(model.HFRuntime(self.original, specs).capacity, 1024)
            ceiling.assert_not_called()
        with patch.object(model, "model_context_limit", return_value=32768) as ceiling:
            self.assertEqual(model.HFRuntime(self.original, []).capacity, 32768)
            ceiling.assert_called_once_with(self.original.config)

    def test_greedy_replay_reference_uses_default_bf16_model(self):
        runtime = Mock(spec=["close"])
        with (
            patch.object(model, "load_model", return_value=self.original) as loader,
            patch.object(model, "HFRuntime", return_value=runtime),
            patch.object(model.torch.cuda, "empty_cache"),
        ):
            with self.suite.reference() as actual:
                self.assertIs(actual, runtime)
            loader.assert_called_once_with(self.root)
        runtime.close.assert_called_once_with()

    def test_hf_generate_uses_fresh_greedy_config_and_cache(self):
        prompt = next(iter(self.prompt_ids.values()))
        self.runtime.position = 17
        self.runtime.cache = sentinel.manual_cache
        actual = self.runtime.generate(prompt, 64, [1])
        self.assertEqual(actual, self.sequence(prompt))
        self.assertEqual(self.runtime.position, 17)
        self.assertIs(self.runtime.cache, sentinel.manual_cache)
        self.cache.assert_called_once_with(self.original.config, max_cache_len=1024)
        arguments = self.original.generate.call_args.kwargs
        self.assertEqual(arguments["input_ids"].tolist(), [prompt])
        self.assertEqual(arguments["attention_mask"].tolist(), [[1] * len(prompt)])
        self.assertIs(arguments["past_key_values"], sentinel.fresh_cache)
        self.assertNotIn("use_model_defaults", arguments)
        config = arguments["generation_config"]
        self.assertIsNot(config, self.original.generation_config)
        self.assertFalse(config.do_sample)
        self.assertEqual(config.max_new_tokens, 64)
        self.assertEqual(config.eos_token_id, [1])
        self.assertEqual(config.bos_token_id, 9)
        self.assertEqual(config.pad_token_id, 0)
        self.assertTrue(config.use_cache)
        self.assertTrue(config.disable_compile)

    def test_hf_generate_mismatch_rejects_live_reference(self):
        self.original.generate.side_effect = lambda **arguments: torch.cat(
            (arguments["input_ids"], torch.tensor([[5]], dtype=torch.int64)), dim=1
        )
        with self.assertRaisesRegex(
            ValueError, r"Manual HF decoding disagrees with HF\.generate\(\) for capital"
        ):
            self.suite.fixtures(self.runtime)


class BatchValidation(unittest.TestCase):
    def test_sparse_traces_preserve_all_bytes_and_trailing_zeros(self):
        from tools.validation.common.batch import write_trace
        from tools.validation.common.validation import raw_bytes

        values = (
            (),
            (torch.zeros(131073, dtype=torch.uint8),),
            (
                torch.zeros(65536, dtype=torch.uint8),
                torch.tensor([0, 1, 0, 255], dtype=torch.uint8),
                torch.zeros(65536, dtype=torch.uint8),
                torch.tensor([-0.0, 1.0], dtype=torch.float32),
                torch.zeros(131073, dtype=torch.uint8),
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace"
            for tensors in values:
                expected = b"".join(raw_bytes(t) for t in tensors)
                digest = write_trace(path, iter(tensors))
                self.assertEqual(path.read_bytes(), expected)
                self.assertEqual(digest, hashlib.sha256(expected).hexdigest())

    def test_models_are_staged_once_and_closed_before_the_next_stage(self):
        from tools.validation.common import batch as common_batch
        from tools.validation.qwen3 import validate_batch as batch

        names = [f"b{size}" for size in range(1, 9)] + ["transitions"]
        groups = {
            name: (
                [name],
                [dict(name="prompts", batch_steps=[[[1]]], state_steps=[0])],
            )
            for name in names
        }
        logits = [torch.ones(1, 4, dtype=torch.bfloat16)]
        state = [torch.ones(1, 1, 1, 2, dtype=torch.bfloat16)]
        for fail_at in (None, "b4", "transitions"):
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "expected").mkdir()
                args = SimpleNamespace(
                    model_dir=Path("model"), packages=Path("package"), output=root
                )
                meta = dict(state=[], vocab_size=64, cache_length=1024)
                events = []

                class TrackedRuntime:
                    def __init__(self, label):
                        self.label = label
                        self.position = 1
                        self.positions = [1] + [0] * 7
                        self.closed = False

                    def close(self):
                        self.closed = True
                        events.append(f"close:{self.label}")

                eager = TrackedRuntime("eager")
                aoti = TrackedRuntime("aoti")
                seen_aoti = []

                def load_model_side_effect(_model_dir):
                    if not events:
                        events.append("load:hf")
                        return sentinel.hf
                    events.append("load:eager")
                    return sentinel.base

                def collect_ref(hf, _steps):
                    self.assertIs(hf, sentinel.hf)
                    return logits, state

                def collect_exp(runtime, _case):
                    if runtime is aoti:
                        self.assertTrue(eager.closed)
                        group = names[len(seen_aoti)]
                        seen_aoti.append(group)
                        if group == fail_at:
                            raise RuntimeError("case failed")
                    return logits, state

                with (
                    patch.object(batch, "load_packaged_attention_ops") as library,
                    patch.object(
                        batch, "load_tokenizer", return_value=sentinel.tokenizer
                    ) as tokenizer,
                    patch.object(batch, "case_groups", return_value=groups) as fixtures,
                    patch.object(batch, "load_model", side_effect=load_model_side_effect) as loader,
                    patch.object(batch, "ModelStep", return_value=sentinel.adapter) as adapter,
                    patch.object(batch, "ExplicitRuntime", return_value=eager),
                    patch.object(batch, "load_aoti_runtime", return_value=aoti) as compiled,
                    patch.object(batch, "collect_hf", side_effect=collect_ref),
                    patch.object(common_batch, "collect_explicit", side_effect=collect_exp),
                    patch.object(common_batch, "capture_state", return_value=state),
                ):
                    if fail_at is None:
                        self.assertEqual(batch.validate_python(args, meta), names)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "case failed"):
                            batch.validate_python(args, meta)
                    self.assertEqual(
                        seen_aoti, names[: names.index(fail_at) + 1] if fail_at else names
                    )
                    self.assertEqual(loader.call_count, 2)
                    adapter.assert_called_once_with(sentinel.base)
                    library.assert_called_once_with(args.packages)
                    tokenizer.assert_called_once_with(args.model_dir)
                    fixtures.assert_called_once_with(sentinel.tokenizer, 1024)
                    compiled.assert_called_once_with(args.packages, model_type="qwen3", specs=[])
                self.assertEqual(events, ["load:hf", "load:eager", "close:eager", "close:aoti"])
                self.assertFalse((root / "_staging").exists())

    def test_one_native_run_preserves_failures_and_requires_complete_parity(self):
        from tools.validation.common import batch
        from compiler.common.artifacts import read_json, write_json

        groups = {f"b{size}": ["prompt"] for size in range(1, 9)}
        groups["transitions"] = ["changing_batch", "growing_batch"]
        for defect in (None, "coverage", "positions", "parity"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "expected").mkdir()
                args = SimpleNamespace(output=root, packages=root / "package", native=root / "exe")
                for group, names in groups.items():
                    directory = root / group
                    directory.mkdir()
                    for name in names:
                        (root / "expected" / f"{group}_{name}.0.states").write_bytes(b"state")
                        (root / "expected" / f"{group}_{name}.f32").write_bytes(b"logits")
                    write_json(
                        directory / "cases.json",
                        dict(cases=[dict(name=name, state_steps=[0]) for name in names]),
                    )
                    write_json(
                        directory / "report.json",
                        dict(
                            cases=[dict(name=name, positions=[3] + [0] * 7) for name in names],
                            failures=[dict(case=names[0])]
                            if group in ("b2", "transitions")
                            else [],
                        ),
                    )

                def execute(*args, **kwargs):
                    manifest = read_json(root / "cases.json")
                    self.assertEqual(len(manifest["cases"]), 10)
                    rows = [
                        dict(name=f"{group}_{name}", positions=[3] + [0] * 7, bitwise_parity=True)
                        for group, names in groups.items()
                        for name in names
                    ]
                    self.assertEqual(
                        [case["name"] for case in manifest["cases"]],
                        [row["name"] for row in rows],
                    )
                    if defect == "coverage":
                        rows.pop()
                    elif defect == "positions":
                        rows[-1]["positions"][0] += 1
                    elif defect == "parity":
                        rows[-1]["bitwise_parity"] = False
                    (root / "native").mkdir()
                    write_json(
                        root / "native/native.json",
                        dict(
                            cases=rows,
                            loader_pairs=1,
                            no_python_library=True,
                            attention_library=True,
                        ),
                    )

                with patch.object(batch.subprocess, "run", side_effect=execute) as native:
                    if defect:
                        with self.assertRaises(ValueError):
                            batch.validate_native(args, list(groups))
                        # Failed native parity preserves expected state dumps for inspection.
                        self.assertTrue((root / "expected/b1_prompt.0.states").is_file())
                    else:
                        result = batch.validate_native(args, list(groups))
                        self.assertEqual(result["b2"]["required_failures"], 1)
                        self.assertEqual(result["transitions"]["required_failures"], 1)
                        self.assertEqual(list(result), list(groups))
                        for group, names in groups.items():
                            report = read_json(root / group / "report.json")
                            self.assertTrue(report["native_bitwise_parity"])
                            self.assertEqual(
                                [case["name"] for case in report["native"]["cases"]], names
                            )
                            for name in names:
                                self.assertFalse(
                                    (root / "expected" / f"{group}_{name}.0.states").exists()
                                )
                                self.assertTrue(
                                    (root / "expected" / f"{group}_{name}.f32").is_file()
                                )
                    native.assert_called_once()


if __name__ == "__main__":
    unittest.main()
