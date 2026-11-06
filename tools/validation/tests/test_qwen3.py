"""Qwen3 builds live HF references and checks the HF generation API on CPU."""

import importlib.metadata
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch, sentinel

from compiler.common.artifacts import MODEL_MANIFEST_NAME
from tools.validation.common.prompts import chat_prompts


try:
    QWEN3_ENVIRONMENT = importlib.metadata.version("transformers") == "4.51.3"
except importlib.metadata.PackageNotFoundError:
    QWEN3_ENVIRONMENT = False

if QWEN3_ENVIRONMENT:
    import torch

    from compiler.qwen3.tests.test_packages import metadata
    from tools.validation.qwen3 import model


@unittest.skipUnless(QWEN3_ENVIRONMENT, "Qwen3 validation requires transformers==4.51.3")
class LiveHFReferences(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / MODEL_MANIFEST_NAME).write_text(json.dumps({}))
        pair = {kind: metadata() for kind in ("prefill", "decode")}
        for entry in pair.values():
            entry.update(torch_version=torch.__version__, vocab_size=64, eos_token_ids=[1])
        pair["prefill"].update(entry_point="prefill", token_range=[2, 512])
        self.prompts = chat_prompts("qwen3")
        self.prompt_ids = {
            item["text"]: [10 + index, 20 + index] for index, item in enumerate(self.prompts)
        }
        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = lambda messages, **options: list(
            self.prompt_ids[messages[0]["content"]]
        )
        with patch.object(model, "load_tokenizer", return_value=tokenizer):
            self.suite = model.Suite(self.root, pair)

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

        self.runtime = CPURuntime(self.original, [])
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

    def test_hf_generate_uses_fresh_greedy_config_and_cache(self):
        prompt = next(iter(self.prompt_ids.values()))
        self.runtime.position = 17
        self.runtime.cache = sentinel.manual_cache
        actual = self.runtime.generate(prompt, 64, [1])
        self.assertEqual(actual, self.sequence(prompt))
        self.assertEqual(self.runtime.position, 17)
        self.assertIs(self.runtime.cache, sentinel.manual_cache)
        self.cache.assert_called_once_with(
            self.original.config,
            1,
            1024,
            device=self.original.device,
            dtype=torch.bfloat16,
        )
        arguments = self.original.generate.call_args.kwargs
        self.assertEqual(arguments["input_ids"].tolist(), [prompt])
        self.assertEqual(arguments["attention_mask"].tolist(), [[1] * len(prompt)])
        self.assertIs(arguments["past_key_values"], sentinel.fresh_cache)
        self.assertFalse(arguments["use_model_defaults"])
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


if __name__ == "__main__":
    unittest.main()
