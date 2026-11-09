"""CPU checks for production CLI references and chunked prompt-end selection."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from tools.validation.common.runner import Trace
from tools.validation.common.validation import raw_bytes
from tools.validation.qwen36.cli import fixtures
from tools.validation.qwen36.eager import save_trace


class CLIFixtures(unittest.TestCase):
    def reference(self, directory):
        (directory / "traces").mkdir()
        rows = [torch.tensor([[1.0, 3.0, 2.0]]), torch.tensor([[4.0, 5.0, 6.0]])]
        chats = [
            dict(name="capital", prompt=[8, 9], limit=3),
            dict(name="limit_one", prompt=[8, 9], limit=1),
        ]
        for case in chats:
            ids = [1, 2] if case["name"] == "capital" else [1]
            trace = Trace(
                greedy={
                    case["name"]: dict(
                        ids=ids,
                        rows=rows[: len(ids)],
                        state=[],
                        position=len(case["prompt"]) + len(ids) - 1,
                    )
                }
            )
            save_trace(directory / "traces" / f"{case['name']}.pt", trace, {})
        name = "multicall_513"
        case = dict(name=name, prompt_steps=2, steps=[[7] * 512, [8], [9], [10]])
        prompt_end = torch.tensor([[3.0, 2.0, -0.0]])
        save_trace(
            directory / "traces" / f"{name}.pt",
            Trace(
                logits={
                    (name, 0): rows[0],
                    (name, 1): prompt_end,
                    (name, 2): rows[0],
                    (name, 3): rows[1],
                }
            ),
            {},
        )
        return dict(chats=chats, groups=[[case]], eos_token_ids=[2]), prompt_end

    def test_chunked_fixture_uses_final_prompt_row_and_one_output(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            protocol, prompt_end = self.reference(directory)
            actual = fixtures(protocol, directory)
            self.assertEqual(
                [case["name"] for case in actual], ["capital", "limit_one", "multicall_513"]
            )
            capital, limit_one, chunked = actual
            self.assertEqual(
                (capital["ids"], capital["position"], capital["stop"]), ([1, 2], 3, "eos")
            )
            self.assertEqual((limit_one["ids"], limit_one["stop"]), ([1], "limit"))
            self.assertEqual(chunked["prompt"], [7] * 512 + [8])
            self.assertEqual((chunked["limit"], chunked["ids"], chunked["position"]), (1, [0], 513))
            self.assertEqual(chunked["data"], raw_bytes(prompt_end))
            self.assertFalse(torch.cuda.is_initialized())

    def test_nonfinite_reference_is_rejected_before_native_launch(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            protocol, _ = self.reference(directory)
            save_trace(
                directory / "traces" / "capital.pt",
                Trace(
                    greedy={
                        "capital": dict(
                            ids=[1],
                            rows=[torch.tensor([[float("nan"), 0.0, 1.0]])],
                            state=[],
                            position=2,
                        )
                    }
                ),
                {},
            )
            with self.assertRaisesRegex(ValueError, "Invalid reference logit rows"):
                fixtures(protocol, directory)


if __name__ == "__main__":
    unittest.main()
