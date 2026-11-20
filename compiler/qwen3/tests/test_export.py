"""Export capacity checks and adapted-model graph integration tests."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from compiler.qwen3 import export


class ExportContextLength(unittest.TestCase):
    def test_workflow_errors_are_not_converted_to_cli_usage_errors(self):
        with (
            patch.object(export, "export_model", side_effect=ValueError("graph export failed")),
            self.assertRaisesRegex(ValueError, "graph export failed"),
        ):
            export.main([])

    def test_context_argument_is_checked_before_model_loading(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps(dict(max_position_embeddings=40960)))
            argv = ["export", "--model-dir", str(root), "--output", str(root / "bundle")]
            for capacity in (511, 40961):
                with (
                    self.subTest(capacity=capacity),
                    patch("sys.argv", argv + ["--max-context-length", str(capacity)]),
                    patch.object(export, "load_model") as load,
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    export.main()
                self.assertEqual(error.exception.code, 2)
                load.assert_not_called()
                self.assertFalse((root / "bundle").exists())
            for capacity in (None, 1024, 40960):
                arguments = [] if capacity is None else ["--max-context-length", str(capacity)]
                output = io.StringIO()
                with (
                    self.subTest(capacity=capacity),
                    patch("sys.argv", argv + arguments),
                    patch.object(
                        export, "load_model", side_effect=RuntimeError("GPU load reached")
                    ) as load,
                    redirect_stdout(output),
                    self.assertRaisesRegex(RuntimeError, "GPU load reached"),
                ):
                    export.main()
                load.assert_called_once_with(root)
                event = json.loads(output.getvalue())
                self.assertEqual(event["cache_length"], 40960 if capacity is None else capacity)
                self.assertEqual(event["model_max_position_embeddings"], 40960)
                self.assertEqual(event["max_prefill_tokens"], 512)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ExportGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        config = Qwen3Config(
            vocab_size=128,
            hidden_size=1024,
            intermediate_size=3072,
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            max_position_embeddings=1024,
            use_sliding_window=False,
        )
        config._attn_implementation = "sdpa"
        model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
        cls.step = export.ModelStep(model.to(device="cuda", dtype=torch.bfloat16))

    @classmethod
    def tearDownClass(cls):
        del cls.step

    @torch.inference_mode()
    def test_dynamic_batch_export_and_cache_writes(self):
        for kind in ("prefill", "decode"):
            program = export.export_step(self.step, kind, 512)
            example = program.example_inputs[0]
            ids = next(node for node in program.graph.nodes if node.target == "input_ids")
            batch_range = program.range_constraints[ids.meta["val"].shape[0].node.expr]
            # Exported Python can accept B=1 even with a [2,8] range; AOTI cannot.
            self.assertEqual((batch_range.lower, batch_range.upper), (1, 8))
            self.assertFalse(program.graph_signature.buffers_to_mutate)
            targets = [node.target for node in program.graph.nodes if node.op == "call_function"]
            sites = self.step.model.config.num_hidden_layers
            self.assertEqual(targets.count(torch.ops.xinfer.batched_flash_attention.default), sites)
            self.assertEqual(targets.count(torch.ops.xinfer.append_kv_batched.default), sites)
            self.assertNotIn(torch.ops.aten.index_copy.default, targets)
            self.assertNotIn(torch.ops.aten.scaled_dot_product_attention.default, targets)
            exported = program.module()
            for batch in range(1, 9):
                patterns = (
                    ((1,), (0, 1), (1, 0))
                    if kind == "decode"
                    else ((7, 16), (1, 2), (0, 16), (512, 1))
                )
                for pattern in patterns:
                    counts = tuple(pattern[row % len(pattern)] for row in range(batch))
                    if not max(counts):
                        continue
                    with self.subTest(kind=kind, batch=batch, counts=counts):
                        width = max(counts)
                        # A prefill call can include one-token or inactive rows.
                        width = max(width, 2 if kind == "prefill" else 1)
                        starts = [
                            0 if count == 512 else 3 + row for row, count in enumerate(counts)
                        ]
                        ids = torch.randint(0, 128, (batch, width), device="cuda")
                        positions = torch.stack(
                            [torch.arange(start, start + width, device="cuda") for start in starts]
                        )
                        initial = tuple(
                            torch.randn((8, *t.shape[1:]), dtype=t.dtype, device=t.device)
                            for t in example[2]
                        )
                        lengths = torch.tensor(
                            [start + count for start, count in zip(starts, counts)],
                            dtype=torch.int32,
                        )
                        queries = torch.tensor(counts, dtype=torch.int32)
                        results = []
                        for call in (self.step, exported):
                            state = tuple(t.clone() for t in initial)
                            scores = call(
                                ids, positions, tuple(t[:batch] for t in state), lengths, queries
                            )
                            results.append((scores, state))
                            self.assertTrue(bool(torch.isfinite(scores).all()))
                            for row, count in enumerate(counts):
                                if count == 0:
                                    self.assertEqual(torch.count_nonzero(scores[row]).item(), 0)
                            for before, after in zip(initial, state):
                                self.assertTrue(torch.equal(before[batch:], after[batch:]))
                                for row, (start, count) in enumerate(zip(starts, counts)):
                                    self.assertTrue(
                                        torch.equal(before[row, :, :start], after[row, :, :start])
                                    )
                                    self.assertTrue(
                                        torch.equal(
                                            before[row, :, start + count :],
                                            after[row, :, start + count :],
                                        )
                                    )
                        self.assertTrue(torch.equal(results[0][0], results[1][0]))
                        self.assertTrue(
                            all(torch.equal(a, b) for a, b in zip(results[0][1], results[1][1]))
                        )
            for batch in (0, 9):
                inputs = (
                    torch.ones((batch, example[0].shape[1]), dtype=torch.int64, device="cuda"),
                    torch.zeros((batch, example[0].shape[1]), dtype=torch.int64, device="cuda"),
                    tuple(t.new_zeros((batch, *t.shape[1:])) for t in example[2]),
                    torch.ones(batch, dtype=torch.int32),
                    torch.ones(batch, dtype=torch.int32),
                )
                with self.assertRaises((RuntimeError, AssertionError)):
                    exported(*inputs)


if __name__ == "__main__":
    unittest.main()
