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
        cls.step = export.FlashStep(model.to(device="cuda", dtype=torch.bfloat16))

    @classmethod
    def tearDownClass(cls):
        del cls.step

    @torch.inference_mode()
    def test_production_export_preserves_attention_and_cache_updates(self):
        for kind in ("prefill", "decode"):
            with self.subTest(kind=kind):
                inputs, shapes = export.example_inputs(self.step.model.config, kind, capacity=512)
                exported = export.export_step(self.step, inputs, shapes)
                self.assertFalse(exported.graph_signature.buffers_to_mutate)
                targets = [
                    node.target for node in exported.graph.nodes if node.op == "call_function"
                ]
                layers = self.step.model.config.num_hidden_layers
                self.assertEqual(targets.count(torch.ops.xinfer.flash_attention.default), layers)
                self.assertEqual(targets.count(torch.ops.aten.index_copy_.default), 2 * layers)
                self.assertNotIn(torch.ops.aten.index_copy.default, targets)
                self.assertNotIn(torch.ops.aten.scaled_dot_product_attention.default, targets)


if __name__ == "__main__":
    unittest.main()
