"""Export capacity checks and optimized mixed-layer graph integration tests."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3_5ForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextConfig,
    Qwen3_5TextRotaryEmbedding,
)

from compiler.qwen35 import export


class ExportContextLength(unittest.TestCase):
    def test_cli_preserves_default_and_optional_compilation_settings(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps(dict(max_position_embeddings=262144)))
            (root / "model_manifest.json").write_text(
                json.dumps(dict(model="fixture", revision="fixed"))
            )
            for optimize in (False, True):
                argv = ["--model-dir", str(root), "--output", str(root / "bundle")]
                if optimize:
                    argv.append("--enable-inductor-optimization")
                with (
                    self.subTest(optimize=optimize),
                    patch.object(export, "configure_text"),
                    patch.object(export, "load_model"),
                    patch.object(export, "install_delta"),
                    patch.object(export, "load_attention_ops"),
                    patch.object(export, "attention_library", return_value=root / "attention.so"),
                    patch.object(export, "attention_c_shims", return_value={}),
                    patch.object(export, "compile_shared") as compile_pair,
                    redirect_stdout(io.StringIO()),
                ):
                    export.main(argv)
                compile_pair.assert_called_once()
                options = {"emulate_precision_casts": True}
                if optimize:
                    options.update(combo_kernels=True, benchmark_combo_kernel=True)
                self.assertEqual(compile_pair.call_args.kwargs["inductor_configs"], options)
                self.assertEqual(compile_pair.call_args.args[0], root / "bundle")

    def test_workflow_errors_are_not_converted_to_cli_usage_errors(self):
        with (
            patch.object(export, "export_model", side_effect=ValueError("graph export failed")),
            self.assertRaisesRegex(ValueError, "graph export failed"),
        ):
            export.main([])

    def test_context_argument_is_checked_before_model_loading(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(
                json.dumps(dict(text_config=dict(max_position_embeddings=262144)))
            )
            argv = ["export", "--model-dir", str(root), "--output", str(root / "bundle")]
            for capacity in (511, 262145):
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
            for capacity in (None, 1024, 262144):
                arguments = [] if capacity is None else ["--max-context-length", str(capacity)]
                output = io.StringIO()
                with (
                    self.subTest(capacity=capacity),
                    patch("sys.argv", argv + arguments),
                    patch.object(export, "configure_text"),
                    patch.object(
                        export, "load_model", side_effect=RuntimeError("GPU load reached")
                    ) as load,
                    redirect_stdout(output),
                    self.assertRaisesRegex(RuntimeError, "GPU load reached"),
                ):
                    export.main()
                load.assert_called_once_with(root)
                event = json.loads(output.getvalue())
                self.assertEqual(event["cache_length"], 262144 if capacity is None else capacity)
                self.assertEqual(event["model_max_position_embeddings"], 262144)
                self.assertEqual(event["max_prefill_tokens"], 512)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ExportGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        export.configure_text()
        torch.manual_seed(35)
        config = Qwen3_5TextConfig(
            vocab_size=128,
            hidden_size=1024,
            intermediate_size=3584,
            num_hidden_layers=3,
            layer_types=["linear_attention", "full_attention", "linear_attention"],
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=256,
            linear_num_key_heads=16,
            linear_num_value_heads=16,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            max_position_embeddings=1024,
            tie_word_embeddings=True,
        )
        config._attn_implementation = "sdpa"
        cls.model = Qwen3_5ForCausalLM(config).eval().requires_grad_(False)
        cls.model.to(device="cuda", dtype=torch.bfloat16)
        # Match the loader's FP32 rotary frequencies around BF16 model weights.
        cls.model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config).to("cuda")
        export.install_delta(cls.model)

    @classmethod
    def tearDownClass(cls):
        del cls.model

    def assert_delta_dispatch(self, exported, kind):
        counts = dict(cond=0, short=0, long=0)

        def visit(module):
            # Count call sites, even when multiple layers share a branch module.
            for node in module.graph.nodes:
                if node.target == torch.ops.higher_order.cond:
                    predicate = node.args[0]
                    self.assertIsInstance(predicate, torch.fx.Node)
                    value = predicate.meta.get("val")
                    self.assertIsInstance(value, torch.SymBool)
                    self.assertNotIn(str(value), ("True", "False"))
                    counts["cond"] += 1
                    for branch in node.args[1:3]:
                        visit(module.get_submodule(branch.target))
                else:
                    for branch, operator in export.DELTA_OPERATORS.items():
                        counts[branch] += str(node.target) == operator

        visit(exported.graph_module)
        layers = self.model.config.layer_types.count("linear_attention")
        conditions = layers if kind == "prefill" else 0
        self.assertEqual(
            sum(n.target == torch.ops.higher_order.cond for n in exported.graph.nodes), conditions
        )
        self.assertEqual(counts, dict(cond=conditions, short=layers, long=conditions))

    @torch.inference_mode()
    def test_production_export_preserves_dispatch_attention_and_cache_updates(self):
        for kind in ("prefill", "decode"):
            with self.subTest(kind=kind):
                exported, packing = export.export_step(self.model, kind, capacity=512)
                self.assertGreater(packing["saved_linears"], 0)
                self.assertFalse(exported.graph_signature.buffers_to_mutate)
                targets = [
                    node.target for node in exported.graph.nodes if node.op == "call_function"
                ]
                layers = self.model.config.layer_types.count("full_attention")
                self.assertEqual(
                    targets.count(torch.ops.xinfer.batched_flash_attention.default), layers
                )
                self.assertEqual(targets.count(torch.ops.xinfer.append_kv_batched.default), layers)
                self.assertNotIn(torch.ops.aten.index_copy.default, targets)
                self.assertNotIn(torch.ops.aten.scaled_dot_product_attention.default, targets)
                if kind == "decode":
                    self.assertNotIn(torch.ops.xinfer.projection_gemv.default, targets)
                self.assert_delta_dispatch(exported, kind)
                executable = exported.module()
                sample = exported.example_inputs[0]
                for batch in (1, 8):

                    def resize(tensor):
                        return tensor[:1].expand(batch, *tensor.shape[1:]).clone()

                    inputs = tuple(
                        [resize(t) for t in value] if isinstance(value, list) else resize(value)
                        for value in sample
                    )
                    logits, *state = executable(*inputs)
                    self.assertEqual(logits.shape, (batch, self.model.config.vocab_size))
                    self.assertTrue(all(t.shape[0] == batch for t in state))


if __name__ == "__main__":
    unittest.main()
