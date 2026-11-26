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
                    patch.object(export, "configure"),
                    patch.object(export, "load_model"),
                    patch.object(export, "install_delta"),
                    patch.object(export, "load_attention_ops") as load_attn,
                    patch.object(export, "load_deltanet_ops") as load_delta,
                    patch.object(export, "attention_library", return_value=root / "attention.so"),
                    patch.object(export, "deltanet_library", return_value=root / "deltanet.so"),
                    patch.object(export, "attention_c_shims", return_value={"attn": ["shim"]}),
                    patch.object(export, "deltanet_c_shims", return_value={"delta": ["shim"]}),
                    patch.object(export, "compile_shared") as compile_pair,
                    redirect_stdout(io.StringIO()),
                ):
                    export.main(argv)
                load_attn.assert_called_once_with()
                load_delta.assert_called_once_with()
                compile_pair.assert_called_once()
                options = {"emulate_precision_casts": True}
                if optimize:
                    options.update(combo_kernels=True, benchmark_combo_kernel=True)
                self.assertEqual(compile_pair.call_args.kwargs["inductor_configs"], options)
                self.assertEqual(
                    compile_pair.call_args.kwargs["native_libraries"],
                    [root / "attention.so", root / "deltanet.so"],
                )
                self.assertEqual(
                    compile_pair.call_args.kwargs["custom_op_shims"],
                    {"attn": ["shim"], "delta": ["shim"]},
                )
                self.assertEqual(compile_pair.call_args.args[0], root / "bundle")
                self.assertIs(
                    compile_pair.call_args.kwargs["metadata"].func, export.bundle_metadata
                )
                self.assertEqual(compile_pair.call_args.kwargs["metadata"].args[-1], "qwen35")
                self.assertEqual(compile_pair.call_args.kwargs["identity"]["model"], "qwen3.5")
                self.assertIs(
                    compile_pair.call_args.args[1].keywords["step_exporter"], export.export_step
                )

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
                    patch.object(export, "configure"),
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
        export.configure()
        export.load_attention_ops()
        export.load_deltanet_ops()
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
        # Match checkpoint-native FP32 DeltaNet decay/norm parameters and RoPE frequencies.
        for layer in cls.model.model.layers:
            if hasattr(layer, "linear_attn"):
                layer.linear_attn.A_log = torch.nn.Parameter(
                    layer.linear_attn.A_log.float(), requires_grad=False
                )
                layer.linear_attn.norm.weight = torch.nn.Parameter(
                    layer.linear_attn.norm.weight.float(), requires_grad=False
                )
        cls.model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config).to("cuda")
        export.install_delta(cls.model)

    @classmethod
    def tearDownClass(cls):
        del cls.model

    @torch.inference_mode()
    def test_production_export_uses_native_deltanet_attention_and_inplace_state(self):
        for kind in ("prefill", "decode"):
            with self.subTest(kind=kind):
                exported, packing = export.export_step(self.model, kind, capacity=512)
                self.assertGreater(packing["saved_linears"], 0)
                self.assertFalse(exported.graph_signature.buffers_to_mutate)
                self.assertEqual(
                    len(exported.run_decompositions({}).graph_signature.user_inputs_to_mutate),
                    2 * self.model.config.num_hidden_layers,
                )
                targets = [
                    node.target for node in exported.graph.nodes if node.op == "call_function"
                ]
                attn_layers = self.model.config.layer_types.count("full_attention")
                delta_layers = self.model.config.layer_types.count("linear_attention")
                self.assertEqual(
                    targets.count(torch.ops.xinfer.batched_flash_attention.default), attn_layers
                )
                self.assertEqual(
                    targets.count(torch.ops.xinfer.append_kv_batched.default), attn_layers
                )
                self.assertEqual(
                    targets.count(torch.ops.xinfer.batched_deltanet.default), delta_layers
                )
                self.assertNotIn(torch.ops.higher_order.cond, targets)
                self.assertNotIn(torch.ops.aten.index_copy.default, targets)
                self.assertNotIn(torch.ops.aten.scaled_dot_product_attention.default, targets)
                if kind == "decode":
                    self.assertEqual(
                        targets.count(torch.ops.xinfer.projection_gemv.default),
                        4 * self.model.config.num_hidden_layers,
                    )
                else:
                    self.assertNotIn(torch.ops.xinfer.projection_gemv.default, targets)
                executable = exported.module()
                sample = exported.example_inputs[0]
                for batch in (1, 8):

                    def resize(tensor):
                        return tensor[:1].expand(batch, *tensor.shape[1:]).clone()

                    inputs = tuple(
                        tuple(resize(t) for t in value)
                        if isinstance(value, (list, tuple))
                        else resize(value)
                        for value in sample
                    )
                    pointers = [t.data_ptr() for t in inputs[2]]
                    logits = executable(*inputs)
                    self.assertIsInstance(logits, torch.Tensor)
                    self.assertEqual(logits.shape, (batch, self.model.config.vocab_size))
                    self.assertEqual([t.data_ptr() for t in inputs[2]], pointers)


if __name__ == "__main__":
    unittest.main()
