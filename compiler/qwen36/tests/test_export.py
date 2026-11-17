"""CPU workflow contracts and CUDA FP8 hybrid-graph integration tests."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from compiler.qwen36 import export


class TinyStep(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(128).to(torch.float8_e4m3fn), requires_grad=False
        )

    def forward(self, ids, positions, state, extent):
        value = (
            ids.float().sum(1)
            + positions.float().sum()
            + extent.shape[0]
            + self.weight.float().sum()
        )
        return (
            value[:, None].bfloat16(),
            state[0] + ids.sum().bfloat16(),
            state[1] + positions.sum(),
        )


class ExportContract(unittest.TestCase):
    def test_context_is_resolved_before_gpu_loading_and_no_prefix_flag_is_exposed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(
                json.dumps(dict(text_config=dict(max_position_embeddings=262144)))
            )
            argv = ["export", "--model-dir", str(root), "--output", str(root / "bundle")]
            for arguments in (
                ["--max-context-length", "511"],
                ["--max-context-length", "262145"],
                ["--layers", "4"],
            ):
                with (
                    self.subTest(arguments=arguments),
                    patch("sys.argv", argv + arguments),
                    patch.object(export, "load_model") as load,
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    export.main()
                self.assertEqual(error.exception.code, 2)
                load.assert_not_called()
                self.assertFalse((root / "bundle").exists())
            for capacity in (None, 1024, 262144):
                output = io.StringIO()
                arguments = [] if capacity is None else ["--max-context-length", str(capacity)]
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
                self.assertEqual(event["cache_length"], capacity or 262144)
                self.assertEqual(event["model_max_position_embeddings"], 262144)
                self.assertEqual(event["max_prefill_tokens"], 512)

    def test_existing_bundle_is_rejected_before_model_loading(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "bundle"
            output.mkdir()
            (output / "retained").write_text("keep")
            with (
                patch("sys.argv", ["export", "--output", str(output)]),
                patch.object(export, "load_model") as load,
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                export.main()
            load.assert_not_called()
            self.assertEqual((output / "retained").read_text(), "keep")

    def test_combo_is_opt_in_and_shared_identity_does_not_apply_bf16_packing(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(
                json.dumps(dict(text_config=dict(max_position_embeddings=262144)))
            )
            checkpoint = dict(model="Qwen/Qwen3.6-35B-A3B-FP8", revision="pinned")
            (root / "model_manifest.json").write_text(json.dumps(checkpoint))
            for optimized in (False, True):
                argv = ["export", "--model-dir", str(root), "--output", str(root / "bundle")]
                if optimized:
                    argv.append("--enable-inductor-optimization")
                with (
                    self.subTest(optimized=optimized),
                    patch("sys.argv", argv),
                    patch.object(export, "configure"),
                    patch.object(export, "load_model") as load,
                    patch.object(export, "check_model"),
                    patch.object(export, "install_fp8", return_value=dict(linear=250, experts=40)),
                    patch.object(export, "install_delta"),
                    patch.object(export, "load_attention_ops"),
                    patch.object(export, "attention_library", return_value=root / "attention.so"),
                    patch.object(export, "attention_c_shims", return_value={"flash": ["shim"]}),
                    patch.object(export, "compile_shared") as compile_pair,
                    redirect_stdout(io.StringIO()),
                ):
                    export.main()
                load.assert_called_once_with(root)
                self.assertEqual(
                    compile_pair.call_args.kwargs["identity"],
                    dict(model="qwen3.6", checkpoint=checkpoint),
                )
                self.assertEqual(
                    compile_pair.call_args.kwargs["native_libraries"], [root / "attention.so"]
                )
                self.assertEqual(
                    compile_pair.call_args.kwargs["custom_op_shims"], {"flash": ["shim"]}
                )
                options = compile_pair.call_args.kwargs["inductor_configs"]
                self.assertEqual(
                    options,
                    dict(
                        emulate_precision_casts=True,
                        **(
                            dict(combo_kernels=True, benchmark_combo_kernel=True)
                            if optimized
                            else {}
                        ),
                    ),
                )
            self.assertEqual(export.model_specific()["linear_packing"], dict(enabled=False))

    def test_workflow_errors_are_not_converted_to_usage_errors(self):
        with (
            patch.object(export, "export_model", side_effect=ValueError("graph export failed")),
            self.assertRaisesRegex(ValueError, "graph export failed"),
        ):
            export.main([])

    def test_entry_metadata_and_build_reporting_stay_separate(self):
        from tools.tests.test_qwen36_packages import metadata

        expected = metadata()
        model = SimpleNamespace(
            config=SimpleNamespace(),
            checkpoint_storage={"source_bytes": 123},
            parameters=lambda: iter([torch.ones(2, dtype=torch.bfloat16)]),
            model=SimpleNamespace(rotary_emb=SimpleNamespace(inv_freq=torch.ones(2))),
        )
        graph = SimpleNamespace(
            graph=SimpleNamespace(nodes=[object()]),
            range_constraints={"tokens": "2..512"},
            graph_module=SimpleNamespace(print_readable=lambda **kwargs: "graph fixture"),
        )
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            compiled = output / "decode.pt2"
            compiled.write_bytes(b"package")
            with (
                patch.object(export, "export_step", return_value=graph) as trace,
                patch.object(export, "entry_metadata", return_value=expected),
                patch.object(export, "report_export") as report,
                redirect_stdout(io.StringIO()),
            ):
                actual, factory = export.export_entry(
                    "decode",
                    output,
                    model=model,
                    checkpoint={},
                    capacity=1024,
                    adapters=dict(linear=250, experts=40),
                    enable_inductor_optimization=False,
                )
                self.assertIs(actual, graph)
                self.assertEqual(factory(compiled), expected)
            trace.assert_called_once_with(model, "decode", 1024)
            self.assertEqual(report.call_args.kwargs["extra_dependencies"], ("kernels",))
            self.assertEqual(report.call_args.kwargs["package_bytes"], 7)
            self.assertEqual((output / "decode.graph.txt").read_text(), "graph fixture\n")
            self.assertNotIn("diagnostics", expected)
            self.assertFalse(report.call_args.kwargs["diagnostics"]["linear_packing"])

    def test_strict_cpu_fixture_preserves_fp8_and_independent_dynamic_extents(self):
        model = SimpleNamespace(
            config=SimpleNamespace(vocab_size=1024, max_position_embeddings=262144),
            model=SimpleNamespace(embed_tokens=torch.nn.Embedding(2, 2)),
        )

        def state(config, capacity):
            return [torch.zeros(2, dtype=torch.bfloat16), torch.zeros(2)]

        for capacity in (1024, 262144):
            for kind in ("prefill", "decode"):
                with (
                    self.subTest(capacity=capacity, kind=kind),
                    patch.object(export, "TextStep", TinyStep),
                    patch.object(export, "zero_state", side_effect=state) as allocate,
                ):
                    program = export.export_step(model, kind, capacity)
                allocate.assert_called_once_with(model.config, capacity)
                self.assertEqual(program.state_dict["weight"].dtype, torch.float8_e4m3fn)
                bounds = {
                    (int(value.lower), int(value.upper))
                    for value in program.range_constraints.values()
                }
                self.assertIn((2, capacity + 1), bounds)
                if kind == "prefill":
                    self.assertIn((2, 512), bounds)
                for length in (2, 512) if kind == "prefill" else (1,):
                    inputs = (
                        torch.ones(1, length, dtype=torch.int64),
                        torch.arange(length),
                        state(None, capacity),
                        torch.empty(capacity + 1, dtype=torch.uint8),
                    )
                    outputs = program.module()(*inputs)
                    self.assertEqual(outputs[0].shape, (1, 1))
                    self.assertEqual(outputs[0].dtype, torch.bfloat16)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ExportGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import FineGrainedFP8Config, Qwen3_5MoeForCausalLM
        from transformers.integrations.finegrained_fp8 import replace_with_fp8_linear
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeTextConfig,
            Qwen3_5MoeTextRotaryEmbedding,
        )

        export.configure()
        torch.manual_seed(36)
        config = Qwen3_5MoeTextConfig(
            vocab_size=128,
            hidden_size=128,
            num_hidden_layers=3,
            layer_types=["linear_attention", "full_attention", "linear_attention"],
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=256,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            num_experts=2,
            num_experts_per_tok=1,
            moe_intermediate_size=128,
            shared_expert_intermediate_size=128,
            max_position_embeddings=1024,
            tie_word_embeddings=False,
        )
        config._attn_implementation = "sdpa"
        with torch.device("meta"):
            model = Qwen3_5MoeForCausalLM(config)
            replace_with_fp8_linear(
                model,
                modules_to_not_convert=[
                    "lm_head",
                    "gate",
                    "shared_expert_gate",
                    "in_proj_a",
                    "in_proj_b",
                ],
                quantization_config=FineGrainedFP8Config(
                    activation_scheme="dynamic",
                    weight_block_size=(128, 128),
                    dequantize=False,
                ),
                pre_quantized=True,
            )
        for name, parameter in list(model.named_parameters()):
            if parameter.dtype == torch.float8_e4m3fn:
                value = (torch.randn(parameter.shape, device="cuda") * 0.02).to(parameter.dtype)
            elif name.endswith("weight_scale_inv") or name.endswith("proj_scale_inv"):
                value = torch.ones(parameter.shape, device="cuda", dtype=torch.bfloat16)
            else:
                value = torch.randn(parameter.shape, device="cuda", dtype=torch.bfloat16) * 0.02
            parent, _, field = name.rpartition(".")
            model.get_submodule(parent).register_parameter(
                field, torch.nn.Parameter(value, requires_grad=False)
            )
        model.model.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config).to("cuda")
        cls.model = model.eval().requires_grad_(False)
        export.install_fp8(cls.model)
        export.install_delta(cls.model)
        export.load_attention_ops(model="qwen36")

    @classmethod
    def tearDownClass(cls):
        del cls.model

    @torch.inference_mode()
    def test_production_export_preserves_fp8_dispatch_attention_and_cache_updates(self):
        from compiler.qwen35.tests.graphs import assert_hybrid_graph

        with export.memoize_triton_reads():
            for kind in ("prefill", "decode"):
                with self.subTest(kind=kind):
                    exported = export.export_step(self.model, kind, capacity=512)
                    targets = assert_hybrid_graph(
                        self,
                        exported,
                        kind,
                        self.model.config.layer_types,
                        export.DELTA_OPERATORS,
                    )
                    self.assertIn(torch.ops.xinfer.fp8_linear.default, targets)
                    self.assertIn(torch.ops.xinfer.fp8_grouped.default, targets)
                    self.assertFalse(
                        any(str(t) == "xinfer.projection_gemv.default" for t in targets)
                    )
                    self.assertEqual(
                        {p.dtype for _, p in exported.named_parameters()},
                        {torch.float8_e4m3fn, torch.bfloat16},
                    )
                    bounds = {
                        (int(r.lower), int(r.upper)) for r in exported.range_constraints.values()
                    }
                    self.assertIn((2, 513), bounds)
                    if kind == "prefill":
                        self.assertIn((2, 512), bounds)


if __name__ == "__main__":
    unittest.main()
