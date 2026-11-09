"""CPU export contracts without a checkpoint download or CUDA allocations."""

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
                    patch.object(export, "compile_shared") as compile_pair,
                    redirect_stdout(io.StringIO()),
                ):
                    export.main()
                load.assert_called_once_with(root)
                self.assertEqual(
                    compile_pair.call_args.kwargs["identity"],
                    dict(model="qwen3.6", checkpoint=checkpoint),
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


if __name__ == "__main__":
    unittest.main()
