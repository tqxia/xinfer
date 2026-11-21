"""Qwen3.8 defaults, build selection and shared export preflight."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from compiler.qwen35 import export as shared
from compiler.qwen38 import export
from compiler.qwen38.config import MODEL_ID


class ExportFrontend(unittest.TestCase):
    def test_defaults_select_qwen38_checkpoint_output_and_build(self):
        with patch.object(shared, "export_model") as compile_model:
            export.main([])
        compile_model.assert_called_once_with(
            model_dir=Path("work/qwen38/27b-fp8/text"),
            output=Path("artifacts/qwen38/27b-fp8/text"),
            max_context_length=None,
            enable_inductor_optimization=False,
            build_model="qwen38",
            model_loader=export.load_model,
        )

    def test_overrides_reach_shared_compiler_and_select_qwen38_attention(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = dict(model=MODEL_ID, revision="fixture")
            (root / "config.json").write_text(
                json.dumps(dict(text_config=dict(max_position_embeddings=262144)))
            )
            (root / "model_manifest.json").write_text(json.dumps(checkpoint))
            for optimize in (False, True):
                argv = [
                    "--model-dir",
                    str(root),
                    "--output",
                    str(root / "bundle"),
                    "--max-context-length",
                    "1024",
                ]
                if optimize:
                    argv.append("--enable-inductor-optimization")
                with (
                    self.subTest(optimize=optimize),
                    patch.object(shared, "configure"),
                    patch.object(export, "load_model") as load,
                    patch.object(shared, "load_model") as fallback,
                    patch.object(shared, "install_delta"),
                    patch.object(shared, "load_attention_ops") as attention,
                    patch.object(shared, "attention_library", return_value=root / "attention.so"),
                    patch.object(shared, "attention_c_shims", return_value={}),
                    patch.object(shared, "compile_shared") as compile_pair,
                    redirect_stdout(io.StringIO()),
                ):
                    export.main(argv)
                load.assert_called_once_with(root)
                fallback.assert_not_called()
                attention.assert_called_once_with(model="qwen38")
                compile_pair.assert_called_once()
                arguments, options = compile_pair.call_args
                self.assertEqual(arguments[0], root / "bundle")
                self.assertIs(arguments[1].func, shared.export_entry)
                self.assertEqual(arguments[1].keywords["capacity"], 1024)
                self.assertEqual(arguments[1].keywords["checkpoint"], checkpoint)
                self.assertEqual(options["identity"], dict(model="qwen3.5", checkpoint=checkpoint))
                expected = {"emulate_precision_casts": True}
                if optimize:
                    expected.update(combo_kernels=True, benchmark_combo_kernel=True)
                self.assertEqual(options["inductor_configs"], expected)
                self.assertEqual(options["native_libraries"], [root / "attention.so"])

    def test_invalid_capacity_is_rejected_before_gpu_loading(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps(dict(max_position_embeddings=262144)))
            with (
                patch.object(export, "load_model") as load,
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                export.main(
                    [
                        "--model-dir",
                        str(root),
                        "--output",
                        str(root / "bundle"),
                        "--max-context-length",
                        "511",
                    ]
                )
            self.assertEqual(error.exception.code, 2)
            load.assert_not_called()
            self.assertFalse((root / "bundle").exists())

    def test_compilation_failure_is_not_reported_as_a_usage_error(self):
        with (
            patch.object(shared, "export_model", side_effect=ValueError("graph export failed")),
            self.assertRaisesRegex(ValueError, "graph export failed"),
        ):
            export.main([])


if __name__ == "__main__":
    unittest.main()
