"""Reject unsupported context capacities before loading Qwen3 GPU weights."""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from compiler.qwen3 import export


class ExportContextLength(unittest.TestCase):
    def test_context_argument_is_checked_before_model_loading(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps({"max_position_embeddings": 40960}))
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


class ExportBackendContract(unittest.TestCase):
    @staticmethod
    def config():
        return {
            "max_position_embeddings": 40960,
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
        }

    def test_mega_profile_preflight_runs_before_gpu_model_loading(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {"repo_id": "example/compatible-qwen3", "revision": "local-revision"}
            (root / "config.json").write_text(json.dumps(self.config()))
            (root / "model_manifest.json").write_text(json.dumps(manifest))
            arguments = [
                "--model-dir",
                str(root),
                "--output",
                str(root / "bundle"),
                "--decode-backend",
                "megaqwen",
                "--max-context-length",
                "1024",
            ]
            with (
                patch.object(
                    export, "load_model", side_effect=RuntimeError("GPU load reached")
                ) as load,
                patch.object(export, "load_megaqwen_ops") as library,
                redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, "GPU load reached"),
            ):
                export.main(arguments)
            load.assert_called_once_with(root)
            library.assert_called_once_with()
            for target, changed in (
                ("config", {**self.config(), "head_dim": 64}),
                ("config", {**self.config(), "rope_scaling": {"factor": 2}}),
            ):
                (root / "model_manifest.json").write_text(json.dumps(manifest))
                (root / "config.json").write_text(json.dumps(self.config()))
                path = root / ("model_manifest.json" if target == "manifest" else "config.json")
                path.write_text(json.dumps(changed))
                with (
                    self.subTest(target=target, changed=changed),
                    patch.object(export, "load_model") as load,
                    patch.object(export, "load_megaqwen_ops") as library,
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    export.main(arguments)
                self.assertEqual(error.exception.code, 2)
                load.assert_not_called()
                library.assert_not_called()
                self.assertFalse((root / "bundle").exists())

    def test_mega_export_accepts_alternate_checkpoint_provenance(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {"repo_id": "example/compatible-qwen3", "revision": "local-revision"}
            (root / "config.json").write_text(json.dumps(self.config()))
            (root / "model_manifest.json").write_text(json.dumps(manifest))
            with (
                patch.object(export, "load_model"),
                patch.object(export, "FlashStep"),
                patch.object(export, "MegaDecodeStep"),
                patch.object(export, "load_megaqwen_ops"),
                patch.object(export, "attention_library", return_value=root / "attention.so"),
                patch.object(export, "attention_c_shims", return_value={}),
                patch.object(export, "megaqwen_library", return_value=root / "megaqwen.so"),
                patch.object(export, "megaqwen_c_shims", return_value={}),
                patch.object(export, "compile_shared") as compile,
                redirect_stdout(io.StringIO()),
            ):
                export.main(
                    [
                        "--model-dir",
                        str(root),
                        "--output",
                        str(root / "bundle"),
                        "--decode-backend",
                        "megaqwen",
                        "--max-context-length",
                        "1024",
                    ]
                )
            compile.assert_called_once()
            self.assertEqual(
                compile.call_args.kwargs["identity"],
                {"model": "qwen3-0.6b", "revision": manifest["revision"]},
            )


if __name__ == "__main__":
    unittest.main()
