"""Validate compiled capacity from Qwen3.5 text configuration before GPU loading."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from compiler.qwen35 import export


class ExportContextLength(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
