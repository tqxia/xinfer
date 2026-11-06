"""Checkpoint loading and model configuration contracts without CUDA."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from compiler.common.artifacts import MODEL_MANIFEST_NAME
from compiler.qwen35 import model


class ModelLoading(unittest.TestCase):
    def test_loader_reads_weights_without_manifest_verification(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "config.json").write_text(
                json.dumps(dict(text_config=dict(tie_word_embeddings=True)))
            )
            read_text = Path.read_text

            def read_without_manifest(path, *arguments, **options):
                self.assertNotEqual(path.name, MODEL_MANIFEST_NAME)
                return read_text(path, *arguments, **options)

            for manifest in (None, "invalid manifest"):
                with self.subTest(manifest=manifest):
                    if manifest is not None:
                        (output / MODEL_MANIFEST_NAME).write_text(manifest)
                    with (
                        patch.object(Path, "read_text", read_without_manifest),
                        patch.object(model, "Qwen3_5ForCausalLM") as constructor,
                        patch.object(
                            model, "load_file", side_effect=RuntimeError("reached weight loading")
                        ) as weights,
                    ):
                        with self.assertRaisesRegex(RuntimeError, "reached weight loading"):
                            model.load_model(output)
                        constructor.assert_called_once()
                        weights.assert_called_once_with(output / "text.safetensors")

    def test_configuration_requires_tied_embeddings_and_default_rope(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(dict(text_config={})))
            for tied, rope in ((False, "default"), (True, "linear")):
                with (
                    self.subTest(tied=tied, rope=rope),
                    patch.object(
                        model,
                        "Qwen3_5TextConfig",
                        return_value=SimpleNamespace(
                            tie_word_embeddings=tied, rope_parameters=dict(rope_type=rope)
                        ),
                    ),
                ):
                    with self.assertRaisesRegex(ValueError, "tied embeddings and default RoPE"):
                        model._text_config(path)


if __name__ == "__main__":
    unittest.main()
