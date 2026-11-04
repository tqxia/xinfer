"""Pinned text extraction and checkpoint rejection without network or CUDA."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from compiler.common.artifacts import sha256
from compiler.qwen35 import model
from compiler.qwen35.config import MODEL_ID, REVISIONS, TEXT_DIR


class PreparedCheckpoint(unittest.TestCase):
    def source(self, directory):
        config = dict(text_config=dict(tie_word_embeddings=True))
        (directory / "config.json").write_text(json.dumps(config))
        for name in model.MODEL_FILES[1:]:
            (directory / name).write_text("fixture")
        names = (
            "model.language_model.embed_tokens.weight",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        )
        values = (torch.arange(4).reshape(2, 2).bfloat16(), torch.ones(2, 2).bfloat16())
        shards = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
        for name, value, shard in zip(names, values, shards):
            save_file({name: value}, directory / shard)
        (directory / "model.safetensors.index.json").write_text(
            json.dumps(
                dict(
                    weight_map=dict(zip(names, shards))
                    | {"model.visual.weight": "vision.safetensors"}
                )
            )
        )
        info = SimpleNamespace(
            siblings=[
                SimpleNamespace(
                    rfilename=name, lfs=SimpleNamespace(sha256=sha256(directory / name))
                )
                for name in shards
            ]
        )
        return info, dict(
            zip(
                ("model.embed_tokens.weight", "model.layers.0.linear_attn.in_proj_qkv.weight"),
                values,
            )
        )

    def test_extracts_both_pinned_checkpoints_and_only_text_shards(self):
        for model_id, revision in REVISIONS.items():
            with self.subTest(model=model_id), TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, output = root / "source", root / "text"
                source.mkdir()
                info, expected = self.source(source)
                api = SimpleNamespace(model_info=lambda *args, **kwargs: info)
                downloads = []

                def download(selected, name, **options):
                    self.assertEqual(selected, model_id)
                    self.assertEqual(options, dict(revision=revision, cache_dir="work/qwen35/hub"))
                    downloads.append(name)
                    return source / name

                with (
                    patch("huggingface_hub.HfApi", return_value=api),
                    patch("huggingface_hub.hf_hub_download", side_effect=download),
                    redirect_stdout(io.StringIO()),
                ):
                    model.prepare(output, model_id)
                self.assertNotIn("vision.safetensors", downloads)
                actual = load_file(output / "text.safetensors")
                self.assertEqual(set(actual), set(expected))
                for name in expected:
                    self.assertTrue(torch.equal(actual[name], expected[name]))
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertEqual(manifest["model"], model_id)
                self.assertEqual(manifest["revision"], revision)
                self.assertEqual(manifest["text_parameters"], len(expected))
                self.assertEqual(
                    set(manifest["source_shards"]),
                    set(downloads) - set(model.MODEL_FILES) - {"model.safetensors.index.json"},
                )
                self.assertEqual(set(manifest["files"]), {*model.MODEL_FILES, "text.safetensors"})
                for name, digest in manifest["files"].items():
                    self.assertEqual(sha256(output / name), digest)

    def test_rejects_hub_checksum_mismatch_before_reading_weights(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            info, _ = self.source(root)
            info.siblings[0].lfs.sha256 = "0" * 64
            with (
                patch(
                    "huggingface_hub.HfApi",
                    return_value=SimpleNamespace(model_info=lambda *a, **k: info),
                ),
                patch(
                    "huggingface_hub.hf_hub_download",
                    side_effect=lambda selected, name, **k: root / name,
                ),
                patch.object(model, "safe_open") as open_weights,
            ):
                with self.assertRaisesRegex(ValueError, "Hub checksum mismatch"):
                    model.prepare(root / "output")
                open_weights.assert_not_called()

    def test_rejects_unknown_model_without_creating_output(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "text"
            with self.assertRaisesRegex(ValueError, "Unsupported checkpoint"):
                model.prepare(output, "Qwen/unknown")
            self.assertFalse(output.exists())

    def test_preparation_requires_new_output_directory(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            retained = output / "retain.txt"
            retained.write_text("keep")
            with self.assertRaises(FileExistsError):
                model.prepare(output)
            self.assertEqual(retained.read_text(), "keep")

    def test_cli_preserves_default_and_selects_distinct_model_directories(self):
        for args, expected_model, expected_directory in (
            ([], MODEL_ID, TEXT_DIR),
            (["--model", "Qwen/Qwen3.5-4B"], "Qwen/Qwen3.5-4B", Path("work/qwen35/4b/text")),
            (
                ["--model", "Qwen/Qwen3.5-4B", "--output", "custom/text"],
                "Qwen/Qwen3.5-4B",
                Path("custom/text"),
            ),
        ):
            with self.subTest(args=args), patch.object(model, "prepare") as prepare:
                model.main(args)
                prepare.assert_called_once_with(expected_directory, expected_model)

    def test_loader_rejects_unknown_revision_and_incomplete_manifest_before_weights(self):
        valid = dict(
            model=MODEL_ID,
            revision=REVISIONS[MODEL_ID],
            files={name: "0" * 64 for name in (*model.MODEL_FILES, "text.safetensors")},
        )
        invalid = (
            (dict(valid, model="Qwen/unknown"), "pinned revision"),
            (dict(valid, revision="other"), "pinned revision"),
            (dict(valid, files={}), "file manifest"),
            (dict(valid, files=valid["files"] | {"../escape": "0" * 64}), "file manifest"),
        )
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            for manifest, message in invalid:
                (output / "manifest.json").write_text(json.dumps(manifest))
                with self.subTest(manifest=manifest), patch.object(model, "load_file") as weights:
                    with self.assertRaisesRegex(ValueError, message):
                        model.load_model(output)
                    weights.assert_not_called()

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
