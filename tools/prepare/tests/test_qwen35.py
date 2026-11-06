"""Pinned text extraction and checkpoint rejection without network or CUDA."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from safetensors.torch import load_file, save_file

from compiler.common.artifacts import MODEL_MANIFEST_NAME, sha256
from compiler.qwen35.config import TEXT_DIR
from tools.prepare.qwen35 import MODEL_FILES, MODEL_ID, REVISIONS
from tools.prepare import cli, qwen35


class PreparedCheckpoint(unittest.TestCase):
    def source(self, directory, revision=REVISIONS[MODEL_ID], *, text=True):
        config = dict(text_config=dict(tie_word_embeddings=True))
        (directory / "config.json").write_text(json.dumps(config))
        for name in MODEL_FILES[1:]:
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
                    weight_map=(dict(zip(names, shards)) if text else {})
                    | {"model.visual.weight": "vision.safetensors"}
                )
            )
        )
        siblings = []
        for name in (*MODEL_FILES, "model.safetensors.index.json", *shards):
            data = (directory / name).read_bytes()
            blob_id = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            siblings.append(
                SimpleNamespace(
                    rfilename=name,
                    size=len(data),
                    blob_id=blob_id,
                    lfs=SimpleNamespace(sha256=sha256(directory / name))
                    if name in shards
                    else None,
                )
            )
        info = SimpleNamespace(sha=revision, siblings=siblings)
        return info, dict(
            zip(
                ("model.embed_tokens.weight", "model.layers.0.linear_attn.in_proj_qkv.weight"),
                values,
            )
        )

    def network(self, source, info, model_id=MODEL_ID, cache_dir=None):
        downloads = []
        api = Mock()
        api.model_info.return_value = info

        def download(repo_id, filename, **options):
            self.assertEqual(repo_id, model_id)
            self.assertEqual(options["revision"], REVISIONS[model_id])
            selected_cache = options.get("cache_dir")
            self.assertEqual(None if selected_cache is None else Path(selected_cache), cache_dir)
            downloads.append(filename)
            return source / filename

        self.enterContext(patch("huggingface_hub.HfApi", return_value=api))
        self.enterContext(patch("huggingface_hub.hf_hub_download", side_effect=download))
        return api, downloads

    def test_extracts_both_pinned_checkpoints_and_only_text_shards(self):
        for model_id, revision in REVISIONS.items():
            with self.subTest(model=model_id), TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, output = root / "source", root / "text"
                source.mkdir()
                info, expected = self.source(source, revision)
                cache_dir = root / "cache"
                api, downloads = self.network(source, info, model_id, cache_dir)
                manifest = qwen35.prepare(output, model_id, cache_dir=cache_dir)
                api.model_info.assert_called_once_with(
                    model_id, revision=revision, files_metadata=True
                )
                self.assertNotIn("vision.safetensors", downloads)
                self.assertTrue(set(MODEL_FILES) <= set(downloads))
                actual = load_file(output / "text.safetensors")
                self.assertEqual(set(actual), set(expected))
                for name in expected:
                    self.assertTrue(torch.equal(actual[name], expected[name]))
                self.assertTrue((output / "model_manifest.json").is_file())
                self.assertFalse((output / "manifest.json").exists())
                self.assertEqual(manifest, json.loads((output / MODEL_MANIFEST_NAME).read_text()))
                self.assertEqual(manifest["model"], model_id)
                self.assertEqual(manifest["revision"], revision)
                self.assertEqual(manifest["text_parameters"], len(expected))
                self.assertEqual(
                    set(manifest["source_shards"]),
                    set(downloads) - set(MODEL_FILES) - {"model.safetensors.index.json"},
                )
                self.assertEqual(set(manifest["files"]), {*MODEL_FILES, "text.safetensors"})
                for name, digest in manifest["files"].items():
                    self.assertEqual(sha256(output / name), digest)
                self.assertEqual(qwen35.verify_model(output), manifest)
                self.assertEqual(set(root.iterdir()), {source, output})

    def test_rejects_hub_checksum_mismatch_before_reading_weights(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source", root / "output"
            source.mkdir()
            info, _ = self.source(source)
            next(entry for entry in info.siblings if entry.lfs).lfs.sha256 = "0" * 64
            self.network(source, info)
            with patch("safetensors.safe_open") as open_weights:
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    qwen35.prepare(output)
            open_weights.assert_not_called()
            self.assertFalse(output.exists())
            self.assertEqual(set(root.iterdir()), {source})

    def test_rejects_unknown_model_without_creating_output_or_querying_hub(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "text"
            with patch("huggingface_hub.HfApi") as api:
                with self.assertRaisesRegex(ValueError, "Unsupported checkpoint"):
                    qwen35.prepare(output, "Qwen/unknown")
            api.assert_not_called()
            self.assertFalse(output.exists())

    def test_preparation_requires_new_output_directory(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            retained = output / "retain.txt"
            retained.write_text("keep")
            with patch("huggingface_hub.HfApi") as api:
                with self.assertRaises(FileExistsError):
                    qwen35.prepare(output)
            api.assert_not_called()
            self.assertEqual(retained.read_text(), "keep")

    def test_missing_text_weights_never_publish_a_checkpoint(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source", root / "output"
            source.mkdir()
            info, _ = self.source(source, text=False)
            self.network(source, info)
            with self.assertRaisesRegex(ValueError, "No text weights"):
                qwen35.prepare(output)
            self.assertFalse(output.exists())
            self.assertEqual(set(root.iterdir()), {source})

    def test_late_copy_failure_removes_the_unpublished_preparation(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source", root / "output"
            source.mkdir()
            info, _ = self.source(source)
            self.network(source, info)
            (source / "chat_template.jinja").unlink()
            with self.assertRaises(FileNotFoundError):
                qwen35.prepare(output)
            self.assertFalse(output.exists())
            self.assertEqual(set(root.iterdir()), {source})

    def test_cli_selects_distinct_default_model_directories_and_explicit_output(self):
        for arguments, expected_model, expected_directory in (
            (["--model", MODEL_ID], MODEL_ID, TEXT_DIR),
            (["--model", "Qwen/Qwen3.5-4B"], "Qwen/Qwen3.5-4B", Path("work/qwen35/4b/text")),
            (
                ["--model", "Qwen/Qwen3.5-4B", "--output", "custom/text"],
                "Qwen/Qwen3.5-4B",
                Path("custom/text"),
            ),
        ):
            with self.subTest(arguments=arguments), patch.object(qwen35, "prepare") as prepare:
                prepare.return_value = {}
                with patch("builtins.print"):
                    cli.main(["download", *arguments])
                prepare.assert_called_once_with(expected_directory, expected_model, cache_dir=None)

    def test_verifier_rejects_unknown_revision_and_incomplete_manifest(self):
        valid = dict(
            model=MODEL_ID,
            revision=REVISIONS[MODEL_ID],
            files={name: "0" * 64 for name in (*MODEL_FILES, "text.safetensors")},
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
                (output / MODEL_MANIFEST_NAME).write_text(json.dumps(manifest))
                with self.subTest(manifest=manifest), patch.object(qwen35, "sha256") as digest:
                    with self.assertRaisesRegex(ValueError, message):
                        qwen35.verify_model(output)
                    digest.assert_not_called()


if __name__ == "__main__":
    unittest.main()
