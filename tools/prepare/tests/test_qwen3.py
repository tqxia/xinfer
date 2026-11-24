"""Pinned Qwen3 download and retained checkpoint checks without network or CUDA."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from compiler.common.artifacts import MODEL_MANIFEST_NAME
from compiler.qwen3.config import DEFAULT_MODEL
from tools.prepare.qwen3 import MODEL_FILES, REVISIONS
from tools.prepare import common, qwen3

MODEL_ID = "Qwen/Qwen3-0.6B"


class PreparedCheckpoint(unittest.TestCase):
    def files(self):
        return {name: ("original HF bytes: " + name).encode() for name in MODEL_FILES}

    def descriptor(self, data):
        return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    def manifest(self, contents):
        return dict(
            model=MODEL_ID,
            revision=REVISIONS[MODEL_ID],
            status="complete",
            huggingface_hub="2.3.0",
            files={name: self.descriptor(data) for name, data in contents.items()},
        )

    def checkpoint(self, directory):
        contents = self.files()
        for name, data in contents.items():
            (directory / name).write_bytes(data)
        manifest = self.manifest(contents)
        (directory / MODEL_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest

    def test_download_publishes_original_files_and_existing_manifest_format(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, cache = root / "model", root / "hub-cache"
            contents = self.files()

            def copy(name, destination):
                self.assertFalse(output.exists())
                data = contents[name]
                destination.write_bytes(data)
                return self.descriptor(data)

            with (
                patch.object(common, "Hub") as hub,
                patch.object(qwen3.importlib.metadata, "version", return_value="2.3.0") as version,
            ):
                hub.return_value.copy.side_effect = copy
                manifest = qwen3.prepare(output, MODEL_ID, cache_dir=cache)
            hub.assert_called_once_with(MODEL_ID, REVISIONS[MODEL_ID], cache_dir=cache)
            self.assertEqual(
                [call.args[0] for call in hub.return_value.copy.call_args_list], list(MODEL_FILES)
            )
            version.assert_called_once_with("huggingface-hub")
            self.assertEqual(manifest, self.manifest(contents))
            self.assertEqual(
                set(path.name for path in output.iterdir()), {*MODEL_FILES, MODEL_MANIFEST_NAME}
            )
            for name, expected in contents.items():
                self.assertEqual((output / name).read_bytes(), expected)
            manifest_bytes = (output / "model_manifest.json").read_bytes()
            self.assertFalse((output / "manifest.json").exists())
            with redirect_stdout(io.StringIO()):
                self.assertEqual(qwen3.verify_model(output), manifest)
            self.assertEqual((output / MODEL_MANIFEST_NAME).read_bytes(), manifest_bytes)

    def test_unknown_model_is_rejected_before_network_or_directory_creation(self):
        for model_id in (None, [], {}, "Qwen/unknown"):
            with self.subTest(model=model_id), TemporaryDirectory() as temporary:
                output = Path(temporary) / "model"
                with patch.object(common, "Hub") as hub:
                    with self.assertRaisesRegex(ValueError, "Unsupported checkpoint"):
                        qwen3.prepare(output, model_id)
                    with self.assertRaisesRegex(ValueError, "Unsupported checkpoint"):
                        qwen3.default_model_dir(model_id)
                hub.assert_not_called()
                self.assertFalse(output.exists())
        self.assertEqual(qwen3.default_model_dir(MODEL_ID), DEFAULT_MODEL)
        self.assertEqual(qwen3.default_model_dir("Qwen/Qwen3-1.7B"), Path("work/models/qwen3-1.7b"))

    def test_sharded_checkpoint_downloads_and_verifies_all_referenced_shards(self):
        model_id = "Qwen/Qwen3-1.7B"
        shards = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
        index_bytes = json.dumps(
            {"weight_map": {"model.embed_tokens.weight": shards[0], "lm_head.weight": shards[1]}}
        ).encode()
        contents = {name: ("original HF bytes: " + name).encode() for name in qwen3.BASE_FILES}
        contents[qwen3.WEIGHT_INDEX] = index_bytes
        for shard in shards:
            contents[shard] = ("shard bytes: " + shard).encode()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, cache = root / "model", root / "hub-cache"

            def copy(name, destination):
                data = contents[name]
                destination.write_bytes(data)
                return self.descriptor(data)

            with (
                patch.object(common, "Hub") as hub,
                patch.object(qwen3.importlib.metadata, "version", return_value="2.3.0"),
            ):
                hub.return_value.files = dict.fromkeys(contents)
                hub.return_value.copy.side_effect = copy
                manifest = qwen3.prepare(output, model_id, cache_dir=cache)
            hub.assert_called_once_with(model_id, REVISIONS[model_id], cache_dir=cache)
            self.assertEqual(manifest["model"], model_id)
            self.assertEqual(manifest["revision"], REVISIONS[model_id])
            self.assertEqual(set(manifest["files"]), set(contents))
            with redirect_stdout(io.StringIO()):
                self.assertEqual(qwen3.verify_model(output), manifest)

            # Removing a shard from the manifest must fail verification against the index.
            bad_manifest = dict(
                manifest,
                files={k: v for k, v in manifest["files"].items() if k != shards[1]},
            )
            (output / MODEL_MANIFEST_NAME).write_text(json.dumps(bad_manifest))
            with redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "file manifest"):
                    qwen3.verify_model(output)

    def test_existing_output_is_preserved_without_network_access(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            retained = output / "retain.txt"
            retained.write_text("keep")
            with patch.object(common, "Hub") as hub:
                with self.assertRaises(FileExistsError):
                    qwen3.prepare(output, MODEL_ID)
            hub.assert_not_called()
            self.assertEqual(retained.read_text(), "keep")
            self.assertEqual(list(output.iterdir()), [retained])

    def test_failed_file_download_does_not_publish_an_incomplete_checkpoint(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "model"
            copied = []

            def copy(name, destination):
                if name == "model.safetensors":
                    raise ValueError("Hub checksum mismatch")
                data = self.files()[name]
                destination.write_bytes(data)
                copied.append(name)
                return self.descriptor(data)

            with patch.object(common, "Hub") as hub:
                hub.return_value.copy.side_effect = copy
                with self.assertRaisesRegex(ValueError, "Hub checksum mismatch"):
                    qwen3.prepare(output, MODEL_ID)
            self.assertTrue(copied)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_verification_rejects_unpinned_identity_and_invalid_inventories_before_hashing(self):
        contents = self.files()
        valid = self.manifest(contents)
        invalid = (
            ([], "must be an object"),
            (dict(valid, model="Qwen/unknown"), "pinned revision"),
            (dict(valid, model=[]), "pinned revision"),
            (
                {k: v for k, v in valid.items() if k != "model"} | {"repo_id": MODEL_ID},
                "pinned revision",
            ),
            (dict(valid, revision="main"), "pinned revision"),
            (dict(valid, files={}), "file manifest"),
            (dict(valid, files=valid["files"] | {"../escape": {}}), "file manifest"),
        )
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for manifest, message in invalid:
                with self.subTest(manifest=manifest):
                    (directory / MODEL_MANIFEST_NAME).write_text(json.dumps(manifest))
                    with patch.object(qwen3, "sha256") as hashing:
                        with self.assertRaisesRegex(ValueError, message):
                            qwen3.verify_model(directory)
                    hashing.assert_not_called()

    def test_verification_rejects_invalid_file_descriptors_before_hashing(self):
        for descriptor in (
            None,
            {"bytes": True, "sha256": "0" * 64},
            {"bytes": -1, "sha256": "0" * 64},
            {"bytes": 1, "sha256": "not-a-digest"},
            {"bytes": 1, "sha256": "0" * 64, "extra": 1},
        ):
            with self.subTest(descriptor=descriptor), TemporaryDirectory() as temporary:
                directory = Path(temporary)
                manifest = self.manifest(self.files())
                manifest["files"]["config.json"] = descriptor
                (directory / MODEL_MANIFEST_NAME).write_text(json.dumps(manifest))
                with patch.object(qwen3, "sha256") as hashing:
                    with self.assertRaisesRegex(ValueError, "Invalid checkpoint file metadata"):
                        qwen3.verify_model(directory)
                hashing.assert_not_called()

    def test_missing_or_modified_retained_files_are_rejected(self):
        for mutation in ("missing", "size", "digest"):
            with self.subTest(mutation=mutation), TemporaryDirectory() as temporary:
                directory = Path(temporary)
                self.checkpoint(directory)
                path = directory / "model.safetensors"
                if mutation == "missing":
                    path.unlink()
                elif mutation == "size":
                    path.write_bytes(path.read_bytes() + b"x")
                else:
                    data = path.read_bytes()
                    path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
                with redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(ValueError, "missing|checksum mismatch"):
                        qwen3.verify_model(directory)


if __name__ == "__main__":
    unittest.main()
