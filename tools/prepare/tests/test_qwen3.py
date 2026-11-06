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
from tools.prepare.qwen3 import MODEL_FILES, MODEL_ID, REVISIONS
from tools.prepare import common, qwen3


class PreparedCheckpoint(unittest.TestCase):
    def files(self):
        return {name: ("original HF bytes: " + name).encode() for name in MODEL_FILES}

    def descriptor(self, data):
        return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    def manifest(self, contents):
        return dict(
            repo_id=MODEL_ID,
            revision=REVISIONS[MODEL_ID],
            status="complete",
            huggingface_hub="0.30.2",
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
                patch.object(qwen3.importlib.metadata, "version", return_value="0.30.2") as version,
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

    def test_existing_output_is_preserved_without_network_access(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            retained = output / "retain.txt"
            retained.write_text("keep")
            with patch.object(common, "Hub") as hub:
                with self.assertRaises(FileExistsError):
                    qwen3.prepare(output)
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
                    qwen3.prepare(output)
            self.assertTrue(copied)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_verification_rejects_unpinned_identity_and_invalid_inventories_before_hashing(self):
        contents = self.files()
        valid = self.manifest(contents)
        invalid = (
            ([], "must be an object"),
            (dict(valid, repo_id="Qwen/unknown"), "pinned revision"),
            (dict(valid, repo_id=[]), "pinned revision"),
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
