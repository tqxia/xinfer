"""Shared export preflight preserves CLI errors and avoids GPU model work."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from compiler.common.export import ExportInputError, prepare_export


class ExportPreflight(unittest.TestCase):
    def test_invalid_destinations_are_rejected_before_reading_the_model(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            populated = root / "populated"
            populated.mkdir()
            saved = populated / "keep.txt"
            saved.write_text("preserve")
            link = root / "link"
            link.symlink_to(populated, target_is_directory=True)
            for output in (populated, saved, link):
                with (
                    self.subTest(output=output),
                    self.assertRaisesRegex(
                        ExportInputError, "Output must be a new or empty directory"
                    ),
                ):
                    prepare_export(root / "missing-model", output, None, 512)
            self.assertEqual(saved.read_text(), "preserve")
            self.assertTrue(link.is_symlink())

    def test_preflight_accepts_new_and_empty_destinations_without_writing(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps(dict(max_position_embeddings=40960)))
            empty = root / "empty"
            empty.mkdir()
            new = root / "new"
            for output in (empty, new):
                for override in (None, 1024):
                    log = io.StringIO()
                    with self.subTest(output=output, override=override), redirect_stdout(log):
                        capacity = prepare_export(root, output, override, 512)
                    self.assertEqual(capacity, 40960 if override is None else override)
                    self.assertEqual(
                        json.loads(log.getvalue()),
                        dict(
                            event="context",
                            cache_length=capacity,
                            model_max_position_embeddings=40960,
                            max_prefill_tokens=512,
                        ),
                    )
            self.assertEqual(list(empty.iterdir()), [])
            self.assertFalse(new.exists())

    def test_config_errors_keep_their_cli_or_filesystem_classification(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for config in ("not json", "{}", '{"max_position_embeddings": 0}'):
                (root / "config.json").write_text(config)
                with self.subTest(config=config), self.assertRaises(ExportInputError):
                    prepare_export(root, root / "output", None, 512)
            (root / "config.json").unlink()
            with self.assertRaises(FileNotFoundError):
                prepare_export(root, root / "output", None, 512)


if __name__ == "__main__":
    unittest.main()
