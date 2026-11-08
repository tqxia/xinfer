"""Select packaged MegaQwen only for eager validation and export shape handling."""

import gc
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
from unittest.mock import patch

from compiler.qwen3 import megaqwen
from tools.validation.common import megaqwen as packaged_megaqwen
from tools.validation import validate


class PackagedMegaQwenSelection(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.enterContext(patch.object(megaqwen, "_loaded", None))
        self.enterContext(patch.object(packaged_megaqwen, "_packaged_directory", None))
        self.addCleanup(self.cleanup_packaged_library)

    def cleanup_packaged_library(self):
        if packaged_megaqwen._packaged_directory is not None:
            packaged_megaqwen._packaged_directory.cleanup()

    def packages(
        self, decode=b"decode MegaQwen implementation", *, duplicate=False, name="packages"
    ):
        directory = self.root / name
        directory.mkdir()
        for kind in ("prefill", "decode"):
            with zipfile.ZipFile(directory / f"{kind}.pt2", "w") as archive:
                payload = b"unused prefill implementation" if kind == "prefill" else decode
                if payload is not None:
                    archive.writestr(
                        "fixture/data/aotinductor/model/" + megaqwen.MEGAQWEN_LIBRARY, payload
                    )
                if duplicate and kind == "decode":
                    archive.writestr(
                        "fixture/data/aotinductor/other/" + megaqwen.MEGAQWEN_LIBRARY, b"duplicate"
                    )
        return directory

    def test_decode_payload_is_selected_and_retained_for_eager_use(self):
        packages = self.packages()
        with patch.object(megaqwen, "load_megaqwen_ops") as load:
            library = packaged_megaqwen.load_packaged_megaqwen_ops(packages)
        load.assert_called_once_with(library)
        self.assertEqual(library.name, megaqwen.MEGAQWEN_LIBRARY)
        self.assertEqual(library.read_bytes(), b"decode MegaQwen implementation")
        self.assertNotEqual(library.parent, packages)
        gc.collect()
        self.assertTrue(library.is_file())

    def test_missing_or_ambiguous_decode_library_fails_before_loading(self):
        for payload, duplicate in ((None, False), (b"valid", True)):
            with self.subTest(payload=payload, duplicate=duplicate):
                packages = self.packages(payload, duplicate=duplicate, name=str(duplicate))
                with (
                    patch.object(megaqwen, "load_megaqwen_ops") as load,
                    self.assertRaisesRegex(ValueError, "missing its MegaQwen library"),
                ):
                    packaged_megaqwen.load_packaged_megaqwen_ops(packages)
                load.assert_not_called()

    def test_empty_payload_fails_before_loading(self):
        packages = self.packages(b"")
        with (
            patch.object(megaqwen, "load_megaqwen_ops") as load,
            self.assertRaisesRegex(ValueError, "Packaged MegaQwen library is empty"),
        ):
            packaged_megaqwen.load_packaged_megaqwen_ops(packages)
        load.assert_not_called()

    def test_stale_build_override_does_not_change_packaged_eager_selection(self):
        packages = self.packages()
        with (
            patch.dict(os.environ, XINFER_MEGAQWEN_LIBRARY="/missing/build/mega.so"),
            patch.object(megaqwen, "load_megaqwen_ops") as load,
        ):
            library = packaged_megaqwen.load_packaged_megaqwen_ops(packages)
        load.assert_called_once_with(library)
        self.assertEqual(library.read_bytes(), b"decode MegaQwen implementation")

    def test_existing_eager_selection_is_reused_without_package_comparison(self):
        library = self.root / megaqwen.MEGAQWEN_LIBRARY
        library.write_bytes(b"already selected eager implementation")
        with (
            patch.object(megaqwen, "_loaded", library),
            patch.object(megaqwen, "load_megaqwen_ops") as load,
        ):
            self.assertEqual(
                packaged_megaqwen.load_packaged_megaqwen_ops(self.root / "absent"), library
            )
        load.assert_not_called()

    def test_failed_load_releases_the_extracted_directory(self):
        packages = self.packages()
        with (
            patch.object(
                megaqwen, "load_megaqwen_ops", side_effect=RuntimeError("load failed")
            ) as load,
            self.assertRaisesRegex(RuntimeError, "load failed"),
        ):
            packaged_megaqwen.load_packaged_megaqwen_ops(packages)
        library = load.call_args.args[0]
        self.assertFalse(library.parent.exists())
        self.assertIsNone(packaged_megaqwen._packaged_directory)


class EagerProfileSelection(unittest.TestCase):
    def test_only_megaqwen_eager_validation_loads_the_packaged_operator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native = root / "xinfer_validate"
            native.touch()
            for backend in ("aoti", "megaqwen"):
                pair = {
                    kind: {
                        "model_type": "qwen3",
                        "model_specific": {} if backend == "aoti" else {"decode_backend": backend},
                    }
                    for kind in ("prefill", "decode")
                }
                modules = {
                    "compiler.qwen3.packages": SimpleNamespace(
                        validate_model_metadata=lambda _: None
                    ),
                    "tools.validation.qwen3.model": SimpleNamespace(
                        Suite=lambda model_dir, metadata: object(),
                        default_model_dir=lambda metadata: root / "model",
                    ),
                }
                arguments = [
                    "validate",
                    "--packages",
                    str(root / "packages"),
                    "--native",
                    str(native),
                    "--output",
                    str(root / backend),
                ]
                with (
                    self.subTest(backend=backend),
                    patch.object(sys, "argv", arguments),
                    patch.object(validate, "read_package_metadata", return_value=pair),
                    patch("compiler.common.weights.validate_shared_weights"),
                    patch("tools.validation.common.attention.load_packaged_attention_ops"),
                    patch.object(packaged_megaqwen, "load_packaged_megaqwen_ops") as load,
                    patch(
                        "tools.validation.common.chain.new_report", return_value={"passed": True}
                    ),
                    patch("tools.validation.common.pipeline.validate_chain"),
                    patch.object(
                        validate.importlib, "import_module", side_effect=modules.__getitem__
                    ),
                    self.assertRaises(SystemExit) as result,
                ):
                    validate.main()
                self.assertEqual(result.exception.code, 0)
                if backend == "megaqwen":
                    load.assert_called_once_with(root / "packages")
                else:
                    load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
