"""Select packaged attention only for eager validation."""

import gc
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
from unittest.mock import Mock, patch

from compiler.common import attention
from tools.validation.common import attention as packaged_attention


class PackagedAttentionSelection(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.library = self.root / "libxinfer_attention_ops.so"
        self.library.touch()
        self.enterContext(patch.dict(os.environ))
        os.environ.pop("XINFER_ATTENTION_OPS_LIBRARY", None)
        self.enterContext(patch.object(attention, "_loaded", None))
        self.enterContext(patch.object(packaged_attention, "_packaged_directory", None))
        self.addCleanup(self.cleanup_packaged_library)
        self.torch = SimpleNamespace(
            ops=SimpleNamespace(load_library=Mock()),
            library=SimpleNamespace(register_fake=Mock(return_value=lambda function: function)),
        )

    def cleanup_packaged_library(self):
        if packaged_attention._packaged_directory is not None:
            packaged_attention._packaged_directory.cleanup()

    def packages(self, name="packages", **payloads):
        directory = self.root / name
        directory.mkdir()
        for kind in ("prefill", "decode"):
            with zipfile.ZipFile(directory / f"{kind}.pt2", "w") as archive:
                archive.writestr("fixture/data/aotinductor/model/model.wrapper.so", b"wrapper")
                payload = payloads.get(kind, b"packaged attention implementation")
                if payload is not None:
                    archive.writestr(
                        "fixture/data/aotinductor/model/libxinfer_attention_ops.so", payload
                    )
        return directory

    def test_missing_prefill_library_is_rejected_before_loading(self):
        packages = self.packages(prefill=None)
        with (
            patch.dict("sys.modules", {"torch": self.torch}),
            self.assertRaisesRegex(ValueError, "Package is missing its attention library"),
        ):
            packaged_attention.load_packaged_attention_ops(packages)
        self.torch.ops.load_library.assert_not_called()
        self.torch.library.register_fake.assert_not_called()

    def test_eager_selection_does_not_require_matching_decode_attention(self):
        for decode in (b"different implementation", None):
            with self.subTest(decode=decode):
                packages = self.packages(
                    str(decode), prefill=b"prefill implementation", decode=decode
                )
                with (
                    patch.object(attention, "_loaded", None),
                    patch.dict("sys.modules", {"torch": self.torch}),
                ):
                    library = packaged_attention.load_packaged_attention_ops(packages)
                    self.assertEqual(library.read_bytes(), b"prefill implementation")
                packaged_attention._packaged_directory.cleanup()
        self.assertEqual(self.torch.ops.load_library.call_count, 2)

    def test_packaged_eager_library_is_retained_and_reused(self):
        packages = self.packages()
        with patch.dict("sys.modules", {"torch": self.torch}):
            library = packaged_attention.load_packaged_attention_ops(packages).resolve()
            self.assertEqual(library.name, "libxinfer_attention_ops.so")
            self.assertEqual(library.read_bytes(), b"packaged attention implementation")
            self.assertNotEqual(library.parent, packages)
            gc.collect()
            self.assertTrue(library.is_file())
            self.assertEqual(
                packaged_attention.load_packaged_attention_ops(packages).resolve(), library
            )
            self.assertEqual(attention.attention_library(), library)
        self.torch.ops.load_library.assert_called_once_with(str(library))
        self.torch.library.register_fake.assert_called_once_with("xinfer::flash_attention")

    def test_packaged_selection_survives_a_stale_environment_override(self):
        packages = self.packages()
        stale = str(self.root / "missing/libxinfer_attention_ops.so")
        os.environ["XINFER_ATTENTION_OPS_LIBRARY"] = stale
        with patch.dict("sys.modules", {"torch": self.torch}):
            library = packaged_attention.load_packaged_attention_ops(packages).resolve()
            attention.load_attention_ops(model="qwen3")
            attention.load_attention_ops(model="qwen35")
            self.assertEqual(attention.attention_library(), library)
        self.torch.ops.load_library.assert_called_once_with(str(library))
        self.assertEqual(os.environ["XINFER_ATTENTION_OPS_LIBRARY"], stale)

    def test_preloaded_eager_operator_is_reused_without_comparing_library_bytes(self):
        self.library.write_bytes(b"separately built implementation")
        packages = self.packages()
        with patch.dict("sys.modules", {"torch": self.torch}):
            attention.load_attention_ops(self.library)
            with patch.object(Path, "read_bytes", side_effect=AssertionError("No byte comparison")):
                self.assertEqual(
                    packaged_attention.load_packaged_attention_ops(packages), self.library
                )
        self.torch.ops.load_library.assert_called_once_with(str(self.library))
        self.torch.library.register_fake.assert_called_once_with("xinfer::flash_attention")


if __name__ == "__main__":
    unittest.main()
