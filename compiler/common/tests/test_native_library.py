"""Use model-specific native build defaults while preserving explicit library selection."""

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from compiler.common import attention


class NativeLibrarySelection(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.library = self.root / "selected build/libxinfer_attention_ops.so"
        self.library.parent.mkdir()
        self.library.touch()
        self.other = self.root / "other.so"
        self.other.touch()
        self.enterContext(patch.dict(os.environ))
        os.environ.pop("XINFER_ATTENTION_OPS_LIBRARY", None)
        self.enterContext(patch.object(attention, "_loaded", None))
        self.default = self.root / "build/qwen3/libxinfer_attention_ops.so"
        self.enterContext(patch("compiler.common.artifacts.ROOT", self.root))
        self.torch = SimpleNamespace(
            ops=SimpleNamespace(load_library=Mock()),
            library=SimpleNamespace(register_fake=Mock(return_value=lambda function: function)),
        )

    def test_unconfigured_loader_ignores_a_stale_default_build(self):
        stale = self.root / "build/libxinfer_attention_ops.so"
        stale.parent.mkdir()
        stale.touch()
        with (
            patch.dict("sys.modules", {"torch": None}),
            self.assertRaises(FileNotFoundError) as error,
        ):
            attention.load_attention_ops(model="qwen3")
        self.assertIn(str(self.default), str(error.exception))
        self.assertIsNone(attention._loaded)

    def test_no_override_loads_the_model_build_library(self):
        self.default.parent.mkdir(parents=True)
        self.default.touch()
        with patch.dict("sys.modules", {"torch": self.torch}):
            attention.load_attention_ops(model="qwen3")
        self.torch.ops.load_library.assert_called_once_with(str(self.default))
        self.assertEqual(attention._loaded, self.default)

    def test_qwen35_uses_its_own_build_even_when_qwen3_exists(self):
        self.default.parent.mkdir(parents=True)
        self.default.touch()
        library = self.root / "build/qwen35/libxinfer_attention_ops.so"
        library.parent.mkdir(parents=True)
        library.touch()
        with patch.dict("sys.modules", {"torch": self.torch}):
            attention.load_attention_ops(model="qwen35")
        self.torch.ops.load_library.assert_called_once_with(str(library))
        self.assertEqual(attention._loaded, library)

    def test_missing_model_build_does_not_fall_back_to_the_other_model(self):
        self.default.parent.mkdir(parents=True)
        self.default.touch()
        library = self.root / "build/qwen35/libxinfer_attention_ops.so"
        with (
            patch.dict("sys.modules", {"torch": None}),
            self.assertRaises(FileNotFoundError) as error,
        ):
            attention.load_attention_ops(model="qwen35")
        self.assertIn(str(library), str(error.exception))
        self.assertIsNone(attention._loaded)

    def test_unconfigured_loader_requires_a_model(self):
        with (
            patch.dict("sys.modules", {"torch": None}),
            self.assertRaisesRegex(ValueError, "Select a model"),
        ):
            attention.load_attention_ops()

    def test_missing_selected_file_fails_before_importing_torch(self):
        selected = self.root / "missing/libxinfer_attention_ops.so"
        os.environ["XINFER_ATTENTION_OPS_LIBRARY"] = str(selected)
        with (
            patch.dict("sys.modules", {"torch": None}),
            self.assertRaises(FileNotFoundError) as error,
        ):
            attention.load_attention_ops()
        self.assertIn(str(selected), str(error.exception))
        self.assertIsNone(attention._loaded)

    def test_environment_selects_exactly_one_library(self):
        os.environ["XINFER_ATTENTION_OPS_LIBRARY"] = str(self.library)
        with patch.dict("sys.modules", {"torch": self.torch}):
            attention.load_attention_ops()
        self.torch.ops.load_library.assert_called_once_with(str(self.library))
        self.assertEqual(attention._loaded, self.library)

    def test_explicit_path_overrides_environment(self):
        os.environ["XINFER_ATTENTION_OPS_LIBRARY"] = str(self.other)
        with patch.dict("sys.modules", {"torch": self.torch}):
            attention.load_attention_ops(self.library)
        self.torch.ops.load_library.assert_called_once_with(str(self.library))

    def test_loaded_operator_can_be_reused_without_reselecting_a_path(self):
        with patch.dict("sys.modules", {"torch": self.torch}):
            attention.load_attention_ops(self.library)
            attention.load_attention_ops()
            attention.load_attention_ops(self.library)
        self.torch.ops.load_library.assert_called_once_with(str(self.library))
        self.torch.library.register_fake.assert_called_once_with("xinfer::flash_attention")

    def test_conflicting_library_is_rejected(self):
        with patch.dict("sys.modules", {"torch": self.torch}):
            attention.load_attention_ops(self.library)
            with self.assertRaisesRegex(ValueError, "two different attention libraries"):
                attention.load_attention_ops(self.other)
        self.torch.ops.load_library.assert_called_once_with(str(self.library))


if __name__ == "__main__":
    unittest.main()
