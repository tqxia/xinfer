"""Reject unsupported HF implementations before loading adapter dependencies."""

import builtins
import importlib.metadata
import runpy
import unittest
from unittest.mock import patch

from compiler.common.artifacts import ROOT


class ReachedModelImports(Exception):
    pass


class AdapterVersion(unittest.TestCase):
    def probe(self, *, version=None, error=None):
        imports = []
        original_import = builtins.__import__

        def model_import(name, *args, **kwargs):
            if name == "torch" or name == "transformers" or name.startswith("transformers."):
                imports.append(name)
                raise ReachedModelImports(name)
            return original_import(name, *args, **kwargs)

        version_mock = patch("importlib.metadata.version", return_value=version, side_effect=error)
        return imports, version_mock, patch("builtins.__import__", side_effect=model_import)

    def test_unsupported_versions_fail_before_torch_or_model_imports(self):
        for version in ("4.51.4", "4.52.0", "5.20.2"):
            with self.subTest(version=version):
                imports, version_mock, import_mock = self.probe(version=version)
                with (
                    version_mock as installed,
                    import_mock,
                    self.assertRaisesRegex(RuntimeError, "requires transformers==4.51.3; found"),
                ):
                    runpy.run_path(str(ROOT / "compiler/qwen3/adapters/qwen3_flash.py"))
                installed.assert_called_once_with("transformers")
                self.assertEqual(imports, [])

    def test_missing_transformers_has_an_actionable_error_before_imports(self):
        imports, version_mock, import_mock = self.probe(
            error=importlib.metadata.PackageNotFoundError("transformers")
        )
        with (
            version_mock as installed,
            import_mock,
            self.assertRaisesRegex(RuntimeError, "requires transformers==4.51.3"),
        ):
            runpy.run_path(str(ROOT / "compiler/qwen3/adapters/qwen3_flash.py"))
        installed.assert_called_once_with("transformers")
        self.assertEqual(imports, [])

    def test_exact_pin_reaches_adapter_dependency_imports(self):
        imports, version_mock, import_mock = self.probe(version="4.51.3")
        with (
            version_mock as installed,
            import_mock,
            self.assertRaises(ReachedModelImports),
        ):
            runpy.run_path(str(ROOT / "compiler/qwen3/adapters/qwen3_flash.py"))
        installed.assert_called_once_with("transformers")
        self.assertEqual(imports, ["torch"])


if __name__ == "__main__":
    unittest.main()
