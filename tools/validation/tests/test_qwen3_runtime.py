"""CPU checks for the Qwen3 runtime validator's metadata and weight preflight."""

import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.tests.test_packages import metadata
from compiler.common.artifacts import ROOT
from tools.validation.qwen3 import validate_runtime as validator


class StopBeforeRuntime(Exception):
    pass


class Qwen3RuntimeDependencies(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.native = self.root / "xinfer_aoti"
        self.packages = self.root / "packages"
        self.metadata = metadata()
        self.enterContext(
            patch.dict(os.environ, XINFER_ATTENTION_OPS_LIBRARY="unrelated-attention.so")
        )
        self.read = self.enterContext(
            patch.object(validator, "read_package_metadata", return_value=self.metadata)
        )
        self.weights = self.enterContext(patch.object(validator, "validate_shared_weights"))
        self.load = self.enterContext(
            patch(
                "tools.validation.common.operators.load_packaged_attention_ops",
                side_effect=AssertionError("AOTI loads its bundled attention dependency"),
            )
        )
        self.manifest = self.enterContext(
            patch.object(validator, "read_json", side_effect=StopBeforeRuntime)
        )
        self.enterContext(
            patch("subprocess.check_output", return_value="libtorch.so => /native/libtorch.so\n")
        )

    def run_validator(self, name="validation"):
        return validator.validate_runtime(
            native=self.native,
            packages=self.packages,
            model_dir=self.root / "checkpoint",
            output=self.root / name,
        )

    def test_valid_tensor_abi_needs_no_eager_operator_preload(self):
        with self.assertRaises(StopBeforeRuntime):
            self.run_validator()
        self.read.assert_called_once_with(self.packages, model_type="qwen3")
        self.weights.assert_called_once_with(self.packages, self.metadata)
        self.manifest.assert_called_once()
        self.load.assert_not_called()
        self.assertEqual(os.environ["XINFER_ATTENTION_OPS_LIBRARY"], "unrelated-attention.so")

    def test_geometry_validation_precedes_loading(self):
        for index, shape in enumerate(([8, 4, 1024, 128], [8, 8, 1024, 256])):
            with self.subTest(valid_shape=shape):
                value = copy.deepcopy(self.metadata)
                for state in value["state"]:
                    state["shape"] = shape
                self.read.return_value = value
                with self.assertRaises(StopBeforeRuntime):
                    self.run_validator(f"valid-{index}")
        self.manifest.reset_mock()
        for index, shape in enumerate(([8, 4, 1024, 128], [8, 8, 1024, 64])):
            with self.subTest(shape=shape):
                value = copy.deepcopy(self.metadata)
                value["state"][0]["shape"] = shape
                self.read.return_value = value
                with self.assertRaisesRegex(ValueError, "^K/V shapes differ"):
                    self.run_validator(f"mismatch-{index}")
                self.load.assert_not_called()
                self.manifest.assert_not_called()

    def test_invalid_weights_fail_before_checkpoint_loading(self):
        self.weights.side_effect = ValueError("Shared weights archive checksum mismatch")
        with self.assertRaisesRegex(ValueError, "Shared weights archive checksum mismatch"):
            self.run_validator()
        self.load.assert_not_called()
        self.manifest.assert_not_called()


class RuntimeImport(unittest.TestCase):
    def test_import_does_not_parse_arguments_or_start_validation(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import tools.validation.qwen3.validate_runtime",
                "--not-a-cli-argument",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
