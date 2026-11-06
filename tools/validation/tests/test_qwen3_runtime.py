"""CPU checks for the Qwen3 runtime validator's attention-library preflight."""

import copy
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

from compiler.qwen3.tests.test_packages import metadata


class StopBeforeRuntime(Exception):
    pass


class Qwen3RuntimeDependencies(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.native = self.root / "xinfer_aoti"
        self.library = self.root / "libxinfer_attention_ops.so"
        self.packages = self.root / "packages"
        self.pair = {kind: metadata() for kind in ("prefill", "decode")}
        self.pair["prefill"].update(entry_point="prefill", token_range=[2, 512])
        self.enterContext(
            patch.dict(os.environ, XINFER_ATTENTION_OPS_LIBRARY="unrelated-attention.so")
        )
        self.read = self.enterContext(
            patch("compiler.common.metadata.read_package_metadata", return_value=self.pair)
        )
        self.weights = self.enterContext(patch("compiler.common.weights.validate_shared_weights"))
        self.resolve = self.enterContext(
            patch("tools.common.native_attention_library", return_value=self.library)
        )
        self.load = self.enterContext(
            patch("compiler.common.attention.load_attention_ops", side_effect=StopBeforeRuntime)
        )
        self.enterContext(
            patch("subprocess.check_output", return_value="libtorch.so => /native/libtorch.so\n")
        )

    def run_validator(self, name="validation"):
        arguments = [
            "validate_runtime",
            "--native",
            str(self.native),
            "--packages",
            str(self.packages),
            "--output",
            str(self.root / name),
        ]
        with patch.object(sys, "argv", arguments):
            runpy.run_module("tools.validation.qwen3.validate_runtime", run_name="__main__")

    def test_valid_tensor_abi_loads_the_native_dependency(self):
        with self.assertRaises(StopBeforeRuntime):
            self.run_validator()
        self.read.assert_called_once_with(self.packages, model_type="qwen3")
        self.weights.assert_called_once_with(
            self.packages, self.pair["prefill"], self.pair["decode"]
        )
        self.resolve.assert_called_once_with(self.native)
        self.load.assert_called_once_with(self.library, model="qwen3")
        self.assertEqual(os.environ["XINFER_ATTENTION_OPS_LIBRARY"], str(self.library))

    def test_geometry_validation_precedes_loading(self):
        for kind in ("prefill", "decode"):
            for index, shape in enumerate(([1, 4, 1024, 128], [1, 8, 1024, 64])):
                with self.subTest(entry=kind, shape=shape):
                    value = copy.deepcopy(self.pair)
                    for state in value[kind]["state"]:
                        state["shape"] = shape
                    self.read.return_value = value
                    with self.assertRaisesRegex(ValueError, "^Unsupported Flash cache shape"):
                        self.run_validator(f"{kind}-{index}")
                    self.resolve.assert_not_called()
                    self.load.assert_not_called()

    def test_missing_native_dependency_does_not_fall_back(self):
        self.resolve.return_value = None
        with self.assertRaisesRegex(
            ValueError, "^Qwen3 runtime is missing its required native attention library"
        ):
            self.run_validator()
        self.resolve.assert_called_once_with(self.native)
        self.load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
