"""CPU checks for the Qwen3 runtime validator's metadata and weight preflight."""

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
        self.load = self.enterContext(
            patch(
                "tools.validation.common.attention.load_packaged_attention_ops",
                side_effect=AssertionError("AOTI loads its bundled attention dependency"),
            )
        )
        self.manifest = self.enterContext(
            patch("compiler.common.artifacts.read_json", side_effect=StopBeforeRuntime)
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

    def test_valid_tensor_abi_needs_no_eager_operator_preload(self):
        with self.assertRaises(StopBeforeRuntime):
            self.run_validator()
        self.read.assert_called_once_with(self.packages, model_type="qwen3")
        self.weights.assert_called_once_with(
            self.packages, self.pair["prefill"], self.pair["decode"]
        )
        self.manifest.assert_called_once()
        self.load.assert_not_called()
        self.assertEqual(os.environ["XINFER_ATTENTION_OPS_LIBRARY"], "unrelated-attention.so")

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
                    self.load.assert_not_called()
                    self.manifest.assert_not_called()

    def test_invalid_weights_fail_before_checkpoint_loading(self):
        self.weights.side_effect = ValueError("Shared weights archive checksum mismatch")
        with self.assertRaisesRegex(ValueError, "Shared weights archive checksum mismatch"):
            self.run_validator()
        self.load.assert_not_called()
        self.manifest.assert_not_called()


if __name__ == "__main__":
    unittest.main()
