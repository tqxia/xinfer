"""Model validators must refuse execution when their assertions are disabled."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from compiler.common.artifacts import ROOT


class ValidationModes(unittest.TestCase):
    def test_normal_python_can_import_model_test_packages(self):
        env = {key: value for key, value in os.environ.items() if key != "PYTHONOPTIMIZE"}
        result = subprocess.run(
            [sys.executable, "-c", "import compiler.qwen3.tests, compiler.qwen35.tests"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_optimized_validators_fail_before_creating_reports(self):
        modules = (
            "compiler.qwen3.tests.validate_aoti",
            "compiler.qwen3.tests.validate_cpp",
            "compiler.qwen3.tests.smoke_cpp",
            "compiler.qwen35.tests.validate_model",
            "compiler.qwen35.tests.validate_runtime",
            "compiler.qwen35.tests.validate_end_to_end",
            "compiler.qwen35.tests.validate_cond",
            "compiler.qwen35.tests.validate_fused",
        )
        env = {key: value for key, value in os.environ.items() if key != "PYTHONOPTIMIZE"}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report"
            for flags, optimize in ((("-O",), None), (("-OO",), None), ((), "2")):
                mode_env = dict(env)
                if optimize:
                    mode_env["PYTHONOPTIMIZE"] = optimize
                for module in modules:
                    with self.subTest(module=module, flags=flags, optimize=optimize):
                        result = subprocess.run(
                            [sys.executable, *flags, "-m", module, "--output", str(output)],
                            cwd=ROOT,
                            env=mode_env,
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("Model validation requires Python assertions", result.stderr)
                        self.assertFalse(output.exists())

    def test_direct_rope_validator_preserves_normal_help_and_refuses_optimization(self):
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in ("PYTHONPATH", "PYTHONOPTIMIZE")
        }
        script = ROOT / "compiler/qwen3/tests/rope_precision.py"
        normal = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(normal.returncode, 0, normal.stdout + normal.stderr)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report"
            result = subprocess.run(
                [
                    sys.executable,
                    "-O",
                    str(script),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Model validation requires Python assertions", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
