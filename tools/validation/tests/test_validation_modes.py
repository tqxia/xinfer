"""Model validation tools refuse execution when assertions are disabled."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from compiler.common.artifacts import ROOT


class ValidationModes(unittest.TestCase):
    def test_optimized_validators_fail_before_creating_reports(self):
        modules = (
            "tools.validation.validate",
            "tools.validation.qwen3.validate_runtime",
            "tools.validation.qwen35.validate_runtime",
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


if __name__ == "__main__":
    unittest.main()
