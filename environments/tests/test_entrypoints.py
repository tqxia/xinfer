"""Bash entry points preserve interpreter choice, argv, cwd and process status."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1]
STUB = '''#!/bin/bash
printf '%s\\0' "$PWD" "$@" > "$STUB_RECORD"
printf '%s' '{"stub":true}'
exit "${STUB_EXIT:-0}"
'''


class EnvironmentEntrypoints(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "project with spaces"
        self.scripts = self.root / "environments"
        self.scripts.mkdir(parents=True)
        for name in ("setup.sh", "check.sh"):
            shutil.copy2(SCRIPTS / name, self.scripts / name)
        self.outside = self.base / "caller"
        self.outside.mkdir()
        self.record = self.base / "argv.bin"
        self.bin = self.root / "host tools"
        self.bin.mkdir()
        self.write_interpreter(self.bin / "python3")
        self.model_python = self.root / "model env/bin/python"
        self.write_interpreter(self.model_python)
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                        STUB_RECORD=str(self.record))

    @staticmethod
    def write_interpreter(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(STUB)
        path.chmod(0o755)

    def call(self, script, *args, status=0):
        return subprocess.run(["bash", str(self.scripts / script), *args], cwd=self.outside,
                              env=dict(self.env, STUB_EXIT=str(status)), text=True, capture_output=True)

    def invocation(self):
        return self.record.read_bytes().decode().split("\0")[:-1]

    def test_setup_bootstraps_from_any_directory_and_preserves_argument_boundaries(self):
        args = ["--target", "gb10", "--model", "qwen35", "--native-python", "native stack/bin/python",
                "--output", "work/env with spaces"]
        result = self.call("setup.sh", *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '{"stub":true}')
        self.assertEqual(self.invocation(), [str(self.root), "-m", "environments.setup", *args])

    def test_check_uses_selected_interpreter_and_repository_relative_paths(self):
        args = ["--target", "l20", "--model", "qwen3", "--output", "artifacts/report with spaces.json"]
        result = self.call("check.sh", "--python", "model env/bin/python", *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '{"stub":true}')
        self.assertEqual(self.invocation(), [str(self.root), "-m", "environments.check", *args])

    def test_absolute_interpreter_and_equals_option(self):
        result = self.call("check.sh", "--python=" + str(self.model_python), "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.invocation(), [str(self.root), "-m", "environments.check", "--help"])

    def test_help_requires_no_model_interpreter(self):
        result = self.call("check.sh", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--python MODEL_PYTHON", result.stdout)
        self.assertEqual(self.invocation(), [str(self.root), "-m", "environments.check", "--help"])

    def test_missing_interpreter_fails_before_running_any_python(self):
        for args in ((), ("--target", "l20"), ("--python",), ("--python", "--target"), ("--python=",)):
            with self.subTest(args=args):
                result = self.call("check.sh", *args)
                self.assertEqual(result.returncode, 2)
                self.assertIn("--python MODEL_PYTHON", result.stderr)
                self.assertFalse(self.record.exists())

    def test_exit_status_and_json_stdout_are_preserved(self):
        for script, args in (("setup.sh", ["--help"]),
                             ("check.sh", ["--python", str(self.model_python)])):
            with self.subTest(script=script):
                result = self.call(script, *args, status=23)
                self.assertEqual(result.returncode, 23)
                self.assertEqual(result.stdout, '{"stub":true}')


if __name__ == "__main__":
    unittest.main()
