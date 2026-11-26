"""Unified target setup installs fresh virtual environments through Bash."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ENVIRONMENTS = Path(__file__).resolve().parents[1]
REPO_ROOT = ENVIRONMENTS.parent
WRAPPER = """#!/usr/bin/env bash
exec "$STUB_REAL_PYTHON" "$STUB_DRIVER" "$0" "$@"
"""
DRIVER = """import json
import os
from pathlib import Path
import sys

executable, arguments = sys.argv[1], sys.argv[2:]
requirements = {arguments[index + 1]: Path(arguments[index + 1]).read_text()
                for index, argument in enumerate(arguments[:-1]) if argument == "-r"}
with Path(os.environ["STUB_RECORD"]).open("a") as output:
    output.write(json.dumps(dict(
        executable=executable,
        arguments=arguments,
        cwd=os.getcwd(),
        pythonpath=os.environ.get("PYTHONPATH"),
        pythonhome=os.environ.get("PYTHONHOME"),
        requirements=requirements,
    )) + chr(10))
if arguments[:2] == ["-m", "venv"]:
    directory = Path(arguments[-1]) / "bin"
    directory.mkdir(parents=True)
    destination = directory / "python"
    destination.write_text(Path(executable).read_text())
    destination.chmod(0o755)
elif arguments[:3] == ["-m", "pip", "install"] and os.environ.get("STUB_FAIL_INSTALL"):
    sys.exit(29)
"""


class FreshTargetSetup(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "project with spaces"
        self.scripts = self.root / "environments"
        self.scripts.mkdir(parents=True)
        shutil.copy2(ENVIRONMENTS / "setup.sh", self.scripts / "setup.sh")
        shutil.copy2(ENVIRONMENTS / "requirements.txt", self.scripts / "requirements.txt")
        shutil.copy2(REPO_ROOT / "requirements-dev.txt", self.root / "requirements-dev.txt")
        for target in ("l20", "gb10"):
            (self.scripts / target).mkdir()
            shutil.copy2(
                ENVIRONMENTS / target / "requirements.txt",
                self.scripts / target / "requirements.txt",
            )
        self.outside = self.base / "caller"
        self.outside.mkdir()
        self.record = self.base / "calls.jsonl"
        self.driver = self.base / "driver.py"
        self.driver.write_text(DRIVER)
        self.bin = self.root / "bootstrap tools"
        self.bin.mkdir()
        self.bootstrap = self.bin / "python3.12"
        self.bootstrap.write_text(WRAPPER)
        self.bootstrap.chmod(0o755)
        self.environment = dict(
            os.environ,
            PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
            STUB_RECORD=str(self.record),
            STUB_DRIVER=str(self.driver),
            STUB_REAL_PYTHON=sys.executable,
        )

    def run_setup(self, *arguments, environment=None):
        return subprocess.run(
            ["bash", str(self.scripts / "setup.sh"), *arguments],
            cwd=self.outside,
            env=self.environment | (environment or {}),
            capture_output=True,
            text=True,
        )

    def calls(self):
        if not self.record.exists():
            return []
        return [json.loads(line) for line in self.record.read_text().splitlines()]

    def test_each_target_installs_into_uniform_default_venv(self):
        for target in ("l20", "gb10"):
            with self.subTest(target=target):
                self.record.unlink(missing_ok=True)
                result = self.run_setup("--target", target)
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = self.calls()
                venv = self.root / "work" / target / "venv"
                self.assertEqual(calls[0]["executable"], str(self.bootstrap))
                self.assertEqual(calls[0]["arguments"], ["-m", "venv", str(venv)])
                self.assertTrue(
                    all(Path(call["cwd"]).resolve() == self.root.resolve() for call in calls)
                )
                self.assertTrue(
                    all(call["executable"] == str(venv / "bin/python") for call in calls[1:])
                )
                installs = [
                    call["arguments"]
                    for call in calls
                    if call["arguments"][:3] == ["-m", "pip", "install"]
                ]
                self.assertEqual(len(installs), 1)
                selected = str(self.root / "environments" / target / "requirements.txt")
                self.assertEqual(installs[0], ["-m", "pip", "install", "-r", selected])
                self.assertEqual(calls[-1]["arguments"], ["-m", "pip", "check"])

    def test_dev_flag_appends_requirements_dev_file(self):
        result = self.run_setup("--target", "l20", "--dev")
        self.assertEqual(result.returncode, 0, result.stderr)
        install = next(
            call["arguments"]
            for call in self.calls()
            if call["arguments"][:3] == ["-m", "pip", "install"]
        )
        self.assertEqual(
            install,
            [
                "-m",
                "pip",
                "install",
                "-r",
                str(self.root / "environments/l20/requirements.txt"),
                "-r",
                str(self.root / "requirements-dev.txt"),
            ],
        )

    def test_custom_interpreter_venv_and_local_wheel_expand_shared_requirements(self):
        custom = self.bin / "custom python"
        custom.write_text(WRAPPER)
        custom.chmod(0o755)
        wheel = self.root / "wheel cache" / "torch-custom.whl"
        wheel.parent.mkdir()
        wheel.write_bytes(b"fixture")
        selected_file = self.root / "environments/gb10/requirements.txt"
        original = selected_file.read_text()
        result = self.run_setup(
            "--target",
            "gb10",
            "--python",
            str(custom),
            "--venv",
            "work/custom env",
            "--torch-wheel",
            "wheel cache/torch-custom.whl",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual(calls[0]["executable"], str(custom))
        self.assertEqual(calls[0]["arguments"], ["-m", "venv", str(self.root / "work/custom env")])
        install = next(call for call in calls if call["arguments"][:3] == ["-m", "pip", "install"])
        self.assertEqual(install["arguments"][3], str(wheel))
        self.assertEqual(len(install["requirements"]), 1)
        filtered_path, filtered_text = next(iter(install["requirements"].items()))
        self.assertNotIn("torch==", filtered_text)
        self.assertNotIn("-r ", filtered_text)
        self.assertIn("transformers==", filtered_text)
        self.assertIn("kernels==", filtered_text)
        self.assertEqual(selected_file.read_text(), original)
        self.assertFalse(Path(filtered_path).exists())

    def test_existing_environment_is_preserved_without_running_an_installer(self):
        venv = self.root / "work/existing env"
        venv.mkdir(parents=True)
        marker = venv / "keep.txt"
        marker.write_text("existing environment\n")
        result = self.run_setup("--target", "l20", "--venv", "work/existing env")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(marker.read_text(), "existing environment\n")
        self.assertEqual(self.calls(), [])

    def test_usage_errors_and_help_do_not_start_environment_creation(self):
        result = self.run_setup("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--target", result.stdout)
        for arguments in (
            ("--target",),
            ("--target", "other"),
            ("--unknown",),
            ("--target", "l20", "--venv"),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_setup(*arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.calls(), [])

    def test_installer_failure_cleans_up_partial_venv_and_stops_before_pip_check(self):
        venv = self.root / "work/l20/venv"
        result = self.run_setup("--target", "l20", environment={"STUB_FAIL_INSTALL": "1"})
        self.assertEqual(result.returncode, 29, result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[-1]["arguments"][:3], ["-m", "pip", "install"])
        self.assertFalse(venv.exists())

    def test_python_search_path_overrides_are_not_inherited(self):
        result = self.run_setup(
            "--target",
            "l20",
            environment={
                "PYTHONPATH": str(self.base / "unrelated packages"),
                "PYTHONHOME": str(self.base / "unrelated python"),
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.calls())
        self.assertTrue(
            all(call["pythonpath"] is None and call["pythonhome"] is None for call in self.calls())
        )


if __name__ == "__main__":
    unittest.main()
