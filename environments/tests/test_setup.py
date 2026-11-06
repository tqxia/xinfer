"""Fresh device setup installs independent model environments through Bash."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ENVIRONMENTS = Path(__file__).resolve().parents[1]
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


class FreshDeviceSetup(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "project with spaces"
        self.scripts = self.root / "environments"
        self.scripts.mkdir(parents=True)
        shutil.copy2(ENVIRONMENTS / "setup-common.sh", self.scripts / "setup-common.sh")
        for target in ("l20", "gb10"):
            (self.scripts / target).mkdir()
            for name in (
                "setup.sh",
                "qwen3_requirements.txt",
                "qwen35_requirements.txt",
            ):
                shutil.copy2(ENVIRONMENTS / target / name, self.scripts / target / name)
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

    def run_setup(self, target, *arguments, environment=None):
        return subprocess.run(
            ["bash", str(self.scripts / target / "setup.sh"), *arguments],
            cwd=self.outside,
            env=self.environment | (environment or {}),
            capture_output=True,
            text=True,
        )

    def calls(self):
        if not self.record.exists():
            return []
        return [json.loads(line) for line in self.record.read_text().splitlines()]

    def test_each_device_installs_its_native_stack_and_selected_model_in_a_fresh_venv(self):
        for target, model, suffix in (
            ("l20", "qwen3", "-torch213-venv"),
            ("l20", "qwen35", "-torch213-venv"),
            ("gb10", "qwen3", "-venv"),
            ("gb10", "qwen35", "-venv"),
        ):
            with self.subTest(target=target):
                self.record.unlink(missing_ok=True)
                result = self.run_setup(target, "--model", model)
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = self.calls()
                venv = self.root / "work" / target / (model + suffix)
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
                selected = str(self.root / "environments" / target / f"{model}_requirements.txt")
                self.assertIn(selected, installs[0])
                installed = next(call["requirements"] for call in calls if call["requirements"])
                self.assertEqual(installed, {selected: Path(selected).read_text()})
                self.assertFalse(
                    any(
                        line.strip().startswith(("-r ", "--requirement "))
                        for line in installed[selected].splitlines()
                    )
                )
                self.assertEqual(calls[-1]["arguments"], ["-m", "pip", "check"])

    def test_custom_interpreter_venv_and_local_wheel_preserve_spaces_and_repo_relative_paths(
        self,
    ):
        custom = self.bin / "custom python"
        custom.write_text(WRAPPER)
        custom.chmod(0o755)
        wheel = self.root / "wheel cache" / "torch-custom.whl"
        wheel.parent.mkdir()
        wheel.write_bytes(b"fixture")
        selected_file = self.root / "environments/gb10/qwen3_requirements.txt"
        original = selected_file.read_text()
        result = self.run_setup(
            "gb10",
            "--model",
            "qwen3",
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
        installs = [
            call["arguments"] for call in calls if call["arguments"][:3] == ["-m", "pip", "install"]
        ]
        wheel_arguments = [
            argument
            for arguments in installs
            for argument in arguments
            if argument.endswith(".whl")
        ]
        self.assertEqual(len(wheel_arguments), 1)
        actual = Path(wheel_arguments[0])
        self.assertEqual(actual if actual.is_absolute() else self.root / actual, wheel)
        self.assertFalse(
            any(argument.startswith("torch==") for arguments in installs for argument in arguments)
        )
        installed = next(call["requirements"] for call in calls if call["requirements"])
        self.assertEqual(len(installed), 1)
        filtered_path = next(iter(installed))
        self.assertEqual(
            installed[filtered_path],
            "".join(
                line
                for line in original.splitlines(keepends=True)
                if not line.startswith("torch==")
            ),
        )
        self.assertEqual(selected_file.read_text(), original)
        self.assertFalse(Path(filtered_path).exists())

    def test_existing_environment_is_preserved_without_running_an_installer(self):
        venv = self.root / "work/existing env"
        venv.mkdir(parents=True)
        marker = venv / "keep.txt"
        marker.write_text("existing environment\n")
        result = self.run_setup("l20", "--model", "qwen3", "--venv", "work/existing env")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(marker.read_text(), "existing environment\n")
        self.assertEqual(self.calls(), [])

    def test_usage_errors_and_help_do_not_start_environment_creation(self):
        result = self.run_setup("l20", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--model", result.stdout)
        for arguments in (
            (),
            ("--model",),
            ("--model", "other"),
            ("--unknown",),
            ("--model", "qwen3", "--venv"),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_setup("l20", *arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.calls(), [])

    def test_installer_failure_stops_before_pip_check(self):
        result = self.run_setup("l20", "--model", "qwen3", environment={"STUB_FAIL_INSTALL": "1"})
        self.assertEqual(result.returncode, 29, result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[-1]["arguments"][:3], ["-m", "pip", "install"])

    def test_python_search_path_overrides_are_not_inherited(self):
        result = self.run_setup(
            "l20",
            "--model",
            "qwen35",
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
