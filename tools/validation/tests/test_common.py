"""CPU checks for validation command execution and failure reporting."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from compiler.common.artifacts import ROOT


class CommandExecution(unittest.TestCase):
    def test_command_status_and_error_expectations_survive_optimization(self):
        code = """
import json
from pathlib import Path
import sys
from tools.common import run

case = json.loads(sys.argv[1])
log = Path(sys.argv[2])
command = [sys.executable, "-c", case["command"]]
try:
    run(command, None, log, success=case["success"], error=case["error"])
except ValueError as error:
    exception = str(error)
else:
    exception = None
print(json.dumps(dict(exception=exception, log=log.read_text())))
"""
        command = (
            "import sys; print('stdout detail'); "
            "print('stderr detail', file=sys.stderr); sys.exit({status})"
        )
        cases = (
            ("expected_success", 0, True, None, False),
            ("expected_failure", 7, False, "stderr detail", False),
            ("unexpected_success", 0, False, None, True),
            ("unexpected_failure", 7, True, None, True),
            ("missing_expected_error", 7, False, "missing error", True),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for option in (None, "-O", "-OO"):
                for name, status, success, error, rejected in cases:
                    with self.subTest(optimization=option, case=name):
                        case = dict(
                            command=command.format(status=status), success=success, error=error
                        )
                        log = Path(temporary) / f"{option}-{name}.log"
                        result = subprocess.run(
                            [
                                sys.executable,
                                *([option] if option else []),
                                "-c",
                                code,
                                json.dumps(case),
                                str(log),
                            ],
                            cwd=ROOT,
                            capture_output=True,
                            text=True,
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
                        record = json.loads(result.stdout)
                        self.assertIn("stdout detail", record["log"])
                        self.assertIn("stderr detail", record["log"])
                        if rejected:
                            self.assertEqual(record["exception"], record["log"])
                        else:
                            self.assertIsNone(record["exception"])


if __name__ == "__main__":
    unittest.main()
