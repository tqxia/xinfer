"""Keep metadata hashes stable and input validation active under Python -O."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from compiler.common.artifacts import ROOT, canonical_json, json_sha256


class ArtifactHelpers(unittest.TestCase):
    def test_metadata_hash_preserves_canonical_encoding(self):
        metadata = {"z": [None, True, -0.0, 1.25], "name": "紫霄", "a": {"right": 2, "left": 1}}
        expected = '{"a":{"left":1,"right":2},"name":"\\u7d2b\\u9704","z":[null,true,-0.0,1.25]}'
        self.assertEqual(canonical_json(metadata), expected)
        self.assertEqual(
            json_sha256(metadata), hashlib.sha256(expected.encode("utf-8")).hexdigest()
        )
        reordered = {"a": {"left": 1, "right": 2}, "name": "紫霄", "z": metadata["z"]}
        self.assertEqual(json_sha256(metadata), json_sha256(reordered))

    def test_input_validation_survives_optimized_python(self):
        code = (
            "from compiler.common.artifacts import require\n"
            "assert False, 'this assertion is removed under -O'\n"
            "require(False, 'invalid archive checksum')\n"
        )
        result = subprocess.run(
            [sys.executable, "-O", "-c", code], cwd=ROOT, capture_output=True, text=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ValueError: invalid archive checksum", result.stderr)

    def test_command_status_and_error_expectations_survive_optimization(self):
        code = """
import json
from pathlib import Path
import sys
from compiler.common.artifacts import run

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
