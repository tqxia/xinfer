"""Small file helpers shared by model tools; no model or Torch imports."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data, *, ensure_ascii=True):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=ensure_ascii) + "\n")


def run(command, env, log, success=True, error=None):
    with log.open("w") as stream:
        result = subprocess.run([str(c) for c in command], env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=1200)
    assert (result.returncode == 0) == success, log.read_text()
    if error:
        assert error in log.read_text(), log.read_text()
