"""Small file helpers shared by model tools; no model or Torch imports."""

import hashlib
import json
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data, *, ensure_ascii=True):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=ensure_ascii) + "\n")
