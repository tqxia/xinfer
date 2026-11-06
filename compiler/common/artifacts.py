"""Artifact I/O, deterministic hashing and validation; no model or Torch imports."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_MANIFEST_NAME = "model_manifest.json"


def default_build_dir(model):
    """Locate the model's native build independently of the working directory."""
    return ROOT / "build" / model


def canonical_json(value):
    """Encode metadata deterministically for hashing and archive records."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def json_sha256(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def require(condition, message):
    """Validate external inputs even when Python assertions are disabled."""
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data, *, ensure_ascii=True):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=ensure_ascii) + "\n")
