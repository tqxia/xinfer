"""Artifact I/O, deterministic hashing and validation; no model or Torch imports."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

NATIVE_RUNTIME_SOURCES = (
    "CMakeLists.txt", "runtime/CMakeLists.txt", "runtime/qwen35/CMakeLists.txt",
    "runtime/main.cpp", "runtime/benchmark.cpp", "runtime/validate.cpp",
    "runtime/qwen3/runtime.h", "runtime/qwen35/runtime.h",
    "runtime/common/spec.h", "runtime/common/runner.h", "runtime/common/model.h",
    "runtime/common/io.h", "runtime/common/generation.h", "runtime/common/benchmark.h",
    "runtime/common/weights.h", "runtime/common/sha256.h",
    "runtime/common/attention.cpp", "runtime/common/attention.h",
)


def canonical_json(value):
    """Encode metadata deterministically for hashing and archive records."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def json_sha256(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def require(condition, message):
    """Validate external inputs even when Python assertions are disabled."""
    if not condition:
        raise ValueError(message)


def validate_model_type(prefill, decode, expected):
    """Accept an untagged legacy pair or two matching explicit model tags."""
    present = "model_type" in decode
    require(("model_type" in prefill) == present and
            prefill.get("model_type") == decode.get("model_type"), "Package model types differ")
    if present:
        require(decode["model_type"] == expected, "Unsupported model type")


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
