"""Process execution and native-library discovery shared by tooling."""

import os
from pathlib import Path
import subprocess

from compiler.common.artifacts import require


def native_attention_library(native):
    """Resolve the library used by a native worker under its sanitized environment."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH")
    }
    linked = subprocess.check_output(["ldd", str(Path(native).resolve())], env=env, text=True)
    for line in linked.splitlines():
        name, separator, target = line.strip().partition(" => ")
        if separator and name == "libxinfer_attention_ops.so":
            library = Path(target.rsplit(" (", 1)[0])
            if not library.is_file():
                raise FileNotFoundError(f"Native attention dependency is missing: {line.strip()}")
            return library.resolve()
    return None


def run(command, env, log, success=True, error=None):
    with log.open("w") as stream:
        result = subprocess.run(
            [str(c) for c in command],
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=1200,
        )
    output = log.read_text()
    require((result.returncode == 0) == success, output)
    if error:
        require(error in output, output)
