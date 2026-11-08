"""Process execution and prompt partitioning shared by tooling."""

import subprocess

from compiler.common.artifacts import require


def prompt_chunks(tokens, max_prefill):
    """Partition a request into legal graph calls, including a decode remainder."""
    require(bool(tokens), "Prompt must contain tokens")
    require(type(max_prefill) is int and max_prefill >= 2, "Invalid prefill bound")
    return [tokens[start : start + max_prefill] for start in range(0, len(tokens), max_prefill)]


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
