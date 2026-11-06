"""Unit tests and execution requirements for common compiler support."""

import sys

from compiler.common.artifacts import require


def require_assertions():
    """Refuse model/operator tests when Python would discard their assertions."""
    require(
        not sys.flags.optimize,
        "Model validation requires Python assertions; rerun without -O/-OO "
        "and unset PYTHONOPTIMIZE.",
    )
