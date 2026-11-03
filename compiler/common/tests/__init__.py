"""Shared test support and execution requirements for model validators."""

import sys

from compiler.common.artifacts import require


def require_assertions():
    """Refuse model validation when Python would discard its assertions."""
    require(
        not sys.flags.optimize,
        "Model validation requires Python assertions; rerun without -O/-OO "
        "and unset PYTHONOPTIMIZE.",
    )
