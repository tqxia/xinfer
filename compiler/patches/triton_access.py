"""Scoped Torch 2.13 workaround for repeated Triton read-dependency DAG walks.

Only read traversal is memoized. Write counts, unknown-op checks and epilogue
eligibility retain the pinned Torch implementation. No installed files change.
"""

import ast
from contextlib import contextmanager
import hashlib
import inspect
from types import CodeType, FunctionType

import torch

from compiler.common.artifacts import require


ANALYZER_SHA256 = "3b2059e547e3f98016df464b97f97c82325ac084241c944ea23cc3da40609931"


def _same(node, text):
    return ast.dump(node, include_attributes=False) == ast.dump(
        ast.parse(text).body[0], include_attributes=False
    )


def _memoized_reads(original, globals_):
    try:
        source = inspect.getsource(original)
    except (OSError, TypeError) as error:
        raise ValueError("Unsupported Torch Triton analyzer source") from error
    require(
        hashlib.sha256(source.encode()).hexdigest() == ANALYZER_SHA256,
        "Unsupported Torch Triton analyzer source",
    )
    tree = ast.parse(source)
    require(
        len(tree.body) == 1
        and isinstance(tree.body[0], ast.FunctionDef)
        and tree.body[0].name == "analyze_kernel_access"
        and len(tree.body[0].decorator_list) == 1
        and isinstance(tree.body[0].decorator_list[0], ast.Name)
        and tree.body[0].decorator_list[0].id == "MemoizeWithCycleCheck"
        and original.__closure__ is None,
        "Unsupported Torch Triton analyzer structure",
    )
    function = tree.body[0]
    parents = {child: node for node in ast.walk(function) for child in ast.iter_child_nodes(node)}
    read_uses = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Name) and node.id == "read_count" and isinstance(node.ctx, ast.Load)
    ]
    require(
        len(read_uses) == 1
        and isinstance(parents[read_uses[0]], ast.Attribute)
        and parents[read_uses[0]].attr == "keys"
        and isinstance(parents[parents[read_uses[0]]], ast.Call)
        and not parents[parents[read_uses[0]]].args
        and not parents[parents[read_uses[0]]].keywords,
        "Unsupported Torch Triton read-count consumer",
    )
    nested = [
        node
        for node in function.body
        if isinstance(node, ast.FunctionDef) and node.name == "_find_arg_access_count"
    ]
    require(len(nested) == 1, "Unsupported Torch Triton analyzer structure")
    walk = nested[0]
    require(
        [arg.arg for arg in walk.args.args] == ["initial_stack", "skip_loads"],
        "Unsupported Torch Triton analyzer structure",
    )
    stacks = [i for i, node in enumerate(walk.body) if _same(node, "stack = initial_stack[:]")]
    loops = [node for node in walk.body if isinstance(node, ast.While)]
    require(
        len(stacks) == len(loops) == 1
        and isinstance(loops[0].test, ast.Name)
        and loops[0].test.id == "stack"
        and _same(loops[0].body[0], "arg = stack.pop()"),
        "Unsupported Torch Triton analyzer structure",
    )
    # read_count is consumed only through its keys. Visiting an Intermediate
    # once preserves that exact reachability set without expanding shared DAG
    # paths exponentially. Never deduplicate writes: their multiplicity controls
    # whether Torch may fuse an epilogue into a single-store kernel.
    walk.body.insert(stacks[0] + 1, ast.parse("visited_intermediates = set()").body[0])
    loops[0].body[1:1] = ast.parse(
        "if not skip_loads and isinstance(arg, Intermediate):\n"
        "    if arg in visited_intermediates:\n"
        "        continue\n"
        "    visited_intermediates.add(arg)\n"
    ).body
    function.decorator_list = []
    ast.fix_missing_locations(tree)
    code = compile(tree, "<xinfer Triton read-dependency workaround>", "exec")
    bodies = [
        value
        for value in code.co_consts
        if isinstance(value, CodeType) and value.co_name == function.name
    ]
    require(len(bodies) == 1, "Unsupported Torch Triton analyzer structure")
    # Keep the original module globals. Recursive nested-function analysis must
    # still go through Torch's original MemoizeWithCycleCheck object/cache.
    result = FunctionType(bodies[0], globals_, original.__name__, original.__defaults__)
    result.__kwdefaults__ = original.__kwdefaults__
    result.__annotations__ = original.__annotations__
    return result


@contextmanager
def memoize_triton_reads():
    """Memoize read reachability during one pinned-stack compilation.

    This temporarily changes a process-global Torch analyzer; use it around
    the sequential compiler entry point, not concurrent independent compiles.
    Unsupported Torch versions or analyzer sources fail before any mutation.
    """
    from torch._higher_order_ops import triton_kernel_wrap as implementation

    require(
        str(torch.__version__).split("+", 1)[0] == "2.13.0",
        "Triton read-dependency workaround requires torch==2.13.0",
    )
    analyzer = implementation.analyze_kernel_access
    original = analyzer.fn
    replacement = _memoized_reads(original, vars(implementation))
    analyzer.reset()
    implementation.get_tma_stores.reset()
    analyzer.fn = replacement
    try:
        yield
    finally:
        analyzer.fn = original
        analyzer.reset()
        implementation.get_tma_stores.reset()
