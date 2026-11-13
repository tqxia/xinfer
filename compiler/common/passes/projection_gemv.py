"""Experimentally replace the profiled decoder singleton linear projections."""

import torch
from torch.export.graph_signature import InputKind
from torch.fx.experimental.symbolic_shapes import statically_known_true

from compiler.common.kernels.projection_gemv import projection_gemv  # registers the operator


# Scope the experiment to the measured decoder regions, leaving the vocabulary
# head and every unprofiled geometry on their existing implementation.
PROFILED_SHAPES = frozenset({(1024, 8224), (2048, 1024), (1024, 7168), (3584, 1024), (1024, 5120)})


def replace_projection_gemv(exported):
    """Preserve parameters, signature, names and output metadata while changing calls."""
    parameters = {
        spec.arg.name: spec
        for spec in exported.graph_signature.input_specs
        if spec.kind == InputKind.PARAMETER
    }
    replacements = []
    for node in exported.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.linear.default:
            continue
        if node.kwargs or len(node.args) not in (2, 3):
            continue
        if len(node.args) == 3 and node.args[2] is not None:
            continue
        hidden, weight = node.args[:2]
        if not isinstance(hidden, torch.fx.Node) or not isinstance(weight, torch.fx.Node):
            continue
        spec = parameters.get(weight.name)
        if spec is None or len(weight.users) != 1:
            continue
        value = exported.state_dict[spec.target]
        x = hidden.meta.get("val")
        if not isinstance(x, torch.Tensor) or value.requires_grad or value.ndim != 2:
            continue
        if x.device.type != "cuda" or value.device != x.device:
            continue
        if x.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
            continue
        n, k = value.shape
        if (k, n) not in PROFILED_SHAPES or x.ndim < 2:
            continue
        if not all(statically_known_true(size == 1) for size in x.shape[:-1]):
            continue
        if not statically_known_true(x.shape[-1] == k) or x.stride(-1) != 1:
            continue
        if not value.is_contiguous():
            continue
        node.target = torch.ops.xinfer.projection_gemv.default
        node.args = (hidden, weight)
        replacements.append(dict(node=node.name, parameter=spec.target, k=k, n=n))
    exported.graph_module.graph.lint()
    exported.graph_module.recompile()
    exported.validate()
    return dict(replaced=len(replacements), projections=replacements)
