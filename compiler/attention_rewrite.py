"""Replace the known QwenStep full-cache SDPA pattern after torch.export.

This is deliberately a narrow pass, not a general SDPA-to-Flash converter.
The runtime guarantees contiguous absolute positions and a populated prefix.
"""

import torch
from torch.fx import Node

from attention_ops import load_attention_ops


def _call(node, target):
    if not isinstance(node, Node) or node.op != "call_function" or node.target != target:
        raise ValueError(f"Unsupported attention pattern: expected {target}, got {node}")
    return node


def _cache_before_repeat(node, positions):
    reshape = _call(node, torch.ops.aten.reshape.default)
    expand = _call(reshape.args[0], torch.ops.aten.expand.default)
    unsqueeze = _call(expand.args[0], torch.ops.aten.unsqueeze.default)
    if unsqueeze.args[1] != 2 or tuple(expand.args[1]) != (1, 8, 2, 1024, 128) or tuple(reshape.args[1]) != (1, 16, 1024, 128):
        raise ValueError("Unsupported GQA expansion")
    update = _call(unsqueeze.args[0], torch.ops.aten.index_copy_.default)
    if update.args[1] != 2 or update.args[2] is not positions:
        raise ValueError("Cache update must use the shared absolute positions")
    return update


def _check_mask(mask, positions):
    # HF may remove fully masked rows. No row is fully masked under our position
    # contract, but verify this exact adjustment rather than discard arbitrary masks.
    if isinstance(mask, Node) and mask.target == torch.ops.aten.alias.default:
        mask = mask.args[0]
    if isinstance(mask, Node) and mask.target == torch.ops.aten.mul.Tensor:
        base, adjustment = mask.args
        invert = _call(adjustment, torch.ops.aten.bitwise_not.default)
        all_rows = _call(invert.args[0], torch.ops.aten.all.dim)
        eq = _call(all_rows.args[0], torch.ops.aten.eq.Scalar)
        if all_rows.args[1:] != (-1, True) or eq.args != (base, torch.finfo(torch.bfloat16).min):
            raise ValueError("Unsupported mask row adjustment")
        mask = base
    outer = _call(mask, torch.ops.aten.unsqueeze.default)
    inner = _call(outer.args[0], torch.ops.aten.unsqueeze.default)
    fill = _call(inner.args[0], torch.ops.aten.masked_fill.Scalar)
    zero = _call(fill.args[0], torch.ops.aten.zeros_like.default)
    gt = _call(fill.args[1], torch.ops.aten.gt.Tensor)
    columns = _call(gt.args[0], torch.ops.aten.unsqueeze.default)
    rows = _call(gt.args[1], torch.ops.aten.unsqueeze.default)
    arange = _call(columns.args[0], torch.ops.aten.arange.default)
    if (outer.args[1] != 1 or inner.args[1] != 0 or fill.args[2] != torch.finfo(torch.bfloat16).min
            or zero.args[0] is not gt or zero.kwargs.get("dtype") != torch.bfloat16
            or columns.args[1] != 0 or rows.args != (positions, 1) or arange.args != (1024,)):
        raise ValueError("Expected the QwenStep absolute-position causal mask")


def rewrite_attention(exported, expected_layers):
    load_attention_ops()
    gm = exported.module()
    graph = gm.graph
    placeholders = {n.target: n for n in graph.nodes if n.op == "placeholder"}
    positions, extent = placeholders["cache_position"], placeholders["kv_extent"]
    sites = [n for n in graph.nodes if n.op == "call_function" and
             n.target == torch.ops.aten.scaled_dot_product_attention.default]
    if len(sites) != expected_layers:
        raise ValueError(f"Expected {expected_layers} SDPA sites, found {len(sites)}")
    updates = [n for n in graph.nodes if n.op == "call_function" and
               n.target == torch.ops.aten.index_copy_.default]
    if len(updates) != 2 * expected_layers:
        raise ValueError("Unexpected cache mutation count")
    replacements = []
    # Validate every site before changing the graph.
    for node in sites:
        if len(node.args) != 4 or set(node.kwargs) != {"scale"}:
            raise ValueError("Only explicit-mask, zero-dropout Qwen SDPA is supported")
        q, k, v, mask = node.args
        _check_mask(mask, positions)
        k, v = _cache_before_repeat(k, positions), _cache_before_repeat(v, positions)
        replacements.append((node, q, k, v))
    for node, q, k, v in replacements:
        with graph.inserting_before(node):
            replacement = graph.call_function(torch.ops.xinfer.flash_attention.default,
                                               (q, k, v, extent, node.kwargs["scale"]))
            # Preserve provenance, not stale output layout metadata. Re-export
            # invokes the fake kernel and reconstructs tensor metadata/signatures.
            replacement.meta = {key: value for key, value in node.meta.items()
                                if key not in ("val", "tensor_meta")}
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
    # Keep all mutation/side-effect nodes even when their return value is unused.
    graph.eliminate_dead_code(is_impure_node=lambda n: n.is_impure() or n in updates)
    assert all(n in graph.nodes for n in updates)
    graph.lint()
    gm.recompile()
    return gm, {"pass": "qwen_sdpa_to_custom_flash_v1", "replacements": len(sites),
                "cache_updates_preserved": len(updates)}
