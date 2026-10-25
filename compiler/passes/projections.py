"""Pack fixed Qwen3-0.6B inference projections once during export.

Weights are concatenated while transforming the module, never in its forward
graph. Linear outputs retain their BF16 boundary; norms, RoPE, SiLU and cache
updates are left in the graph. This is deliberately not a general linear fuser.
"""

import re

import torch
from torch.fx import Node


PACKING_MODES = ("none", "qkv", "gate-up", "all")
_GROUPS = {
    "qkv": ("self_attn", ("q_proj", "k_proj", "v_proj"), (2048, 1024, 1024)),
    "gate-up": ("mlp", ("gate_proj", "up_proj"), (3072, 3072)),
}
_WEIGHT = re.compile(r"(?:^|\.)layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.weight$")


def _attribute(module, target):
    for part in target.split("."):
        module = getattr(module, part)
    return module


def pack_projections(exported, expected_layers, mode="all"):
    if mode not in PACKING_MODES[1:]:
        raise ValueError(f"Unsupported projection packing mode: {mode}")
    gm = exported.module()
    graph = gm.graph
    linears = [n for n in graph.nodes if n.target == torch.ops.aten.linear.default]
    updates = [n for n in graph.nodes if n.target == torch.ops.aten.index_copy_.default]
    sites = {}
    order = {node: i for i, node in enumerate(graph.nodes)}
    for node in linears:
        weight = node.args[1]
        if not isinstance(weight, Node) or weight.op != "get_attr":
            continue
        match = _WEIGHT.search(weight.target)
        if match:
            key = (int(match[1]), match[2], match[3])
            if key in sites:
                raise ValueError(f"Repeated projection: {key}")
            sites[key] = node
    groups = tuple(_GROUPS) if mode == "all" else (mode,)
    plans = []
    # Validate the entire transformation before altering any graph or attribute.
    for kind in groups:
        scope, names, sizes = _GROUPS[kind]
        found = {layer for layer, parent, name in sites if parent == scope and name in names}
        if found != set(range(expected_layers)):
            raise ValueError(f"Unexpected {kind} layer set: {found}")
        for layer in range(expected_layers):
            keys = [(layer, scope, name) for name in names]
            if any(key not in sites for key in keys):
                raise ValueError(f"Incomplete {kind} projections in layer {layer}")
            nodes = [sites[key] for key in keys]
            first, last = min(order[n] for n in nodes), max(order[n] for n in nodes)
            # Export marks metadata assertions impure so DCE retains them. They
            # do not change tensors or arithmetic state; keep them in place but
            # allow the packed read-only linear to precede them.
            if any(n.is_impure() and n.target != torch.ops.aten._assert_tensor_metadata.default
                   for n, index in order.items() if first < index < last):
                raise ValueError("Cannot move a projection across a graph side effect")
            activation = nodes[0].args[0]
            weights = []
            for node, size in zip(nodes, sizes):
                if (node.kwargs or len(node.args) not in (2, 3) or
                        (len(node.args) == 3 and node.args[2] is not None)):
                    raise ValueError("Only bias-free positional linear calls can be packed")
                if node.args[0] is not activation:
                    raise ValueError(f"{kind} projections must share the same input")
                attr = node.args[1]
                if set(attr.users) != {node}:
                    raise ValueError("Packed projection weights must have one graph consumer")
                weight = _attribute(gm, attr.target)
                if weight.shape != (size, 1024) or weight.dtype != torch.bfloat16:
                    raise ValueError(f"Unsupported projection weight: {attr.target}")
                weights.append(weight)
            if any(w.device != weights[0].device for w in weights):
                raise ValueError("Packed weights must share a device")
            buffer_name = f"packed_layer_{layer}_{kind.replace('-', '_')}_weight"
            if hasattr(gm, buffer_name):
                raise ValueError(f"Packed buffer already exists: {buffer_name}")
            plans.append((buffer_name, nodes, weights, sizes))

    packed_bytes = 0
    removed = []
    for buffer_name, nodes, weights, sizes in plans:
        # This eager cat runs only during export. The new module owns a frozen
        # contiguous buffer; original model tensors are neither edited nor aliased.
        packed = torch.cat([w.detach() for w in weights], dim=0).contiguous()
        gm.register_buffer(buffer_name, packed)
        packed_bytes += packed.numel() * packed.element_size()
        with graph.inserting_before(min(nodes, key=order.__getitem__)):
            weight_node = graph.get_attr(buffer_name)
            combined = graph.call_function(torch.ops.aten.linear.default,
                                           (nodes[0].args[0], weight_node))
            combined.meta = {key: value for key, value in nodes[0].meta.items()
                             if key not in ("val", "tensor_meta")}
            offset = 0
            for node, size in zip(nodes, sizes):
                part = graph.call_function(torch.ops.aten.slice.Tensor,
                                           (combined, -1, offset, offset + size))
                part.meta = {key: value for key, value in node.meta.items()
                             if key not in ("val", "tensor_meta")}
                node.replace_all_uses_with(part)
                removed.append(node.args[1].target)
                offset += size
        for node in nodes:
            graph.erase_node(node)
    graph.eliminate_dead_code(is_impure_node=lambda n: n.is_impure() or n in updates)
    assert all(n in graph.nodes for n in updates)
    # Do not serialize redundant original weights alongside the packed buffers.
    live_attrs = {n.target for n in graph.nodes if n.op == "get_attr"}
    for target in removed:
        assert target not in live_attrs
        parent, _, leaf = target.rpartition(".")
        delattr(_attribute(gm, parent) if parent else gm, leaf)
    graph.lint()
    gm.recompile()
    after = sum(n.target == torch.ops.aten.linear.default for n in graph.nodes)
    reduction = expected_layers * sum(len(_GROUPS[k][1]) - 1 for k in groups)
    assert after == len(linears) - reduction
    return gm, dict(pass_name="qwen_pack_projections_v1", mode=mode,
                    groups=len(plans), linear_calls_before=len(linears), linear_calls_after=after,
                    packed_weight_bytes=packed_bytes, cache_updates_preserved=len(updates),
                    runtime_weight_packing=False, removed_weight_attributes=removed)
