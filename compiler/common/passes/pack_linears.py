"""Pack independent frozen linear projections before Inductor decomposition.

Weights are concatenated once here, then stored as lifted parameters. There is
no weight concatenation in the inference graph. User inputs, outputs and state
ABI stay intact; only the graph's frozen parameter inputs change.
"""

from collections import Counter, defaultdict
import operator

import torch
from torch.export.graph_signature import InputKind


def _storage_key(tensor):
    # Skip unsupported tensor subclasses/storage rather than changing their rules.
    if type(tensor) not in (torch.Tensor, torch.nn.Parameter) or tensor.layout != torch.strided:
        return None
    return tensor.device, tensor.untyped_storage().data_ptr()


def pack_linears(exported):
    """Mutate an ExportedProgram and return deterministic packing provenance.

    Only bias-free aten.linear calls sharing the same input are candidates.
    Their plain, frozen weight parameters must have exclusive storage and one
    graph consumer. Shared/tied weights, buffers and other consumers are left
    alone. Contiguous outputs preserve downstream views for dynamic prefill.
    """
    graph = exported.graph_module.graph
    signature = exported.graph_signature
    parameters = {s.arg.name: s for s in signature.input_specs if s.kind == InputKind.PARAMETER}
    storage_counts = Counter(_storage_key(t) for t in exported.state_dict.values())
    target_counts = Counter(s.target for s in parameters.values())
    linears = [n for n in graph.nodes if n.op == "call_function" and n.target == torch.ops.aten.linear.default]
    candidates = defaultdict(list)
    for node in linears:
        if len(node.args) not in (2, 3) or node.kwargs or (len(node.args) == 3 and node.args[2] is not None):
            continue
        hidden, weight = node.args[:2]
        if not isinstance(hidden, torch.fx.Node) or not isinstance(weight, torch.fx.Node):
            continue
        spec = parameters.get(weight.name)
        if spec is None or len(weight.users) != 1 or target_counts[spec.target] != 1:
            continue
        value = exported.state_dict[spec.target]
        key = _storage_key(value)
        if (key is None or storage_counts[key] != 1 or value.requires_grad or value.ndim != 2
                or min(value.shape) <= 0 or "val" not in hidden.meta or "val" not in weight.meta):
            continue
        candidates[hidden, value.shape[1], value.dtype, value.device].append(node)

    groups = []
    removed_inputs = set()
    for nodes in candidates.values():
        if len(nodes) < 2:
            continue
        weights = [node.args[1] for node in nodes]
        specs = [parameters[weight.name] for weight in weights]
        sources = [spec.target for spec in specs]
        values = [exported.state_dict[name] for name in sources]
        widths = [value.shape[0] for value in values]
        name = f"_packed_linear_{len(groups)}"
        while name in exported.state_dict:
            name += "_"
        packed = torch.nn.Parameter(torch.cat(values, dim=0), requires_grad=False)
        exported.state_dict[name] = packed
        for source in sources:
            del exported.state_dict[source]
        specs[0].target = name
        fake_weight = weights[0].meta["val"].new_empty(packed.shape)
        weights[0].meta = dict(weights[0].meta, val=fake_weight)
        weights[0].meta.pop("tensor_meta", None)
        hidden = nodes[0].args[0]
        with fake_weight.fake_mode:
            fake_hidden = hidden.meta["val"]
            fake_output = fake_hidden.new_empty((*fake_hidden.shape[:-1], sum(widths)))
            fake_parts = torch.ops.aten.split_with_sizes.default(fake_output, widths, -1)
        with graph.inserting_before(nodes[0]):
            linear = graph.call_function(torch.ops.aten.linear.default, (hidden, weights[0]))
            linear.meta = dict(nodes[0].meta, val=fake_output)
            linear.meta.pop("tensor_meta", None)
            split = graph.call_function(torch.ops.aten.split_with_sizes.default, (linear, widths, -1))
            split.meta["val"] = fake_parts
        for index, node in enumerate(nodes):
            with graph.inserting_before(node):
                part = graph.call_function(operator.getitem, (split, index))
                part.meta["val"] = fake_parts[index]
            # Retain the original name: output specs/module-call signatures may
            # refer to it directly. Restore the old linear's contiguous contract.
            node.target = torch.ops.aten.contiguous.default
            node.args = (part,)
            node.kwargs = {}
            # The original linear metadata already has this shape/dtype and
            # contiguous layout; keep it instead of evaluating symbolic linear.
            node.meta.pop("tensor_meta", None)
        for weight in weights[1:]:
            assert not weight.users
            removed_inputs.add(weight.name)
            graph.erase_node(weight)
        groups.append(dict(packed_parameter=name, source_parameters=sources,
                           output_widths=widths, input=hidden.name, outputs=[node.name for node in nodes]))

    signature.input_specs[:] = [s for s in signature.input_specs if s.arg.name not in removed_inputs]
    graph.lint()
    exported.graph_module.recompile()
    exported.validate()
    after = sum(n.op == "call_function" and n.target == torch.ops.aten.linear.default for n in graph.nodes)
    return dict(groups=groups, linear_calls_before=len(linears), linear_calls_after=after,
                saved_linears=len(linears) - after)
