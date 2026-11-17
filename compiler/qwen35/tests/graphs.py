"""Shared structural assertions for dense and MoE hybrid export graphs."""

import torch


def assert_hybrid_graph(test, exported, kind, layer_types, delta_operators):
    test.assertFalse(exported.graph_signature.buffers_to_mutate)
    targets = [node.target for node in exported.graph.nodes if node.op == "call_function"]
    full_layers = layer_types.count("full_attention")
    test.assertEqual(targets.count(torch.ops.xinfer.flash_attention.default), full_layers)
    test.assertEqual(targets.count(torch.ops.aten.index_copy_.default), 2 * full_layers)
    test.assertNotIn(torch.ops.aten.index_copy.default, targets)
    test.assertNotIn(torch.ops.aten.scaled_dot_product_attention.default, targets)
    counts = dict(cond=0, short=0, long=0)

    def visit(module):
        # Count call sites, even when multiple layers share a branch module.
        for node in module.graph.nodes:
            if node.target == torch.ops.higher_order.cond:
                predicate = node.args[0]
                test.assertIsInstance(predicate, torch.fx.Node)
                value = predicate.meta.get("val")
                test.assertIsInstance(value, torch.SymBool)
                test.assertNotIn(str(value), ("True", "False"))
                counts["cond"] += 1
                for branch in node.args[1:3]:
                    visit(module.get_submodule(branch.target))
            else:
                for branch, operator in delta_operators.items():
                    counts[branch] += str(node.target) == operator

    visit(exported.graph_module)
    layers = layer_types.count("linear_attention")
    conditions = layers if kind == "prefill" else 0
    test.assertEqual(
        sum(n.target == torch.ops.higher_order.cond for n in exported.graph.nodes), conditions
    )
    test.assertEqual(counts, dict(cond=conditions, short=layers, long=conditions))
    return targets
