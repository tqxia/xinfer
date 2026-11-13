"""CPU fake-tensor contracts for the experimental projection rewrite."""

import unittest

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from compiler.common.passes.projection_gemv import replace_projection_gemv


class Projection(nn.Module):
    def __init__(self, k=2048, n=1024, *, dtype=torch.bfloat16, device="cuda", bias=False):
        super().__init__()
        self.linear = nn.Linear(k, n, bias=bias, dtype=dtype, device=device)
        self.requires_grad_(False)

    def forward(self, x, state):
        return self.linear(x), state + 2


class ProjectionRewrite(unittest.TestCase):
    def export(self, *, tokens=1, dtype=torch.bfloat16, device="cuda", **kwargs):
        with FakeTensorMode():
            model = Projection(dtype=dtype, device=device, **kwargs)
            x = torch.empty(1, tokens, model.linear.in_features, dtype=dtype, device=device)
            state = torch.empty(3, dtype=torch.float32, device=device)
            return torch.export.export(model, (x, state), strict=True)

    def test_rewrite_preserves_parameters_signature_and_metadata(self):
        exported = self.export()
        signature = str(exported.graph_signature)
        parameters = dict(exported.state_dict)
        graph_nodes = list(exported.graph.nodes)
        node = next(n for n in graph_nodes if n.target == torch.ops.aten.linear.default)
        metadata = node.meta
        report = replace_projection_gemv(exported)
        self.assertEqual(report["replaced"], 1)
        self.assertEqual(node.target, torch.ops.xinfer.projection_gemv.default)
        self.assertIs(node.meta, metadata)
        self.assertEqual(list(exported.graph.nodes), graph_nodes)
        self.assertEqual(str(exported.graph_signature), signature)
        self.assertEqual(set(exported.state_dict), set(parameters))
        for name, value in parameters.items():
            self.assertIs(exported.state_dict[name], value)
        self.assertEqual(replace_projection_gemv(exported)["replaced"], 0)

    def test_prefill_dtype_cpu_bias_and_unprofiled_shapes_are_unchanged(self):
        for kwargs in (
            dict(tokens=2),
            dict(dtype=torch.float32),
            dict(device="cpu"),
            dict(bias=True),
            dict(k=1024, n=248320),
            dict(k=1024, n=4096),
        ):
            with self.subTest(kwargs=kwargs):
                exported = self.export(**kwargs)
                code = exported.graph_module.code
                self.assertEqual(replace_projection_gemv(exported)["replaced"], 0)
                self.assertEqual(exported.graph_module.code, code)


if __name__ == "__main__":
    unittest.main()
