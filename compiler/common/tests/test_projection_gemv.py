"""CPU fake-tensor contracts for the small-batch projection rewrite."""

import unittest

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental import _config as symbolic_config

from compiler.common.passes.projection_gemv import replace_projection_gemv


class Projection(nn.Module):
    def __init__(self, k=2048, n=1024, *, dtype=torch.bfloat16, device="cuda", bias=False):
        super().__init__()
        self.linear = nn.Linear(k, n, bias=bias, dtype=dtype, device=device)
        self.requires_grad_(False)

    def forward(self, x, state):
        return self.linear(x), state + 2


class ProjectionRewrite(unittest.TestCase):
    def export(
        self,
        *,
        batch=1,
        tokens=1,
        rank=3,
        dynamic_batch=None,
        dynamic_tokens=None,
        dtype=torch.bfloat16,
        device="cuda",
        **kwargs,
    ):
        with FakeTensorMode(), symbolic_config.patch(backed_size_oblivious=True):
            model = Projection(dtype=dtype, device=device, **kwargs)
            shape = (
                (batch, model.linear.in_features)
                if rank == 2
                else (batch, tokens, model.linear.in_features)
            )
            x = torch.empty(*shape, dtype=dtype, device=device)
            state = torch.empty(3, dtype=torch.float32, device=device)
            x_shapes = {}
            if dynamic_batch is not None:
                x_shapes[0] = torch.export.Dim("batch", min=dynamic_batch[0], max=dynamic_batch[1])
            if dynamic_tokens is not None:
                x_shapes[1] = torch.export.Dim(
                    "tokens", min=dynamic_tokens[0], max=dynamic_tokens[1]
                )
            dynamic_shapes = (x_shapes, None) if x_shapes else None
            return torch.export.export(
                model, (x, state), strict=True, dynamic_shapes=dynamic_shapes
            )

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

    def test_rewrite_matches_static_and_dynamic_small_batches(self):
        for kwargs in (
            dict(batch=4, tokens=1, rank=3),
            dict(batch=8, rank=2),
            dict(batch=2, tokens=1, rank=3, dynamic_batch=(1, 8)),
            dict(batch=2, rank=2, dynamic_batch=(1, 8)),
        ):
            with self.subTest(kwargs=kwargs):
                exported = self.export(**kwargs)
                report = replace_projection_gemv(exported)
                self.assertEqual(report["replaced"], 1)
                targets = [
                    node.target for node in exported.graph.nodes if node.op == "call_function"
                ]
                self.assertIn(torch.ops.xinfer.projection_gemv.default, targets)
                self.assertNotIn(torch.ops.aten.linear.default, targets)

    def test_prefill_large_batch_dtype_cpu_bias_and_unprofiled_shapes_are_unchanged(self):
        for kwargs in (
            dict(tokens=2),
            dict(batch=2, tokens=4, dynamic_batch=(1, 8), dynamic_tokens=(2, 512)),
            dict(batch=9, tokens=1),
            dict(batch=2, tokens=1, dynamic_batch=(1, 9)),
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
