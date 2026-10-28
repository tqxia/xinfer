"""CPU contracts for frozen linear packing and the exported tensor interface."""

import io
import json
import unittest

import torch
from torch import nn

from compiler.common.passes.pack_linears import pack_linears


class Projections(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.a = nn.Linear(8, 3, bias=False, dtype=dtype)
        self.b = nn.Linear(8, 5, bias=False, dtype=dtype)
        self.c = nn.Linear(8, 2, bias=False, dtype=dtype)
        self.requires_grad_(False)

    def forward(self, x, state):
        return self.a(x), self.b(x), self.c(x), state + 2


class FlattenedProjection(Projections):
    def forward(self, x, state):
        a, b, c, next_state = super().forward(x, state)
        return a.view(-1), b, c, next_state


class IndependentGroups(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(8, n, bias=False) for n in (3, 5, 2, 4))
        self.requires_grad_(False)

    def forward(self, x, y):
        return self.layers[0](x), self.layers[1](x), self.layers[2](y), self.layers[3](y)


class ExcludedPair(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.a = nn.Linear(8, 3, bias=kind == "bias")
        self.b = nn.Linear(8, 5, bias=False)
        self.requires_grad_(kind == "trainable")
        if kind == "mixed_dtype":
            self.b.to(torch.bfloat16)
        if kind == "shared_storage":
            storage = torch.randn(5, 8)
            self.a.weight = nn.Parameter(storage[:3], requires_grad=False)
            self.b.weight = nn.Parameter(storage, requires_grad=False)

    def forward(self, x, y):
        left = self.a(x)
        if self.kind == "single":
            right = y
        elif self.kind == "reused_weight":
            right = self.a(x)
        else:
            right = self.b(y if self.kind in ("different_inputs", "mixed_dtype") else x)
        extra = self.a.weight.sum() if self.kind == "weight_consumer" else x.sum()
        return left, right, extra


def linear_count(exported):
    return sum(n.target == torch.ops.aten.linear.default for n in exported.graph.nodes)


def dynamic_export(model):
    length = torch.export.Dim("tokens", min=2, max=17)
    dtype = next(model.parameters()).dtype
    return torch.export.export(model, (torch.randn(1, 4, 8, dtype=dtype), torch.randn(1, 4)),
                               dynamic_shapes=({1: length}, None), strict=True)


class PackingContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def assert_no_runtime_packing(self, exported):
        self.assertFalse(any(n.target == torch.ops.aten.cat.default for n in exported.graph.nodes))
        exported.validate()

    def assert_outputs_close(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertEqual(a.dtype, b.dtype)
            self.assertEqual(a.shape, b.shape)
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_dynamic_uneven_projections_preserve_interface_and_prune_weights(self):
        model = Projections().eval()
        original = {k: v.clone() for k, v in model.state_dict().items()}
        exported = dynamic_export(model)
        call_spec, ranges = exported.call_spec, dict(exported.range_constraints)
        self.assertEqual(linear_count(exported), 3)
        report = pack_linears(exported)
        self.assertEqual((report["linear_calls_before"], report["linear_calls_after"], report["saved_linears"]),
                         (3, 1, 2))
        self.assertEqual(len(report["groups"]), 1)
        json.dumps(report)  # The report is embedded in package provenance.
        self.assertEqual(linear_count(exported), 1)
        self.assertEqual(exported.call_spec, call_spec)
        self.assertEqual(dict(exported.range_constraints), ranges)
        self.assert_no_runtime_packing(exported)
        self.assertEqual(len(exported.state_dict), 1)
        self.assertFalse(set(exported.state_dict) & set(original))
        packed, = exported.state_dict.values()
        self.assertEqual(packed.shape, (10, 8))
        self.assertFalse(packed.requires_grad)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original[name]), "Original module weight changed")
        module = exported.module()
        for tokens in (2, 3, 8, 17):
            x, state = torch.randn(1, tokens, 8), torch.randn(1, 4)
            before = x.clone(), state.clone()
            self.assert_outputs_close(module(x, state), model(x, state))
            self.assertTrue(torch.equal(x, before[0]) and torch.equal(state, before[1]))
        # Reapplying the pass must not repack or introduce another parameter.
        code, targets = exported.graph_module.code, tuple(exported.state_dict)
        self.assertEqual(pack_linears(exported)["groups"], [])
        self.assertEqual(exported.graph_module.code, code)
        self.assertEqual(tuple(exported.state_dict), targets)

    def test_split_outputs_keep_layout_required_by_existing_view(self):
        model = FlattenedProjection().eval()
        exported = dynamic_export(model)
        pack_linears(exported)
        self.assert_no_runtime_packing(exported)
        for tokens in (2, 4, 17):
            x, state = torch.randn(1, tokens, 8), torch.randn(1, 4)
            self.assert_outputs_close(exported.module()(x, state), model(x, state))

    def test_bfloat16_results_round_after_accumulation(self):
        model = Projections(torch.bfloat16).eval()
        # Binary fractions make the FP32 sum exact, so this checks the BF16
        # output boundary without making a CPU BLAS reduction-order assumption.
        with torch.no_grad():
            for weight in model.parameters():
                values = (torch.arange(weight.numel()).reshape(weight.shape) % 9 - 4) / 16
                weight.copy_(values)
        exported = dynamic_export(model)
        pack_linears(exported)
        self.assert_no_runtime_packing(exported)
        x = ((torch.arange(56).reshape(1, 7, 8) % 11 - 5) / 16).bfloat16()
        state = torch.zeros(1, 4)
        actual = exported.module()(x, state)
        for index, layer in enumerate((model.a, model.b, model.c)):
            expected = torch.nn.functional.linear(x.double(), layer.weight.double()).bfloat16()
            self.assertEqual(actual[index].dtype, torch.bfloat16)
            self.assertTrue(torch.equal(actual[index], expected))
        self.assertTrue(torch.equal(actual[-1], state + 2))

    def test_groups_have_the_exact_same_input(self):
        model = IndependentGroups().eval()
        x, y = torch.randn(1, 4, 8), torch.randn(1, 3, 8)
        exported = torch.export.export(model, (x, y), strict=True)
        pack_linears(exported)
        self.assertEqual(linear_count(exported), 2)
        self.assertEqual(len(exported.state_dict), 2)
        self.assert_outputs_close(exported.module()(x, y), model(x, y))
        self.assert_no_runtime_packing(exported)

    def test_unsupported_or_shared_parameters_are_not_rewritten(self):
        for kind in ("single", "bias", "trainable", "different_inputs", "mixed_dtype",
                     "reused_weight", "shared_storage", "weight_consumer"):
            with self.subTest(kind=kind):
                model = ExcludedPair(kind).eval()
                x = torch.randn(4, 8)
                y = torch.randn(4, 8, dtype=torch.bfloat16 if kind == "mixed_dtype" else torch.float32)
                exported = torch.export.export(model, (x, y), strict=True)
                code, targets = exported.graph_module.code, tuple(exported.state_dict)
                pack_linears(exported)
                self.assertEqual(exported.graph_module.code, code)
                self.assertEqual(tuple(exported.state_dict), targets)
                self.assert_outputs_close(exported.module()(x, y), model(x, y))
                exported.validate()

    def test_packed_program_serializes_with_dynamic_state_interface(self):
        model = Projections().eval()
        exported = dynamic_export(model)
        pack_linears(exported)
        stream = io.BytesIO()
        torch.export.save(exported, stream)
        stream.seek(0)
        restored = torch.export.load(stream)
        restored.validate()
        self.assertEqual(restored.call_spec, exported.call_spec)
        self.assertEqual(linear_count(restored), 1)
        x, state = torch.randn(1, 9, 8), torch.randn(1, 4)
        self.assert_outputs_close(restored.module()(x, state), model(x, state))
        self.assert_no_runtime_packing(restored)


if __name__ == "__main__":
    unittest.main()
