"""CPU contracts for zero-copy FP8 adapters and shared hybrid-model attention."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers.integrations.finegrained_fp8 import FP8Experts as HFExperts, FP8Linear as HFLinear
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5TextConfig,
    apply_rotary_pos_emb as dense_rotary,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention,
    Qwen3_5MoeTextConfig,
    apply_rotary_pos_emb as moe_rotary,
)

from compiler.qwen36 import adapters
from compiler.qwen35.adapters import attention


def linear(**kwargs):
    module = HFLinear(128, 128, block_size=(128, 128), **kwargs)
    module.weight_scale_inv = torch.nn.Parameter(torch.ones(1, 1, dtype=torch.bfloat16))
    return module.eval().requires_grad_(False)


def experts():
    config = Qwen3_5MoeTextConfig(
        hidden_size=128, moe_intermediate_size=128, num_experts=4, num_experts_per_tok=2
    )
    module = HFExperts(config, block_size=(128, 128))
    module.gate_up_proj_scale_inv = torch.nn.Parameter(torch.ones(4, 2, 1, dtype=torch.bfloat16))
    module.down_proj_scale_inv = torch.nn.Parameter(torch.ones(4, 1, 1, dtype=torch.bfloat16))
    return module.eval().requires_grad_(False)


class FP8Adapters(unittest.TestCase):
    def test_linear_passes_original_parameters_and_preserves_output_contract(self):
        original = linear()
        module = adapters.FP8Linear(original)
        x = torch.ones(1, 2, 128, dtype=torch.bfloat16)
        expected = torch.empty_like(x)
        with patch.object(adapters.fp8, "linear", return_value=expected) as operation:
            self.assertIs(module(x), expected)
        self.assertIs(module.original, original)
        self.assertEqual(operation.call_args.args, (x, original.weight, original.weight_scale_inv))
        self.assertEqual(expected.dtype, torch.bfloat16)
        self.assertFalse(module.training)
        self.assertEqual(
            {id(value) for value in module.parameters()},
            {id(value) for value in original.parameters()},
        )

    def test_experts_pass_routes_and_original_weight_bank_without_copies(self):
        original = experts()
        module = adapters.FP8Experts(original)
        x = torch.ones(3, 128, dtype=torch.bfloat16)
        ids = torch.tensor([[0, 1], [2, 0], [1, 3]])
        weights = torch.ones(3, 2, dtype=torch.bfloat16) / 2
        expected = torch.empty_like(x)
        with patch.object(adapters.fp8, "experts", return_value=expected) as operation:
            self.assertIs(module(x, ids, weights), expected)
        for actual, value in zip(
            operation.call_args.args,
            (
                x,
                ids,
                weights,
                original.gate_up_proj,
                original.gate_up_proj_scale_inv,
                original.down_proj,
                original.down_proj_scale_inv,
            ),
        ):
            self.assertIs(actual, value)
        self.assertEqual(expected.shape, x.shape)
        self.assertEqual(
            {id(value) for value in module.parameters()},
            {id(value) for value in original.parameters()},
        )

    def test_installs_all_projection_paths_and_leaves_bf16_modules_intact(self):
        model = torch.nn.Module()
        for name in ("self_attn", "linear_attn", "shared_expert"):
            setattr(model, name, torch.nn.Module())
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(model.self_attn, name, linear())
        for name in ("in_proj_qkv", "in_proj_z", "out_proj"):
            setattr(model.linear_attn, name, linear())
        for name in ("in_proj_a", "in_proj_b"):
            setattr(model.linear_attn, name, torch.nn.Linear(128, 16, bias=False).bfloat16())
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(model.shared_expert, name, linear())
        model.experts = experts()
        model.router = torch.nn.Linear(128, 4, bias=False).bfloat16()
        model.shared_expert_gate = torch.nn.Linear(128, 1, bias=False).bfloat16()
        model.eval()
        untouched = (
            model.router,
            model.shared_expert_gate,
            model.linear_attn.in_proj_a,
            model.linear_attn.in_proj_b,
        )
        before = {id(value): value.data_ptr() for value in model.parameters()}
        with patch.object(adapters.fp8, "load_fp8_kernels") as preload:
            self.assertEqual(adapters.install_fp8(model), dict(linear=10, experts=1))
            self.assertEqual(adapters.install_fp8(model), dict(linear=0, experts=0))
            preload.assert_called_once()
        self.assertEqual({id(value): value.data_ptr() for value in model.parameters()}, before)
        self.assertEqual(
            untouched,
            (
                model.router,
                model.shared_expert_gate,
                model.linear_attn.in_proj_a,
                model.linear_attn.in_proj_b,
            ),
        )
        self.assertIsInstance(model.shared_expert.gate_proj, adapters.FP8Linear)
        self.assertIsInstance(model.shared_expert.up_proj, adapters.FP8Linear)
        self.assertIsInstance(model.experts, adapters.FP8Experts)

    def test_invalid_module_or_failed_kernel_preload_does_not_partially_install(self):
        for failure in ("format", "preload"):
            with self.subTest(failure=failure):
                model = torch.nn.Module()
                model.good = linear()
                model.bad = linear(activation_scheme="static") if failure == "format" else linear()
                original = (model.good, model.bad)
                with patch.object(
                    adapters.fp8, "load_fp8_kernels", side_effect=RuntimeError("preload failed")
                ) as preload:
                    with self.assertRaisesRegex(
                        (ValueError, RuntimeError), "dynamic|preload failed"
                    ):
                        adapters.install_fp8(model)
                    self.assertEqual((model.good, model.bad), original)
                    if failure == "format":
                        preload.assert_not_called()

    def test_rejects_unsupported_weight_layout_gate_and_training(self):
        original = linear(has_bias=True)
        with self.assertRaisesRegex(ValueError, "bias-free"):
            adapters.FP8Linear(original)
        original = linear()
        original.weight = torch.nn.Parameter(original.weight.bfloat16())
        with self.assertRaisesRegex(ValueError, "E4M3"):
            adapters.FP8Linear(original)
        for attribute, value in (
            ("is_transposed", True),
            ("has_gate", False),
            ("_is_expert_parallel", True),
            ("limit", 1.0),
        ):
            with self.subTest(attribute=attribute):
                original = experts()
                setattr(original, attribute, value)
                with self.assertRaisesRegex(ValueError, "concatenated SiLU"):
                    adapters.FP8Experts(original)
        module = adapters.FP8Linear(linear()).train()
        with patch.object(adapters.fp8, "linear") as operation:
            with self.assertRaisesRegex(ValueError, "inference only"):
                module(torch.empty(1, 128, dtype=torch.bfloat16))
            operation.assert_not_called()


class SharedAttention(unittest.TestCase):
    def test_dense_and_moe_attention_share_adapter_without_parameter_changes(self):
        for configuration, implementation in (
            (Qwen3_5TextConfig, Qwen3_5Attention),
            (Qwen3_5MoeTextConfig, Qwen3_5MoeAttention),
        ):
            with self.subTest(implementation=implementation.__name__):
                config = configuration(
                    hidden_size=128,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    head_dim=64,
                    num_hidden_layers=1,
                    layer_types=["full_attention"],
                )
                original = implementation(config, layer_idx=0).eval()
                model = torch.nn.Module()
                model.config = config
                model.model = torch.nn.Module()
                layer = torch.nn.Module()
                layer.self_attn = original
                model.model.layers = torch.nn.ModuleList([layer])
                before = {id(value): value.data_ptr() for value in model.parameters()}
                with patch.object(attention, "load_attention_ops"):
                    attention.install_attention(model)
                    replacement = layer.self_attn
                    attention.install_attention(model)
                self.assertIsInstance(replacement, attention.FlashAttention)
                self.assertIs(replacement.original, original)
                self.assertIs(layer.self_attn, replacement)
                self.assertEqual(
                    {id(value): value.data_ptr() for value in model.parameters()}, before
                )
        with self.assertRaisesRegex(ValueError, "full attention"):
            attention.FlashAttention(SimpleNamespace())

    def test_pinned_hf_dense_and_moe_rotary_math_matches_for_partial_rotation(self):
        query = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(1, 2, 3, 8).bfloat16()
        key = query[:, :1].clone()
        cos = torch.randn(1, 3, 4).bfloat16()
        sin = torch.randn(1, 3, 4).bfloat16()
        dense = dense_rotary(query, key, cos, sin)
        moe = moe_rotary(query, key, cos, sin)
        self.assertTrue(all(torch.equal(left, right) for left, right in zip(dense, moe)))
        self.assertTrue(torch.equal(moe[0][..., 4:], query[..., 4:]))


if __name__ == "__main__":
    unittest.main()
