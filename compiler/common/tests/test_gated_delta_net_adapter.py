"""Shared Qwen Gated DeltaNet adapter installation and geometry contracts."""

import unittest
from unittest.mock import patch

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet,
    Qwen3_5TextConfig,
)

from compiler.common.adapters.gated_delta_net import DeltaCore, DeltaNet, install_delta


def make_block(value_heads, device="meta"):
    config = Qwen3_5TextConfig(
        hidden_size=1024 if value_heads == 16 else 2560,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        linear_num_key_heads=16,
        linear_num_value_heads=value_heads,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    with torch.device(device):
        block = Qwen3_5GatedDeltaNet(config, layer_idx=0)
    return block.eval().requires_grad_(False), config


class DeltaAdapterContract(unittest.TestCase):
    def test_installation_reuses_parameters_and_is_idempotent(self):
        for value_heads in (16, 32, 48):
            with self.subTest(value_heads=value_heads):
                block, config = make_block(value_heads)
                core = DeltaCore(block)
                self.assertIs(core.block, block)
                self.assertEqual(
                    {id(parameter) for parameter in core.parameters()},
                    {id(parameter) for parameter in block.parameters()},
                )
                model = torch.nn.Module()
                model.config = config
                model.model = torch.nn.Module()
                layer = torch.nn.Module()
                layer.linear_attn = block
                model.model.layers = torch.nn.ModuleList([layer])
                parameters = {id(parameter) for parameter in model.parameters()}
                install_delta(model)
                adapter = layer.linear_attn
                self.assertIsInstance(adapter, DeltaNet)
                self.assertIs(adapter.block, block)
                self.assertEqual({id(parameter) for parameter in model.parameters()}, parameters)
                install_delta(model)
                self.assertIs(layer.linear_attn, adapter)

    def test_adapter_rejects_unsupported_geometry_and_convolution(self):
        block, _ = make_block(16)
        invalid = (
            ("num_k_heads", 8),
            ("num_v_heads", 24),
            ("head_k_dim", 64),
            ("head_v_dim", 64),
            ("conv_kernel_size", 3),
            ("activation", "relu"),
        )
        for name, value in invalid:
            with self.subTest(field=name), patch.object(block, name, value):
                with self.assertRaisesRegex(ValueError, "Fused DeltaNet requires"):
                    DeltaCore(block)
        with patch.object(block.conv1d, "bias", torch.nn.Parameter(torch.empty(1, device="meta"))):
            with self.assertRaisesRegex(ValueError, "bias-free"):
                DeltaCore(block)


if __name__ == "__main__":
    unittest.main()
