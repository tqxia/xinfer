"""DeltaNet adapter contracts and grouped-head integration with the HF module."""

import unittest
from unittest.mock import patch

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet,
    Qwen3_5TextConfig,
)

from compiler.common.tests.validation import bits_equal, row_errors
from compiler.qwen35.adapters.delta import DeltaCore, DeltaNet, install_delta
from compiler.qwen35.config import configure
from compiler.qwen35.kernels.recurrent import fused
from compiler.qwen35.model import TextCache
from compiler.qwen35.tests.block import BlockStep
from compiler.qwen35.tests.validate_fused import reference


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


def inputs(value_heads, device="meta"):
    channels = (32 + value_heads) * 128
    return (
        torch.empty(1, 1, channels, device=device),
        torch.empty(1, 1, value_heads, device=device),
        torch.empty(1, 1, value_heads, device=device),
        torch.empty(1, channels, 4, device=device),
        torch.empty(1, value_heads, 128, 128, device=device),
        torch.empty(channels, 4, device=device),
        torch.empty(value_heads, device=device),
        torch.empty(value_heads, device=device),
    )


class DeltaAdapterContract(unittest.TestCase):
    def test_installation_reuses_parameters_and_is_idempotent(self):
        for value_heads in (16, 32):
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

    def test_kernel_rejects_inconsistent_shapes_before_cuda(self):
        for value_heads in (16, 32):
            sample = inputs(value_heads)
            invalid = {
                "key_heads": (0, sample[0][..., : (16 + value_heads) * 128]),
                "channels": (0, sample[0][..., :-1]),
                "gates": (1, sample[1][..., :-1]),
                "conv": (3, sample[3][..., :3]),
                "state": (4, sample[4][:, :-1]),
                "decay": (6, sample[6][:-1]),
            }
            for name, (index, value) in invalid.items():
                broken = list(sample)
                broken[index] = value
                with self.subTest(value_heads=value_heads, field=name):
                    with self.assertRaisesRegex(ValueError, "Expected"):
                        fused(*broken)
            with self.subTest(value_heads=value_heads, field="device"):
                with self.assertRaisesRegex(ValueError, "Expected CUDA"):
                    fused(*sample)
        with self.assertRaisesRegex(ValueError, "Hk=16"):
            fused(*inputs(24))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DeltaAdapterNumerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure()

    def check_numerical(self, actual, expected, width, limit):
        error = row_errors(actual.float().reshape(-1, width), expected.float().reshape(-1, width))
        self.assertLessEqual(error["relative_l2"], limit)
        self.assertLessEqual(error["peak_relative"], limit)

    @torch.inference_mode()
    def test_hf_module_interface_and_functional_state(self):
        for value_heads in (16, 32):
            torch.manual_seed(35 + value_heads)
            block, config = make_block(value_heads, device="cuda")
            block.to(dtype=torch.bfloat16)
            # The standalone HF constructor leaves some parameters uninitialized.
            for parameter in block.parameters():
                parameter.normal_(std=0.02)
            block.norm.weight.fill_(1)
            block.A_log.fill_(-2)
            block.dt_bias.copy_(torch.linspace(-0.1, 0.1, value_heads, device="cuda"))
            hf = BlockStep(block)
            core = DeltaCore(block)
            adapter = DeltaNet(block)
            channels = (32 + value_heads) * 128
            conv = torch.randn(1, channels, 4, device="cuda", dtype=torch.bfloat16) * 0.1
            state = torch.randn(1, value_heads, 128, 128, device="cuda") * 0.1
            reference_conv, reference_state = conv.clone(), state.clone()
            position = 0
            for tokens in (1, 2, 97, 98, 1):
                with self.subTest(value_heads=value_heads, tokens=tokens, position=position):
                    hidden = torch.randn(
                        1, tokens, config.hidden_size, device="cuda", dtype=torch.bfloat16
                    )
                    original = (hidden.clone(), conv.clone(), state.clone())
                    expected, reference_conv, reference_state = hf(
                        hidden, reference_conv, reference_state
                    )
                    cache = TextCache(
                        ["linear_attention"],
                        torch.arange(position, position + tokens, device="cuda"),
                        [conv, state],
                    )
                    actual = adapter(hidden, cache_params=cache)
                    next_conv, next_state = cache.tensors()
                    direct = core(hidden, conv, state)
                    self.assertTrue(
                        all(
                            bits_equal(x, y)
                            for x, y in zip((actual, next_conv, next_state), direct)
                        )
                    )
                    self.assertTrue(
                        all(bits_equal(x, y) for x, y in zip((hidden, conv, state), original))
                    )
                    self.assertFalse(
                        any(
                            torch._C._is_alias_of(x, y)
                            for x in direct
                            for y in (hidden, conv, state)
                        )
                    )
                    self.assertTrue(all(bool(torch.isfinite(x).all()) for x in direct))
                    self.check_numerical(actual, expected, config.hidden_size, 0.05)
                    self.assertTrue(bits_equal(next_conv, reference_conv))
                    self.check_numerical(next_state, reference_state, 128 * 128, 0.05)
                    with self.assertRaisesRegex(ValueError, "unpadded"):
                        adapter(hidden, cache_params=cache, attention_mask=torch.empty(0))
                    conv, state = next_conv, next_state
                    position += tokens

    @torch.inference_mode()
    def test_distinct_value_heads_use_grouped_qk(self):
        torch.manual_seed(67)
        value_heads, channels = 32, 8192
        for tokens in (1, 98):
            with self.subTest(tokens=tokens):
                qkv = torch.randn(1, tokens, channels, device="cuda")
                # Distinct gates and states prevent the repeated key heads from
                # concealing an incorrect value-head mapping or shared state.
                b = torch.linspace(-2, 2, value_heads, device="cuda").expand(1, tokens, -1).clone()
                a = torch.linspace(-1, 1, value_heads, device="cuda").expand(1, tokens, -1).clone()
                conv = torch.randn(1, channels, 4, device="cuda") * 0.1
                state = torch.randn(1, value_heads, 128, 128, device="cuda") * 0.1
                weight = torch.randn(channels, 4, device="cuda") * 0.2
                A_log = torch.linspace(-3, -1, value_heads, device="cuda")
                dt_bias = torch.linspace(-0.2, 0.2, value_heads, device="cuda")
                sample = (qkv, b, a, conv, state, weight, A_log, dt_bias)
                actual = fused(*sample)
                expected = reference(*sample, chunked=tokens > 1)
                self.check_numerical(actual[0], expected[0], 128, 0.0001)
                self.assertTrue(bits_equal(actual[1], expected[1]))
                self.check_numerical(actual[2], expected[2], 128 * 128, 0.0001)


if __name__ == "__main__":
    unittest.main()
