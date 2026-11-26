"""Shared DeltaNet adapter integration with the Qwen HF module and HybridCache."""

import unittest

import torch

from compiler.common.adapters.gated_delta_net import DeltaNet
from compiler.common.kernels.gated_delta_net.deltanet import load_deltanet_ops
from compiler.common.tests.tensors import bits_equal, row_errors
from compiler.common.tests.test_gated_delta_net_adapter import make_block
from compiler.qwen35.config import configure
from compiler.qwen35.model import HybridCache


class TensorState:
    """The small part of HF's cache protocol used by this one block.

    Zero tensors represent a fresh request. Clone convolution storage because
    HF's decode convolution mutates it while the reference helper stays functional.
    """

    def __init__(self, conv, recurrent):
        self.layers = [self]
        self.record_past = False
        self.conv_states = [conv.clone()]
        self.recurrent_states = [recurrent]

    def has_previous_state(self, layer_idx, state_idx=0):
        return True

    def update_conv_state(self, value, layer_idx, conv_kernel_size):
        combined = torch.cat((self.conv_states[0], value), dim=-1)
        self.conv_states[0] = combined[..., -conv_kernel_size:].contiguous()
        return combined

    def update_recurrent_state(self, value, layer_idx):
        self.recurrent_states[0] = value
        return value


class BlockStep(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden, conv, recurrent):
        cache = TensorState(conv, recurrent)
        output = self.block(hidden, cache_params=cache)
        return output, cache.conv_states[0], cache.recurrent_states[0]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DeltaAdapterNumerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure()
        load_deltanet_ops()

    def check_numerical(self, actual, expected, width, limit):
        error = row_errors(actual.float().reshape(-1, width), expected.float().reshape(-1, width))
        self.assertLessEqual(error["relative_l2"], limit)
        self.assertLessEqual(error["peak_relative"], limit)

    @torch.inference_mode()
    def test_hf_module_interface_and_inplace_state(self):
        for value_heads in (16, 32, 48):
            torch.manual_seed(35 + value_heads)
            block, config = make_block(value_heads, device="cuda")
            block.to(dtype=torch.bfloat16)
            # The standalone HF constructor leaves some parameters uninitialized.
            for parameter in block.parameters():
                parameter.normal_(std=0.02)
            block.norm.weight.fill_(1)
            block.A_log.fill_(-2)
            block.dt_bias.zero_()
            hf = BlockStep(block)
            adapter = DeltaNet(block)
            channels = (32 + value_heads) * 128
            conv = torch.zeros(1, channels, 4, device="cuda", dtype=torch.bfloat16)
            state = torch.zeros(1, value_heads, 128, 128, device="cuda")
            reference_conv, reference_state = conv.clone(), state.clone()
            position = 0
            for tokens in (1, 97, 98, 1):
                with self.subTest(value_heads=value_heads, tokens=tokens):
                    hidden = torch.randn(
                        1, tokens, config.hidden_size, device="cuda", dtype=torch.bfloat16
                    )
                    conv_ptr, state_ptr = conv.data_ptr(), state.data_ptr()
                    expected, reference_conv, reference_state = hf(
                        hidden, reference_conv, reference_state
                    )
                    cache = HybridCache(
                        ["linear_attention"],
                        torch.arange(position, position + tokens, device="cuda")[None, :],
                        (conv, state),
                        torch.tensor([position + tokens], dtype=torch.int32),
                        torch.tensor([tokens], dtype=torch.int32),
                    )
                    actual = adapter(hidden, cache_params=cache)
                    self.assertEqual(conv.data_ptr(), conv_ptr)
                    self.assertEqual(state.data_ptr(), state_ptr)
                    self.check_numerical(actual, expected, config.hidden_size, 0.05)
                    self.assertTrue(bits_equal(conv, reference_conv))
                    self.check_numerical(state, reference_state, 128 * 128, 0.05)
                    with self.assertRaisesRegex(ValueError, "attention masks"):
                        adapter(hidden, cache_params=cache, attention_mask=torch.empty(0))
                    position += tokens

    @torch.inference_mode()
    def test_checkpoint_native_fp32_decay_and_norm(self):
        for value_heads in (16, 32, 48):
            torch.manual_seed(135 + value_heads)
            block, config = make_block(value_heads, device="cuda")
            block.to(dtype=torch.bfloat16)
            for parameter in block.parameters():
                parameter.normal_(std=0.02)
            # Keep checkpoint-native FP32 decay and gated norm weights with non-BF16 values.
            block.A_log = torch.nn.Parameter(
                torch.linspace(-2.5, 0.5, value_heads, device="cuda", dtype=torch.float32),
                requires_grad=False,
            )
            block.norm.weight = torch.nn.Parameter(
                torch.linspace(0.5, 1.5, 128, device="cuda", dtype=torch.float32),
                requires_grad=False,
            )
            block.dt_bias.copy_(torch.linspace(-0.1, 0.1, value_heads, device="cuda"))
            hf = BlockStep(block)
            adapter = DeltaNet(block)
            channels = (32 + value_heads) * 128
            conv = torch.randn(1, channels, 4, device="cuda", dtype=torch.bfloat16) * 0.1
            state = torch.randn(1, value_heads, 128, 128, device="cuda") * 0.1
            ref_conv, ref_state = conv.clone(), state.clone()
            position = 0
            for tokens in (1, 97, 98):
                with self.subTest(value_heads=value_heads, tokens=tokens):
                    hidden = torch.randn(
                        1, tokens, config.hidden_size, device="cuda", dtype=torch.bfloat16
                    )
                    conv_ptr, state_ptr = conv.data_ptr(), state.data_ptr()
                    expected, ref_conv, ref_state = hf(hidden, ref_conv, ref_state)
                    cache = HybridCache(
                        ["linear_attention"],
                        torch.arange(position, position + tokens, device="cuda")[None, :],
                        (conv, state),
                        torch.tensor([position + tokens], dtype=torch.int32),
                        torch.tensor([tokens], dtype=torch.int32),
                    )
                    actual = adapter(hidden, cache_params=cache)
                    self.assertEqual(conv.data_ptr(), conv_ptr)
                    self.assertEqual(state.data_ptr(), state_ptr)
                    self.assertEqual(actual.dtype, torch.bfloat16)
                    self.check_numerical(actual, expected, config.hidden_size, 0.05)
                    self.assertTrue(bits_equal(conv, ref_conv))
                    self.check_numerical(state, ref_state, 128 * 128, 0.05)
                    position += tokens


if __name__ == "__main__":
    unittest.main()
