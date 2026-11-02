"""Compare gated Qwen3.5 prefix Flash attention with the original HF module."""

import unittest

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention, Qwen3_5TextConfig, Qwen3_5TextRotaryEmbedding,
)

from compiler.common.attention import cache_extent, load_attention_ops
from compiler.qwen35.adapters.attention import FlashAttention, install_attention
from compiler.qwen35.config import CACHE_LENGTH
from compiler.qwen35.model import TextCache, configure_text


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class GatedPrefixFlash(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_text()
        load_attention_ops()
        torch.manual_seed(35)
        cls.config = Qwen3_5TextConfig(
            hidden_size=1024, num_attention_heads=8, num_key_value_heads=2, head_dim=256,
            num_hidden_layers=1, layer_types=["full_attention"],
        )
        cls.config._attn_implementation = "sdpa"
        cls.original = Qwen3_5Attention(cls.config, layer_idx=0).eval().requires_grad_(False)
        cls.original.to(device="cuda", dtype=torch.bfloat16)
        cls.flash = FlashAttention(cls.original).eval()
        # As in the prepared full model, rotary frequencies remain FP32.
        cls.rotary = Qwen3_5TextRotaryEmbedding(cls.config).eval().to("cuda")

    @torch.inference_mode()
    def test_gated_attention_and_cache_writes(self):
        self.assertFalse(torch.backends.cuda.flash_sdp_enabled())
        self.assertTrue(torch.backends.cuda.mem_efficient_sdp_enabled())
        self.assertEqual(self.config.rope_parameters["partial_rotary_factor"], 0.25)
        self.assertEqual(self.rotary.inv_freq.dtype, torch.float32)
        for tokens, length in ((1, 1), (1, 129), (2, 3), (16, 129), (2, 1024)):
            with self.subTest(tokens=tokens, length=length):
                hidden = torch.randn(1, tokens, 1024, device="cuda", dtype=torch.bfloat16)
                positions = torch.arange(length - tokens, length, device="cuda")
                rotary = self.rotary(hidden, positions.view(1, 1, -1).expand(3, 1, -1))
                self.assertEqual(rotary[0].shape, (1, tokens, 64))
                initial = [torch.randn(1, 2, CACHE_LENGTH, 256, device="cuda", dtype=torch.bfloat16)
                           for _ in range(2)]
                states = [[tensor.clone() for tensor in initial] for _ in range(2)]
                caches = [TextCache(["full_attention"], positions, state, cache_extent(length))
                          for state in states]
                pointers = [[tensor.data_ptr() for tensor in state] for state in states]
                future = torch.arange(CACHE_LENGTH, device="cuda")[None, :] > positions[:, None]
                mask = torch.zeros_like(future, dtype=hidden.dtype).masked_fill(
                    future, torch.finfo(hidden.dtype).min)[None, None]
                reference, _ = self.original(hidden, position_embeddings=rotary,
                                             attention_mask=mask, past_key_values=caches[0])
                actual, _ = self.flash(hidden, position_embeddings=rotary,
                                       past_key_values=caches[1])
                self.assertTrue(bool(torch.isfinite(actual).all()))
                self.assertTrue(bool(torch.isfinite(reference).all()))
                relative_error = (actual.float() - reference.float()).norm() / reference.float().norm()
                self.assertLessEqual(relative_error.item(), 0.05)
                for cache, state, addresses in zip(caches, states, pointers):
                    self.assertEqual([tensor.data_ptr() for tensor in state], addresses)
                    for returned, owned, before in zip(cache.tensors(), state, initial):
                        self.assertIs(returned, owned)
                        self.assertTrue(torch.equal(owned[:, :, :length - tokens], before[:, :, :length - tokens]))
                        self.assertTrue(torch.equal(owned[:, :, length:], before[:, :, length:]))
                for expected, updated in zip(*states):
                    self.assertTrue(torch.equal(expected, updated))
                poisoned = [tensor.clone() for tensor in initial]
                for tensor, value in zip(poisoned, (100., -100.)):
                    tensor[:, :, length:].fill_(value)
                poisoned_cache = TextCache(["full_attention"], positions, poisoned, cache_extent(length))
                output, _ = self.flash(hidden, position_embeddings=rotary, past_key_values=poisoned_cache)
                self.assertTrue(torch.equal(output, actual))
                for updated, clean, value in zip(poisoned, states[1], (100., -100.)):
                    self.assertTrue(torch.equal(updated[:, :, :length], clean[:, :, :length]))
                    self.assertTrue(bool((updated[:, :, length:] == value).all()))

    @torch.inference_mode()
    def test_installation_reuses_parameters_and_is_idempotent(self):
        model = torch.nn.Module()
        model.config = self.config
        model.model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.self_attn = self.original
        model.model.layers = torch.nn.ModuleList([layer])
        model.eval()
        parameters = {id(parameter) for parameter in model.parameters()}
        install_attention(model)
        attention = layer.self_attn
        self.assertIsInstance(attention, FlashAttention)
        self.assertIs(attention.original, self.original)
        self.assertFalse(attention.training)
        self.assertEqual({id(parameter) for parameter in model.parameters()}, parameters)
        install_attention(model)
        self.assertIs(layer.self_attn, attention)
        self.assertEqual({id(parameter) for parameter in model.parameters()}, parameters)


if __name__ == "__main__":
    unittest.main()
