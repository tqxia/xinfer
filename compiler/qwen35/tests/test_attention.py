"""Compare gated Qwen3.5 prefix Flash attention with the original HF module."""

import unittest

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5TextConfig,
    Qwen3_5TextRotaryEmbedding,
)

from compiler.common.attention import cache_extent, load_attention_ops
from compiler.qwen35.adapters.attention import FlashAttention, install_attention
from compiler.qwen35.config import CACHE_LENGTH
from compiler.qwen35.model import TextCache, configure_text


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class GatedPrefixFlash:
    hidden_size = 1024
    query_heads = 8
    kv_heads = 2

    @classmethod
    def setUpClass(cls):
        configure_text()
        load_attention_ops(model="qwen35")
        torch.manual_seed(35)
        cls.config = Qwen3_5TextConfig(
            hidden_size=cls.hidden_size,
            num_attention_heads=cls.query_heads,
            num_key_value_heads=cls.kv_heads,
            head_dim=256,
            num_hidden_layers=1,
            layer_types=["full_attention"],
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
                hidden = torch.randn(
                    1, tokens, self.hidden_size, device="cuda", dtype=torch.bfloat16
                )
                positions = torch.arange(length - tokens, length, device="cuda")
                rotary = self.rotary(hidden, positions.view(1, 1, -1).expand(3, 1, -1))
                self.assertEqual(rotary[0].shape, (1, tokens, 64))
                initial = [
                    torch.randn(
                        1, self.kv_heads, CACHE_LENGTH, 256, device="cuda", dtype=torch.bfloat16
                    )
                    for _ in range(2)
                ]
                states = [[tensor.clone() for tensor in initial] for _ in range(2)]
                caches = [
                    TextCache(["full_attention"], positions, state, cache_extent(length))
                    for state in states
                ]
                pointers = [[tensor.data_ptr() for tensor in state] for state in states]
                future = torch.arange(CACHE_LENGTH, device="cuda")[None, :] > positions[:, None]
                mask = torch.zeros_like(future, dtype=hidden.dtype).masked_fill(
                    future, torch.finfo(hidden.dtype).min
                )[None, None]
                reference, _ = self.original(
                    hidden,
                    position_embeddings=rotary,
                    attention_mask=mask,
                    past_key_values=caches[0],
                )
                actual, _ = self.flash(
                    hidden, position_embeddings=rotary, past_key_values=caches[1]
                )
                self.assertTrue(bool(torch.isfinite(actual).all()))
                self.assertTrue(bool(torch.isfinite(reference).all()))
                relative_error = (
                    actual.float() - reference.float()
                ).norm() / reference.float().norm()
                self.assertLessEqual(relative_error.item(), 0.05)
                for cache, state, addresses in zip(caches, states, pointers):
                    self.assertEqual([tensor.data_ptr() for tensor in state], addresses)
                    for returned, owned, before in zip(cache.tensors(), state, initial):
                        self.assertIs(returned, owned)
                        self.assertTrue(
                            torch.equal(
                                owned[:, :, : length - tokens], before[:, :, : length - tokens]
                            )
                        )
                        self.assertTrue(torch.equal(owned[:, :, length:], before[:, :, length:]))
                for expected, updated in zip(*states):
                    self.assertTrue(torch.equal(expected, updated))
                poisoned = [tensor.clone() for tensor in initial]
                for tensor, value in zip(poisoned, (100.0, -100.0)):
                    tensor[:, :, length:].fill_(value)
                poisoned_cache = TextCache(
                    ["full_attention"], positions, poisoned, cache_extent(length)
                )
                output, _ = self.flash(
                    hidden, position_embeddings=rotary, past_key_values=poisoned_cache
                )
                self.assertTrue(torch.equal(output, actual))
                for updated, clean, value in zip(poisoned, states[1], (100.0, -100.0)):
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


class GatedPrefixFlash08B(GatedPrefixFlash, unittest.TestCase):
    pass


class GatedPrefixFlash4B(GatedPrefixFlash, unittest.TestCase):
    hidden_size = 2560
    query_heads = 16
    kv_heads = 4


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class PrefixFlashGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_text()
        load_attention_ops(model="qwen35")

    @staticmethod
    def inputs(heads, kv_heads, dim, tokens=16, length=129):
        q = torch.randn(1, heads, tokens, dim, device="cuda", dtype=torch.bfloat16)
        k, v = (
            torch.randn(1, kv_heads, CACHE_LENGTH, dim, device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        )
        return q, k, v, cache_extent(length), max(dim, 1) ** -0.5

    @torch.inference_mode()
    def test_input_derived_gqa_geometry_and_fake_contract(self):
        # Include a geometry outside either model to catch a hidden profile whitelist.
        for heads, kv_heads, dim in ((16, 8, 128), (8, 2, 256), (16, 4, 256), (6, 2, 64)):
            with self.subTest(heads=heads, kv_heads=kv_heads, dim=dim):
                inputs = self.inputs(heads, kv_heads, dim)
                q, k, v, extent, scale = inputs
                length = extent.numel() - 1
                key = k[:, :, :length].repeat_interleave(heads // kv_heads, dim=1).float()
                value = v[:, :, :length].repeat_interleave(heads // kv_heads, dim=1).float()
                scores = (q.float() @ key.transpose(-2, -1)) * scale
                # Appended queries attend through their corresponding prefix positions.
                future = (
                    torch.arange(length, device="cuda")[None, :]
                    > torch.arange(length - q.shape[2], length, device="cuda")[:, None]
                )
                expected = scores.masked_fill(future, -torch.inf).softmax(-1) @ value
                actual = torch.ops.xinfer.flash_attention(*inputs)
                self.assertEqual(actual.shape, q.shape)
                self.assertTrue(actual.is_contiguous())
                relative_error = (actual.float() - expected).norm() / expected.norm()
                self.assertLessEqual(relative_error.item(), 0.01)
                torch.library.opcheck(
                    torch.ops.xinfer.flash_attention.default,
                    inputs,
                    test_utils=("test_schema", "test_faketensor"),
                )

    @torch.inference_mode()
    def test_reject_invalid_head_geometry(self):
        for heads, kv_heads, dim in ((0, 2, 128), (8, 0, 128), (3, 2, 128)):
            with self.subTest(heads=heads, kv_heads=kv_heads, dim=dim):
                with self.assertRaisesRegex(RuntimeError, "Query heads"):
                    torch.ops.xinfer.flash_attention(*self.inputs(heads, kv_heads, dim))
        for dim in (0, 7, 127, 264):
            with self.subTest(dim=dim):
                inputs = self.inputs(8, 2, dim)
                with self.assertRaisesRegex(RuntimeError, "Flash head dimension"):
                    torch.ops.xinfer.flash_attention(*inputs[:-1], 1.0)


if __name__ == "__main__":
    unittest.main()
