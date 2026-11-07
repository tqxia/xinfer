"""Qwen3 attention adapter, exported cache contract and parameter reuse."""

import unittest

import torch
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3Config,
    Qwen3RotaryEmbedding,
)

from compiler.common.attention import cache_extent, load_attention_ops
from compiler.common.tests.tensors import bits_equal, row_errors
from compiler.qwen3.adapters.qwen3_flash import FlashAttention, FlashStep, PrefixCache
from compiler.qwen3.model import TensorCache


TEST_CAPACITY = 16384


class AttentionProbe(torch.nn.Module):
    """Run the original HF attention and its adapter with the same tensor cache."""

    def __init__(self, attention, rotary, adapter=False):
        super().__init__()
        self.attention = FlashAttention(attention) if adapter else attention
        self.rotary = rotary
        self.adapter = adapter
        self.eval()

    def forward(self, hidden, cache_position, kv, kv_extent):
        mask = None
        if self.adapter:
            cache = PrefixCache(kv, cache_position, kv_extent)
        else:
            cache = TensorCache(kv, cache_position)
            future = torch.arange(kv[0].shape[2], device=hidden.device).unsqueeze(
                0
            ) > cache_position.unsqueeze(1)
            mask = torch.zeros_like(future, dtype=hidden.dtype)
            mask = mask.masked_fill(future, torch.finfo(hidden.dtype).min)[None, None]
        rotary = self.rotary(hidden, cache_position.unsqueeze(0))
        return self.attention(
            hidden,
            position_embeddings=rotary,
            attention_mask=mask,
            past_key_value=cache,
            cache_position=cache_position,
        )[0]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class PrefixFlashAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        load_attention_ops(model="qwen3")
        cls.config = Qwen3Config(
            hidden_size=1024,
            intermediate_size=3072,
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            max_position_embeddings=40960,
            rope_theta=1000000.0,
            use_sliding_window=False,
        )
        cls.config._attn_implementation = "sdpa"
        cls.original = Qwen3Attention(cls.config, layer_idx=0).eval().requires_grad_(False)
        cls.original.to(device="cuda", dtype=torch.bfloat16)
        cls.rotary = Qwen3RotaryEmbedding(cls.config).eval().to("cuda")
        cls.hf = AttentionProbe(cls.original, cls.rotary)
        cls.adapter = AttentionProbe(cls.original, cls.rotary, adapter=True)

    def inputs(self, tokens, length):
        return (
            torch.randn(1, tokens, self.config.hidden_size, device="cuda", dtype=torch.bfloat16),
            torch.arange(length - tokens, length, device="cuda"),
            tuple(
                torch.randn(1, 8, TEST_CAPACITY, 128, device="cuda", dtype=torch.bfloat16)
                for _ in range(2)
            ),
            cache_extent(length),
        )

    @torch.inference_mode()
    def test_hf_module_and_exported_cache_contract(self):
        for kind in ("decode", "prefill"):
            example = self.inputs(1 if kind == "decode" else 16, 32)
            tokens = torch.export.Dim("tokens", min=2, max=512)
            extent = torch.export.Dim("extent", min=2, max=TEST_CAPACITY + 1)
            shapes = (
                {} if kind == "decode" else {1: tokens},
                {} if kind == "decode" else {0: tokens},
                ({}, {}),
                {0: extent},
            )
            exported = torch.export.export(
                self.adapter, example, dynamic_shapes=shapes, strict=False
            )
            exported.validate()
            targets = [node.target for node in exported.graph.nodes if node.op == "call_function"]
            self.assertEqual(targets.count(torch.ops.xinfer.flash_attention.default), 1)
            self.assertEqual(targets.count(torch.ops.aten.index_copy_.default), 2)
            self.assertNotIn(torch.ops.aten.scaled_dot_product_attention.default, targets)
            self.assertNotIn(torch.ops.aten._scaled_dot_product_flash_attention.default, targets)
            site = next(
                node
                for node in exported.graph.nodes
                if node.target == torch.ops.xinfer.flash_attention.default
            )
            self.assertTrue(site.args[0].meta["val"].is_contiguous())
            exported_adapter = exported.module()
            cases = (
                ((1, 1), (1, 2), (1, TEST_CAPACITY))
                if kind == "decode"
                else (
                    (2, 2),
                    (2, 3),
                    (16, 32),
                    (510, 1022),
                    (512, 512),
                    (512, TEST_CAPACITY),
                )
            )
            for count, length in cases:
                with self.subTest(kind=kind, tokens=count, length=length):
                    hidden, positions, initial, valid = self.inputs(count, length)
                    outputs, caches = [], []
                    for call in (self.hf, self.adapter, exported_adapter):
                        kv = tuple(t.clone() for t in initial)
                        pointers = [t.data_ptr() for t in kv]
                        outputs.append(call(hidden, positions, kv, valid))
                        self.assertEqual(pointers, [t.data_ptr() for t in kv])
                        start = length - count
                        for before, after in zip(initial, kv):
                            self.assertTrue(bits_equal(before[:, :, :start], after[:, :, :start]))
                            self.assertTrue(bits_equal(before[:, :, length:], after[:, :, length:]))
                        caches.append(kv)
                    # Preserve the former validator's independent 5% bounds.
                    hf_error = row_errors(
                        outputs[1].flatten(0, 1).float(), outputs[0].flatten(0, 1).float()
                    )
                    with self.subTest(adapter_vs_hf=hf_error):
                        self.assertLessEqual(hf_error["relative_l2"], 0.05)
                        self.assertLessEqual(hf_error["peak_relative"], 0.05)
                        self.assertTrue(bits_equal(outputs[2], outputs[1]))
                        for kv in caches[1:]:
                            self.assertTrue(all(bits_equal(a, b) for a, b in zip(kv, caches[0])))
                    poisoned = tuple(t.clone() for t in initial)
                    for tensor in poisoned:
                        tensor[:, :, length:].fill_(100)
                    valid.fill_(255)
                    self.assertTrue(
                        bits_equal(self.adapter(hidden, positions, poisoned, valid), outputs[1])
                    )

    @torch.inference_mode()
    def test_unsupported_inputs_are_rejected(self):
        hidden, positions, kv, extent = self.inputs(1, 1)
        arguments = dict(
            position_embeddings=self.rotary(hidden, positions[None]),
            past_key_value=PrefixCache(kv, positions, extent),
            cache_position=positions,
        )
        try:
            for name, changes in (
                ("mask", {"attention_mask": torch.zeros(1, 1, 1, TEST_CAPACITY, device="cuda")}),
                ("cache", {"past_key_value": TensorCache(kv, positions)}),
                ("positions", {"cache_position": positions.clone()}),
                ("attention_weights", {"output_attentions": True}),
                ("training", {}),
            ):
                with self.subTest(unsupported=name):
                    self.adapter.attention.train(name == "training")
                    with self.assertRaises(ValueError):
                        self.adapter.attention(hidden, **(arguments | changes))
        finally:
            self.adapter.attention.eval()

    def test_installation_reuses_parameters_and_decoder_layers(self):
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        layers = []
        originals = []
        for index in range(2):
            layer = torch.nn.Module()
            layer.self_attn = Qwen3Attention(self.config, layer_idx=index).eval()
            originals.append(layer.self_attn)
            layers.append(layer)
        model.model.layers = torch.nn.ModuleList(layers)
        model.eval()
        parameters = {id(parameter) for parameter in model.parameters()}
        step = FlashStep(model)
        self.assertEqual(parameters, {id(parameter) for parameter in step.parameters()})
        self.assertIs(step.model, model)
        self.assertTrue(all(before is after for before, after in zip(layers, model.model.layers)))
        for layer, original in zip(layers, originals):
            self.assertIsInstance(layer.self_attn, FlashAttention)
            self.assertIs(layer.self_attn.original, original)
            self.assertFalse(layer.self_attn.training)


if __name__ == "__main__":
    unittest.main()
