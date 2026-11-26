"""CPU checks for Qwen3.5 HybridCache metadata and in-place KV updates under export."""

import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from compiler.qwen35.model import HybridCache, state_specs


class StateCapacity(unittest.TestCase):
    def test_kv_capacity_is_independent_of_deltanet_geometry(self):
        config = SimpleNamespace(
            linear_num_key_heads=16,
            linear_key_head_dim=128,
            linear_num_value_heads=32,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            num_key_value_heads=4,
            head_dim=256,
            layer_types=["linear_attention", "full_attention"],
            max_position_embeddings=262144,
        )
        baseline = state_specs(config, capacity=1024)
        scaled = state_specs(config, capacity=16384)
        default = state_specs(config)
        self.assertEqual(baseline[:2], scaled[:2])
        self.assertEqual(default[:2], scaled[:2])
        self.assertEqual(default[2]["shape"], [1, 4, 262144, 256])
        self.assertEqual(baseline[2]["shape"], [1, 4, 1024, 256])
        self.assertEqual(scaled[2]["shape"], [1, 4, 16384, 256])
        self.assertEqual(scaled[2]["shape"], scaled[3]["shape"])
        self.assertTrue(all(set(spec) == {"layer", "kind", "shape", "dtype"} for spec in baseline))
        batched = state_specs(config, capacity=1024, batch_size=2)
        for single, batch in zip(baseline, batched):
            self.assertEqual(batch["shape"], [2, *single["shape"][1:]])


class AttentionUpdate(torch.nn.Module):
    def forward(self, key, cache):
        return cache.update(key, key + 10, 0)


class CacheStep(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = AttentionUpdate()

    def forward(self, key, positions, state):
        cache = HybridCache(
            ["full_attention"],
            positions,
            state,
            torch.full((1,), key.shape[2] + 1, dtype=torch.int32),
            torch.full((1,), key.shape[2], dtype=torch.int32),
        )
        attended_key, attended_value = self.attention(key, cache)
        return attended_key + attended_value


class HybridCacheExportContract(unittest.TestCase):
    @staticmethod
    def append_reference(key, value, positions, cu_query, cache_key, cache_value):
        # CPU stand-in for the CUDA writer: GPU tests cover lengths and masks.
        indices = positions[:, None, :, None].expand_as(key)
        cache_key.scatter_(2, indices, key)
        cache_value.scatter_(2, indices, value)

    @torch.inference_mode()
    def test_inplace_kv_updates_survive_strict_export(self):
        writer = patch("compiler.qwen35.model.append_kv_batched", self.append_reference)
        writer.start()
        self.addCleanup(writer.stop)
        model = CacheStep().eval()

        def inputs(tokens):
            key = torch.arange(2 * tokens * 3, dtype=torch.float32).reshape(1, 2, tokens, 3)
            positions = torch.arange(1, tokens + 1)[None, :]
            state = tuple(torch.full((1, 2, 8, 3), fill) for fill in (-3.0, -7.0))
            return key, positions, state

        tokens = torch.export.Dim("tokens", min=2, max=6)
        exported = torch.export.export(
            model,
            inputs(2),
            strict=True,
            dynamic_shapes=({2: tokens}, {1: tokens}, (None, None)),
        )
        self.assertEqual(
            sum(n.target == torch.ops.higher_order.cond for n in exported.graph.nodes), 0
        )
        updates = sum(n.target == torch.ops.aten.scatter_.src for n in exported.graph.nodes)
        self.assertEqual(updates, 2)
        self.assertEqual(
            len(exported.run_decompositions({}).graph_signature.user_inputs_to_mutate), 2
        )
        compiled_graph = exported.module()
        for length in (2, 3, 4, 6):
            with self.subTest(tokens=length):
                for function in (model, compiled_graph):
                    key, positions, state = inputs(length)
                    before = [t.clone() for t in (key, positions, *state)]
                    pointers = [t.data_ptr() for t in state]
                    expected_key = state[0].index_copy(2, positions.flatten(), key)
                    expected_value = state[1].index_copy(2, positions.flatten(), key + 10)
                    actual = function(key, positions, state)
                    torch.testing.assert_close(
                        actual, expected_key + expected_value, rtol=0, atol=0
                    )
                    for result, reference in zip((key, positions), before):
                        torch.testing.assert_close(result, reference, rtol=0, atol=0)
                    for i, (result, reference) in enumerate(
                        zip(state, (expected_key, expected_value))
                    ):
                        self.assertEqual(result.data_ptr(), pointers[i])
                        torch.testing.assert_close(result, reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
