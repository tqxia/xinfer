"""CPU regression for full-attention cache updates after symbolic DeltaNet dispatch."""

import unittest
from types import SimpleNamespace

import torch

from compiler.qwen35.model import TextCache, state_specs


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


class AttentionUpdate(torch.nn.Module):
    def forward(self, key, cache):
        return cache.update(key, key + 10, 0)


class CacheAfterCond(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = AttentionUpdate()

    def forward(self, key, positions, state):
        # The cache must exist before cond, as it does before the first
        # DeltaNet layer in TextStep. Creating it afterwards misses the bug.
        cache = TextCache(["full_attention"], positions, state)
        key = torch.cond(key.shape[2] <= 3, lambda x: x + 1, lambda x: x + 2, (key,))
        attended_key, attended_value = self.attention(key, cache)
        return attended_key, attended_value, *cache.tensors()


class TextCacheExportContract(unittest.TestCase):
    @torch.inference_mode()
    def test_updates_survive_cond_and_mutate_only_cache_rows(self):
        model = CacheAfterCond().eval()

        def inputs(tokens):
            key = torch.arange(2 * tokens * 3, dtype=torch.float32).reshape(1, 2, tokens, 3)
            positions = torch.arange(1, tokens + 1)
            state = [torch.full((1, 2, 8, 3), fill) for fill in (-3.0, -7.0)]
            return key, positions, state

        tokens = torch.export.Dim("tokens", min=2, max=6)
        exported = torch.export.export(
            model,
            inputs(2),
            strict=True,
            dynamic_shapes=({2: tokens}, {0: tokens}, [None, None]),
        )
        self.assertEqual(
            sum(n.target == torch.ops.higher_order.cond for n in exported.graph.nodes), 1
        )
        updates = sum(n.target == torch.ops.aten.index_copy_.default for n in exported.graph.nodes)
        self.assertEqual(updates, 2)
        compiled_graph = exported.module()
        for length in (2, 3, 4, 6):
            with self.subTest(tokens=length):
                for function in (model, compiled_graph):
                    key, positions, state = inputs(length)
                    before = [t.clone() for t in (key, positions, *state)]
                    pointers = [t.data_ptr() for t in state]
                    updated_key = key + (1 if length <= 3 else 2)
                    expected_key = state[0].index_copy(2, positions, updated_key)
                    expected_value = state[1].index_copy(2, positions, updated_key + 10)
                    expected = (expected_key, expected_value, expected_key, expected_value)
                    actual = function(key, positions, state)
                    self.assertEqual(len(actual), len(expected))
                    for result, reference in zip(actual, expected):
                        torch.testing.assert_close(result, reference, rtol=0, atol=0)
                    for result, reference in zip((key, positions), before):
                        torch.testing.assert_close(result, reference, rtol=0, atol=0)
                    for i, (result, reference) in enumerate(zip(state, expected[:2])):
                        self.assertEqual(result.data_ptr(), pointers[i])
                        torch.testing.assert_close(result, reference, rtol=0, atol=0)
                        self.assertEqual(actual[i].data_ptr(), pointers[i])
                        self.assertEqual(actual[i + 2].data_ptr(), pointers[i])


if __name__ == "__main__":
    unittest.main()
