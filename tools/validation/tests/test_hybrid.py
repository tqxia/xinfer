"""CPU contracts for shared HF hybrid-cache execution."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch, sentinel
import torch

from tools.validation.common import hybrid as model


class HFCacheProtocol(unittest.TestCase):
    def original(self, dtype=torch.bfloat16):
        original = Mock()
        original.config = SimpleNamespace(
            layer_types=["linear_attention", "full_attention"],
            num_key_value_heads=1,
            head_dim=2,
            vocab_size=4,
        )
        original.parameters.side_effect = lambda: iter([torch.empty(1, dtype=dtype)])
        original.return_value = SimpleNamespace(
            logits=torch.arange(4, dtype=dtype).reshape(1, 1, 4)
        )
        return original

    def cache(self, length=8):
        linear = SimpleNamespace(
            conv_states=[torch.ones(1, 2, 3)],
            recurrent_states=[torch.ones(1, 1, 2, 2)],
        )
        attention = SimpleNamespace(
            keys=torch.ones(1, 1, length, 2),
            values=torch.full((1, 1, length, 2), 2.0),
            lazy_initialization=Mock(),
        )
        return SimpleNamespace(layers=[linear, attention])

    def cpu_allocations(self):
        # Execute the wrapper's tensor/mask construction on CPU, preserving
        # every argument except its explicitly requested CUDA device.
        for name in ("zeros", "tensor", "arange", "full"):
            original = getattr(torch, name)

            def allocate(*args, _original=original, **kwargs):
                kwargs.pop("device", None)
                return _original(*args, **kwargs)

            self.enterContext(patch.object(model.torch, name, side_effect=allocate))

    def test_static_reset_uses_capacity_and_poisons_only_attention(self):
        original, cache = self.original(), self.cache()
        stage = model.HybridHFStage(original, [], 8)
        stage.position = 5
        self.cpu_allocations()
        with patch.object(model, "StaticCache", return_value=cache) as constructor:
            stage.reset(7)
        constructor.assert_called_once_with(original.config, max_cache_len=8)
        self.assertEqual(stage.position, 0)
        attention = cache.layers[1]
        attention.lazy_initialization.assert_called_once()
        key, value = attention.lazy_initialization.call_args.args
        self.assertIs(key, value)
        self.assertEqual(key.shape, (1, 1, 1, 2))
        self.assertEqual(key.dtype, torch.bfloat16)
        expected = [
            *cache.layers[0].conv_states,
            *cache.layers[0].recurrent_states,
            attention.keys,
            attention.values,
        ]
        self.assertEqual(len(stage.state), 4)
        self.assertTrue(all(actual is wanted for actual, wanted in zip(stage.state, expected)))
        self.assertTrue(bool((attention.keys == 7).all()))
        self.assertTrue(bool((attention.values == 7).all()))
        self.assertTrue(bool((stage.state[0] == 1).all()))
        self.assertTrue(bool((stage.state[1] == 1).all()))

    def test_dynamic_reset_rejects_poison_and_pads_only_attention_state(self):
        original, cache = self.original(), self.cache(length=3)
        specs = [{"kind": kind} for kind in ("conv", "recurrent", "key", "value")]
        stage = model.HybridHFStage(original, specs, 8, dynamic=True)
        with patch.object(model, "DynamicCache", return_value=cache) as constructor:
            with self.assertRaisesRegex(ValueError, "unpoisoned"):
                stage.reset(7)
            constructor.assert_not_called()
            stage.reset()
        constructor.assert_called_once_with(config=original.config)
        self.assertEqual(stage.position, 0)
        stage.position = 3
        conv, recurrent, key, value = stage.state
        self.assertIs(conv, cache.layers[0].conv_states[0])
        self.assertIs(recurrent, cache.layers[0].recurrent_states[0])
        for actual, expected in ((key, cache.layers[1].keys), (value, cache.layers[1].values)):
            self.assertEqual(actual.shape, (1, 1, 8, 2))
            torch.testing.assert_close(actual[:, :, :3], expected)
            self.assertTrue(bool((actual[:, :, 3:] == 0).all()))

    def test_static_and_dynamic_prefill_decode_masks_positions_and_dtypes(self):
        self.cpu_allocations()
        for dynamic in (False, True):
            for dtype in (torch.bfloat16, torch.float32):
                with self.subTest(dynamic=dynamic, dtype=dtype):
                    original = self.original(dtype)
                    stage = model.HybridHFStage(original, [], 8, dynamic=dynamic)
                    stage.cache = sentinel.cache
                    for tokens, start in (([1, 2], 0), ([3], 2)):
                        actual = stage.step(tokens)
                        args, kwargs = original.call_args
                        torch.testing.assert_close(args[0], torch.tensor([tokens]))
                        torch.testing.assert_close(
                            kwargs["position_ids"], torch.arange(start, start + len(tokens))[None]
                        )
                        self.assertIs(kwargs["past_key_values"], sentinel.cache)
                        self.assertTrue(kwargs["use_cache"])
                        self.assertEqual(kwargs["logits_to_keep"], 1)
                        self.assertIsNone(kwargs["attention_mask"]["linear_attention"])
                        mask = kwargs["attention_mask"]["full_attention"]
                        width = start + len(tokens) if dynamic else 8
                        self.assertEqual(mask.shape, (1, 1, len(tokens), width))
                        self.assertEqual(mask.dtype, dtype)
                        future = (
                            torch.arange(width)[None, :]
                            > torch.arange(start, start + len(tokens))[:, None]
                        )
                        expected = torch.where(future, torch.finfo(dtype).min, 0.0).to(dtype)[
                            None, None
                        ]
                        torch.testing.assert_close(mask, expected)
                        torch.testing.assert_close(actual, original.return_value.logits[:, 0])
                        self.assertEqual(stage.position, start + len(tokens))
                    stage.close()
                    self.assertIsNone(stage.model)
                    self.assertIsNone(stage.cache)

    def test_invalid_steps_leave_cache_and_position_unchanged(self):
        for dynamic in (False, True):
            original = self.original()
            stage = model.HybridHFStage(original, [], 8, dynamic=dynamic)
            stage.cache = sentinel.cache
            stage.position = 7
            for tokens in ([], [1] * 513, [1, 2]):
                with (
                    self.subTest(dynamic=dynamic, length=len(tokens)),
                    self.assertRaises(ValueError),
                ):
                    stage.step(tokens)
                original.assert_not_called()
                self.assertEqual(stage.position, 7)
                self.assertIs(stage.cache, sentinel.cache)


if __name__ == "__main__":
    unittest.main()
