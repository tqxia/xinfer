"""CPU context-limit checks independent of Torch, transformers and device setup."""

from types import SimpleNamespace
import unittest

from compiler.common.context import model_context_limit, resolve_context_length


class ContextLength(unittest.TestCase):
    def test_context_reader_is_independent_of_compiled_call_minimum(self):
        self.assertEqual(model_context_limit(dict(max_position_embeddings=256)), 256)
        with self.assertRaisesRegex(ValueError, "integer in 512..256"):
            resolve_context_length(dict(max_position_embeddings=256))

    def test_default_capacity_uses_the_declared_model_ceiling(self):
        config = dict(max_position_embeddings=40960)
        self.assertEqual(resolve_context_length(config), 40960)
        self.assertEqual(resolve_context_length(SimpleNamespace(**config)), 40960)
        for capacity in (512, 1024, 16384, 40960):
            with self.subTest(capacity=capacity):
                self.assertEqual(resolve_context_length(config, capacity), capacity)

    def test_multimodal_configuration_uses_text_ceiling(self):
        for config in (
            dict(max_position_embeddings=512, text_config=dict(max_position_embeddings=262144)),
            SimpleNamespace(
                max_position_embeddings=512,
                text_config=SimpleNamespace(max_position_embeddings=262144),
            ),
        ):
            with self.subTest(config=config):
                self.assertEqual(resolve_context_length(config), 262144)
                self.assertEqual(resolve_context_length(config, 16384), 16384)

    def test_override_must_fit_declared_context_without_rope_factor_multiplication(self):
        config = dict(
            max_position_embeddings=1024,
            rope_scaling=dict(rope_type="linear", factor=16),
        )
        self.assertEqual(resolve_context_length(config), 1024)
        for capacity in (False, True, 512.0, "512", -1, 0, 511, 1025, 16384):
            with self.subTest(capacity=capacity):
                with self.assertRaisesRegex(ValueError, "integer in 512..1024"):
                    resolve_context_length(config, capacity)

    def test_missing_or_invalid_model_ceiling_is_rejected(self):
        for config in (
            {},
            dict(max_position_embeddings=None),
            dict(max_position_embeddings=True),
            dict(max_position_embeddings=16384.0),
            dict(max_position_embeddings="16384"),
            dict(max_position_embeddings=0),
            dict(max_position_embeddings=2**63 - 1),
            dict(max_position_embeddings=16384, text_config={}),
        ):
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, "declare a positive integer"):
                    resolve_context_length(config)


if __name__ == "__main__":
    unittest.main()
