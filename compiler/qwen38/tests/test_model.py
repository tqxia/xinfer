"""Qwen3.8 checkpoint policy and FP8 loading through shared construction helpers."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch, sentinel

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5ForCausalLM
from transformers.integrations.finegrained_fp8 import FP8Linear

from compiler.common.tests.tensors import bits_equal
from compiler.qwen38 import model


def quantization():
    return dict(
        quant_method="fp8", fmt="e4m3", activation_scheme="dynamic", weight_block_size=[128, 128]
    )


def write_config(directory, *, quant=None, text=None):
    path = Path(directory) / "config.json"
    path.write_text(
        json.dumps(
            dict(
                text_config=dict(tie_word_embeddings=False) if text is None else text,
                quantization_config=quantization() if quant is None else quant,
            )
        )
    )
    return path


class ModelLoading(unittest.TestCase):
    def test_config_preserves_quantization_and_untied_output_head(self):
        with TemporaryDirectory() as directory:
            config = model._text_config(write_config(directory))
        self.assertFalse(config.tie_word_embeddings)
        self.assertEqual(config.quantization_config, quantization())
        self.assertEqual(config._attn_implementation, "sdpa")

    def test_quantization_errors_are_rejected_before_model_construction(self):
        for quant in (
            False,
            {},
            dict(quantization(), fmt="e5m2"),
            dict(quantization(), weight_block_size=[64, 128]),
        ):
            with self.subTest(quant=quant), TemporaryDirectory() as directory:
                write_config(directory, quant=quant)
                with (
                    patch.object(model, "create_text_model") as create,
                    self.assertRaises(ValueError),
                ):
                    model.load_model(directory)
                create.assert_not_called()
        with TemporaryDirectory() as directory:
            path = write_config(directory)
            path.write_text(json.dumps(dict(text_config=dict(tie_word_embeddings=False))))
            with self.assertRaisesRegex(ValueError, "requires an FP8 checkpoint"):
                model._text_config(path)

    def test_tied_embeddings_and_nondefault_rope_are_rejected(self):
        with TemporaryDirectory() as directory:
            path = write_config(directory)
            for tied, rope in ((True, "default"), (False, "linear")):
                with (
                    self.subTest(tied=tied, rope=rope),
                    patch.object(
                        model,
                        "Qwen3_5TextConfig",
                        return_value=SimpleNamespace(
                            tie_word_embeddings=tied, rope_parameters=dict(rope_type=rope)
                        ),
                    ),
                    self.assertRaisesRegex(ValueError, "untied embeddings and default RoPE"),
                ):
                    model._text_config(path)

    def test_non_bf16_activations_fail_before_reading_or_allocating_model(self):
        with (
            patch.object(model, "_text_config") as config,
            patch.object(model, "create_text_model") as create,
            self.assertRaisesRegex(ValueError, "BF16 activations"),
        ):
            model.load_model("unused-checkpoint", dtype=torch.float32)
        config.assert_not_called()
        create.assert_not_called()

    def test_loader_delegates_construction_weights_and_finalization(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root)
            original = Mock()
            with (
                patch.object(model, "create_text_model", return_value=original) as create,
                patch.object(model, "load_weights") as weights,
                patch.object(
                    model.GenerationConfig, "from_pretrained", return_value=sentinel.generation
                ) as generation,
                patch.object(
                    model, "finalize_text_model", return_value=sentinel.finished
                ) as finalize,
            ):
                self.assertIs(model.load_model(root), sentinel.finished)
            self.assertEqual(create.call_args.args[0].quantization_config, quantization())
            weights.assert_called_once_with(original, root / "text.safetensors")
            generation.assert_called_once_with(root, local_files_only=True)
            self.assertIs(original.generation_config, sentinel.generation)
            finalize.assert_called_once_with(original)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QuantizedCheckpoint(unittest.TestCase):
    @torch.inference_mode()
    def test_real_loader_preserves_weights_generation_and_rotary_buffers(self):
        torch.manual_seed(38)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(
                root,
                text=dict(
                    vocab_size=128,
                    hidden_size=128,
                    intermediate_size=256,
                    num_hidden_layers=1,
                    layer_types=["full_attention"],
                    num_attention_heads=1,
                    num_key_value_heads=1,
                    head_dim=128,
                    max_position_embeddings=512,
                    tie_word_embeddings=False,
                ),
            )
            config = model._text_config(root / "config.json")
            reference = Qwen3_5ForCausalLM(config).to(torch.bfloat16)
            weights = {name: value.clone() for name, value in reference.state_dict().items()}
            name = "model.layers.0.mlp.down_proj"
            weights[name + ".weight"] = weights[name + ".weight"].to(torch.float8_e4m3fn)
            weights[name + ".weight_scale_inv"] = torch.full((1, 2), 0.01, dtype=torch.bfloat16)
            save_file(weights, root / "text.safetensors")
            model.GenerationConfig(eos_token_id=[1, 2]).save_pretrained(root)
            loaded = model.load_model(root)
            self.assertIs(type(loaded.get_submodule(name)), FP8Linear)
            for key, value in loaded.state_dict().items():
                self.assertEqual(value.device.type, "cuda", key)
                self.assertTrue(bits_equal(value.cpu(), weights[key]), key)
            self.assertNotEqual(
                loaded.lm_head.weight.data_ptr(), loaded.model.embed_tokens.weight.data_ptr()
            )
            self.assertEqual(loaded.model.rotary_emb.inv_freq.dtype, torch.float32)
            self.assertEqual(loaded.generation_config.eos_token_id, [1, 2])
            self.assertFalse(loaded.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in loaded.parameters()))
            output = loaded(torch.tensor([[3, 4]], device="cuda"), use_cache=False).logits
            self.assertEqual(output.shape, (1, 2, 128))
            self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
