"""Checkpoint loading and mixed-layer batched model contracts."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3_5ForCausalLM

from compiler.common.artifacts import MODEL_MANIFEST_NAME
from compiler.common.tests.tensors import bits_equal, row_errors
from compiler.qwen35 import model
from compiler.qwen35.adapters.delta import install_delta


class ModelLoading(unittest.TestCase):
    def test_text_rotary_matches_hf_grid_recomposition_bitwise(self):
        config = model.Qwen3_5TextConfig()
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            rotary = model.Qwen3_5TextRotaryEmbedding(config).to(device)
            for batch in (1, 2, 8):
                for tokens in (1, 2, 98, 512):
                    with self.subTest(device=device, batch=batch, tokens=tokens):
                        positions = (
                            torch.arange(batch * tokens, device=device).reshape(batch, tokens)
                            + 8191
                        )
                        hidden = torch.empty(batch, tokens, 1, device=device, dtype=torch.bfloat16)
                        expected = rotary(hidden, positions.unsqueeze(0).expand(3, -1, -1))
                        actual = model.text_rotary(rotary, positions, hidden.dtype)
                        self.assertTrue(all(bits_equal(a, b) for a, b in zip(actual, expected)))

    def test_loader_reads_weights_without_manifest_verification(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "config.json").write_text(
                json.dumps(dict(text_config=dict(tie_word_embeddings=True)))
            )
            read_text = Path.read_text

            def read_without_manifest(path, *arguments, **options):
                self.assertNotEqual(path.name, MODEL_MANIFEST_NAME)
                return read_text(path, *arguments, **options)

            for manifest in (None, "invalid manifest"):
                with self.subTest(manifest=manifest):
                    if manifest is not None:
                        (output / MODEL_MANIFEST_NAME).write_text(manifest)
                    with (
                        patch.object(Path, "read_text", read_without_manifest),
                        patch.object(model, "Qwen3_5ForCausalLM") as constructor,
                        patch.object(
                            model, "load_file", side_effect=RuntimeError("reached weight loading")
                        ) as weights,
                    ):
                        with self.assertRaisesRegex(RuntimeError, "reached weight loading"):
                            model.load_model(output)
                        constructor.assert_called_once()
                        weights.assert_called_once_with(output / "text.safetensors")

    def test_configuration_requires_tied_embeddings_and_default_rope(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(dict(text_config={})))
            for tied, rope in ((False, "default"), (True, "linear")):
                with (
                    self.subTest(tied=tied, rope=rope),
                    patch.object(
                        model,
                        "Qwen3_5TextConfig",
                        return_value=SimpleNamespace(
                            tie_word_embeddings=tied, rope_parameters=dict(rope_type=rope)
                        ),
                    ),
                ):
                    with self.assertRaisesRegex(ValueError, "tied embeddings and default RoPE"):
                        model._text_config(path)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class BatchedTextStep(unittest.TestCase):
    @torch.inference_mode()
    def test_unequal_appends_select_real_logits_and_preserve_inactive_state(self):
        model.configure_text()
        torch.manual_seed(3508)
        config = model.Qwen3_5TextConfig(
            vocab_size=128,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            layer_types=["linear_attention", "full_attention"],
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=256,
            linear_num_key_heads=16,
            linear_num_value_heads=16,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            tie_word_embeddings=True,
        )
        original = Qwen3_5ForCausalLM(config).eval().requires_grad_(False)
        original.to(device="cuda", dtype=torch.bfloat16)
        original.model.rotary_emb = model.Qwen3_5TextRotaryEmbedding(config).to("cuda")
        install_delta(original)
        step = model.TextStep(original)
        state = model.zero_state(config, 512, batch_size=2)
        independent = [model.zero_state(config, 512) for _ in range(2)]
        starts = torch.zeros(2, dtype=torch.int32)
        for counts in ((7, 3), (1, 0), (0, 2), (97, 98), (1, 1)):
            with self.subTest(counts=counts):
                lengths = torch.tensor(counts, dtype=torch.int32)
                width = max(counts)
                ids = torch.randint(0, config.vocab_size, (2, width), device="cuda")
                positions = starts.to("cuda")[:, None] + torch.arange(width, device="cuda")
                saved = [t.clone() for t in state]
                logits, *next_state = step(ids, positions, state, starts + lengths, lengths)
                # Poison only padding and rerun from exactly the same carried state.
                poisoned = ids.clone()
                for row, count in enumerate(counts):
                    poisoned[row, count:] = (poisoned[row, count:] + 1) % config.vocab_size
                again = step(
                    poisoned, positions, [t.clone() for t in saved], starts + lengths, lengths
                )
                self.assertTrue(all(bits_equal(a, b) for a, b in zip((logits, *next_state), again)))
                for index, (before, after) in enumerate(zip(state, next_state)):
                    self.assertEqual(after.data_ptr() == before.data_ptr(), index >= 2)
                    if index < 2:
                        self.assertTrue(bits_equal(before, saved[index]))
                    for row, count in enumerate(counts):
                        if index >= 2:
                            start, end = starts[row].item(), (starts[row] + count).item()
                            self.assertTrue(
                                bits_equal(after[row, :, :start], saved[index][row, :, :start])
                            )
                            self.assertTrue(
                                bits_equal(after[row, :, end:], saved[index][row, :, end:])
                            )
                        if count == 0:
                            self.assertTrue(bits_equal(after[row], saved[index][row]))
                for row, count in enumerate(counts):
                    if count == 0:
                        self.assertEqual(torch.count_nonzero(logits[row]).item(), 0)
                        continue
                    reference, *independent[row] = step(
                        ids[row : row + 1, :count].contiguous(),
                        positions[row : row + 1, :count].contiguous(),
                        independent[row],
                        (starts + lengths)[row : row + 1],
                        lengths[row : row + 1],
                    )
                    error = row_errors(logits[row : row + 1].float(), reference.float())
                    self.assertLessEqual(error["relative_l2"], 0.05)
                    self.assertLessEqual(error["peak_relative"], 0.05)
                state = next_state
                starts += lengths


if __name__ == "__main__":
    unittest.main()
