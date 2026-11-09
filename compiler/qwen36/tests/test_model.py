"""Small CPU checkpoints exercise mixed-dtype loading without model inference."""

import json
import re
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3_5MoeForCausalLM
from transformers.integrations import finegrained_fp8 as hf_fp8

from compiler.qwen36 import config, model
from tools.prepare import qwen36 as preparation
from tools.prepare.tests import test_qwen36 as preparation_tests


def bytes_equal(left, right):
    return torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


class ModelLoading(unittest.TestCase):
    def checkpoint(self, root, mutate=None):
        source, output = root / "source", root / "text"
        source.mkdir()

        def configure_fixture(tensors, document, generation):
            text = document["text_config"]
            text["rope_parameters"].update(
                rope_theta=10000.0,
                partial_rotary_factor=0.25,
                mrope_section=[3, 3, 2],
                mrope_interleaved=True,
            )
            exclusions = document["quantization_config"]["modules_to_not_convert"]
            exclusions.append(preparation.PREFIX + "layers.0.linear_attn.in_proj_b")
            for layer in range(2):
                exclusions.extend(
                    preparation.PREFIX + f"layers.{layer}.mlp." + suffix
                    for suffix in ("gate", "shared_expert_gate")
                )
            if mutate:
                mutate(tensors, document, generation)

        fixture = preparation_tests.PreparedFP8Checkpoint
        info, original = fixture.source(self, source, configure_fixture)
        fixture.network(self, source, info, output)
        preparation.prepare(output)
        return output, original

    def test_preserves_every_source_byte_and_official_hf_module(self):
        with TemporaryDirectory() as temporary:
            directory, originals = self.checkpoint(Path(temporary))
            with patch.object(model, "configure_hf_fp8") as preload:
                loaded = model.load_model(directory, device="cpu")
            preload.assert_not_called()
            self.assertIsInstance(loaded, Qwen3_5MoeForCausalLM)
            self.assertEqual(loaded.config._experts_implementation, "grouped_mm")
            self.assertIsInstance(loaded.model.layers[0].linear_attn.in_proj_qkv, hf_fp8.FP8Linear)
            self.assertIsInstance(loaded.model.layers[1].self_attn.q_proj, hf_fp8.FP8Linear)
            self.assertIsInstance(loaded.model.layers[0].mlp.experts, hf_fp8.FP8Experts)
            self.assertIs(type(loaded.model.layers[0].linear_attn.in_proj_a), torch.nn.Linear)
            self.assertIs(type(loaded.model.layers[0].linear_attn.in_proj_b), torch.nn.Linear)
            self.assertIs(type(loaded.model.layers[0].mlp.gate.weight), torch.nn.Parameter)
            expert_pattern = re.compile(
                r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
                r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$"
            )
            for name, expected in originals.items():
                match = expert_pattern.fullmatch(name)
                if match:
                    layer, expert, projection, suffix = match.groups()
                    experts = loaded.model.layers[int(layer)].mlp.experts
                    scale = suffix == "weight_scale_inv"
                    target = "down_proj" if projection == "down_proj" else "gate_up_proj"
                    target += "_scale_inv" if scale else ""
                    actual = getattr(experts, target)[int(expert)]
                    if projection != "down_proj":
                        rows = 1 if scale else 128
                        start = rows if projection == "up_proj" else 0
                        actual = actual[start : start + rows]
                else:
                    prepared = (
                        "model." + name.removeprefix(preparation.PREFIX)
                        if name.startswith(preparation.PREFIX)
                        else name
                    )
                    actual = loaded.get_parameter(prepared)
                with self.subTest(source=name):
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assertTrue(bytes_equal(actual, expected))
            self.assertNotEqual(
                loaded.lm_head.weight.data_ptr(), loaded.model.embed_tokens.weight.data_ptr()
            )
            self.assertFalse(bytes_equal(loaded.lm_head.weight, loaded.model.embed_tokens.weight))
            self.assertEqual(loaded.generation_config.eos_token_id, [2, 3])
            self.assertFalse(loaded.training)
            self.assertTrue(
                all(not value.requires_grad and not value.is_meta for value in loaded.parameters())
            )
            self.assertTrue(
                all(
                    value.dtype == torch.float32 and torch.isfinite(value).all()
                    for value in loaded.model.rotary_emb.buffers()
                )
            )
            self.assertEqual(
                loaded.checkpoint_storage["source_bytes"],
                loaded.checkpoint_storage["parameter_bytes"],
            )
            self.assertEqual(
                set(loaded.checkpoint_storage["parameter_bytes_by_dtype"]),
                {"float8_e4m3fn", "bfloat16"},
            )

    def test_prefix_skips_later_shards_and_uses_one_parameter_bank(self):
        with TemporaryDirectory() as temporary:
            directory, originals = self.checkpoint(Path(temporary))
            opened = []
            original_open = model.safe_open

            def observe(path, **options):
                opened.append(path.name)
                return original_open(path, **options)

            with (
                patch.object(model, "safe_open", side_effect=observe),
                patch.object(
                    Qwen3_5MoeForCausalLM,
                    "to",
                    side_effect=AssertionError("No whole-model cast/copy"),
                ),
            ):
                loaded = model.load_model(directory, device="cpu", num_hidden_layers=1)
            self.assertEqual(len(loaded.model.layers), 1)
            self.assertNotIn("text-00002-of-00003.safetensors", opened)
            self.assertEqual(loaded.config.layer_types, ["linear_attention"])
            expected = {
                name: value
                for name, value in originals.items()
                if not name.startswith(preparation.PREFIX + "layers.1.")
            }
            proof = loaded.checkpoint_storage
            self.assertEqual(proof["loaded_source_tensors"], len(expected))
            self.assertEqual(
                proof["source_bytes"],
                sum(value.numel() * value.element_size() for value in expected.values()),
            )
            self.assertEqual(proof["max_cpu_packed_bytes"], 2 * 256 * 128)
            self.assertEqual(proof["checkpoint_layers"], 2)
            self.assertEqual(proof["loaded_layers"], 1)

    def test_rejects_nonfinite_fp8_weights_and_bf16_scales(self):
        for suffix in ("weight", "weight_scale_inv"):
            with self.subTest(suffix=suffix), TemporaryDirectory() as temporary:

                def corrupt(tensors, *_):
                    name = preparation.PREFIX + "layers.0.mlp.experts.0.gate_proj." + suffix
                    tensors[name].reshape(-1)[0] = float("nan")

                directory, _ = self.checkpoint(Path(temporary), corrupt)
                with self.assertRaisesRegex(ValueError, "Nonfinite checkpoint tensor"):
                    model.load_model(directory, device="cpu")

    def test_rejects_invalid_prefix_and_unsupported_quantization_before_construction(self):
        with TemporaryDirectory() as temporary:
            directory, _ = self.checkpoint(Path(temporary))
            for count in (0, 3, True):
                with (
                    self.subTest(prefix=count),
                    patch.object(model, "Qwen3_5MoeForCausalLM") as constructor,
                ):
                    with self.assertRaisesRegex(ValueError, "prefix length"):
                        model.load_model(directory, device="cpu", num_hidden_layers=count)
                    constructor.assert_not_called()
            config_path = directory / "config.json"
            document = json.loads(config_path.read_text())
            document["quantization_config"]["activation_scheme"] = "static"
            config_path.write_text(json.dumps(document))
            with patch.object(model, "Qwen3_5MoeForCausalLM") as constructor:
                with self.assertRaisesRegex(ValueError, "dynamic E4M3"):
                    model.load_model(directory, device="cpu")
                constructor.assert_not_called()


class PinnedHFDependency(unittest.TestCase):
    def test_resolver_is_restored_after_success_failure_and_foreign_cache(self):
        kernel = SimpleNamespace(
            matmul_2d=object(), matmul_batched=object(), matmul_grouped=object()
        )
        original = hf_fp8.lazy_load_kernel

        def official_load():
            resolved = hf_fp8.lazy_load_kernel("finegrained-fp8")
            return SimpleNamespace(
                matmul=resolved.matmul_2d,
                batched_matmul=resolved.matmul_batched,
                grouped_matmul=resolved.matmul_grouped,
            )

        for mode in ("success", "failure", "foreign"):
            with (
                self.subTest(mode=mode),
                patch.object(config, "configure"),
                patch.object(hf_fp8, "is_kernels_available", return_value=True),
                patch("compiler.common.fp8.load_fp8_kernels", return_value=kernel),
            ):
                effect = (
                    official_load
                    if mode == "success"
                    else RuntimeError("load failed")
                    if mode == "failure"
                    else lambda: SimpleNamespace(matmul=None)
                )
                with patch.object(hf_fp8, "load_finegrained_fp8_kernel", side_effect=effect):
                    if mode == "success":
                        self.assertIs(config.configure_hf_fp8().matmul, kernel.matmul_2d)
                    else:
                        with self.assertRaises((RuntimeError, ValueError)):
                            config.configure_hf_fp8()
                self.assertIs(hf_fp8.lazy_load_kernel, original)


if __name__ == "__main__":
    unittest.main()
