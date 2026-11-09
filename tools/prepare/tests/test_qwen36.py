"""Byte-preserving sharded FP8 preparation, using small CPU-only checkpoints."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import torch
from safetensors.torch import load_file, save_file

from compiler.common.artifacts import MODEL_MANIFEST_NAME, sha256
from tools.prepare import cli, qwen36


def raw(value):
    return value.contiguous().view(torch.uint8).numpy().tobytes()


class PreparedFP8Checkpoint(unittest.TestCase):
    def source(self, directory, mutate=None):
        text = dict(
            model_type="qwen3_5_moe_text",
            tie_word_embeddings=False,
            rope_parameters=dict(rope_type="default"),
            hidden_act="silu",
            attention_bias=False,
            attn_output_gate=True,
            hidden_size=128,
            vocab_size=7,
            num_hidden_layers=2,
            layer_types=["linear_attention", "full_attention"],
            num_experts=2,
            num_experts_per_tok=1,
            moe_intermediate_size=128,
            shared_expert_intermediate_size=128,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=64,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
            max_position_embeddings=1024,
            eos_token_id=3,
        )
        config = dict(
            model_type="qwen3_5_moe",
            tie_word_embeddings=False,
            text_config=text,
            quantization_config=dict(
                quant_method="fp8",
                fmt="e4m3",
                activation_scheme="dynamic",
                weight_block_size=[128, 128],
                modules_to_not_convert=[
                    "model.visual",
                    "lm_head",
                    "model.embed_tokens",
                    "model.language_model.layers.0.linear_attn.in_proj_a",
                    "model.language_model.layers.0.linear_attn.in_proj_ba",
                    "mtp.norm",
                ],
            ),
        )
        generation = dict(eos_token_id=[2, 3], bos_token_id=3, pad_token_id=3)
        tensors = {}
        prefix = qwen36.PREFIX

        def bf16(name, shape):
            tensors[name] = torch.full(shape, (len(tensors) + 1) / 32, dtype=torch.bfloat16)

        def fp8(name, rows, columns):
            # Exact representable values keep the test about bytes, not quantization arithmetic.
            tensors[name + ".weight"] = (
                (torch.arange(rows * columns).reshape(rows, columns) % 16 - 8)
                .float()
                .to(torch.float8_e4m3fn)
            )
            bf16(name + ".weight_scale_inv", (rows // 128, columns // 128))

        bf16(prefix + "embed_tokens.weight", (7, 128))
        bf16(prefix + "norm.weight", (128,))
        bf16("lm_head.weight", (7, 128))
        for layer in range(2):
            base = prefix + f"layers.{layer}."
            bf16(base + "input_layernorm.weight", (128,))
            bf16(base + "post_attention_layernorm.weight", (128,))
            bf16(base + "mlp.gate.weight", (2, 128))
            bf16(base + "mlp.shared_expert_gate.weight", (1, 128))
            for expert in ("experts.0", "experts.1", "shared_expert"):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    fp8(base + f"mlp.{expert}.{projection}", 128, 128)
            if layer == 0:
                bf16(base + "linear_attn.conv1d.weight", (256, 1, 4))
                bf16(base + "linear_attn.norm.weight", (32,))
                for name in ("A_log", "dt_bias"):
                    bf16(base + "linear_attn." + name, (4,))
                for name in ("in_proj_a", "in_proj_b"):
                    bf16(base + f"linear_attn.{name}.weight", (4, 128))
                fp8(base + "linear_attn.in_proj_qkv", 256, 128)
                fp8(base + "linear_attn.in_proj_z", 128, 128)
                fp8(base + "linear_attn.out_proj", 128, 128)
            else:
                for name in ("q_norm", "k_norm"):
                    bf16(base + f"self_attn.{name}.weight", (64,))
                for name, rows in (
                    ("q_proj", 256),
                    ("k_proj", 128),
                    ("v_proj", 128),
                    ("o_proj", 128),
                ):
                    fp8(base + "self_attn." + name, rows, 128)
        if mutate:
            mutate(tensors, config, generation)
        weight_map, by_shard = {}, {}
        for name, value in tensors.items():
            shard = "outside.safetensors"
            for layer in range(2):
                if name.startswith(prefix + f"layers.{layer}."):
                    shard = f"layers-{layer}.safetensors"
            weight_map[name] = shard
            by_shard.setdefault(shard, {})[name] = value
        # Mixed source shards are filtered tensor-by-tensor; unrelated shards are never fetched.
        by_shard["outside.safetensors"]["model.visual.mixed.weight"] = torch.ones(1)
        for name, shard in (
            ("model.visual.only.weight", "vision.safetensors"),
            ("mtp.weight", "mtp.safetensors"),
        ):
            weight_map[name] = shard
            by_shard[shard] = {name: torch.ones(1)}
        for shard, values in by_shard.items():
            save_file(values, directory / shard)
        (directory / "config.json").write_text(json.dumps(config, indent=2))
        (directory / "generation_config.json").write_text(json.dumps(generation))
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            (directory / name).write_text("fixture " + name)
        (directory / "model.safetensors.index.json").write_text(
            json.dumps(dict(weight_map=weight_map))
        )
        siblings = []
        for path in directory.iterdir():
            data = path.read_bytes()
            siblings.append(
                SimpleNamespace(
                    rfilename=path.name,
                    size=len(data),
                    blob_id=hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
                    lfs=SimpleNamespace(sha256=sha256(path))
                    if path.suffix == ".safetensors"
                    else None,
                )
            )
        return SimpleNamespace(sha=qwen36.REVISIONS[qwen36.MODEL_ID], siblings=siblings), tensors

    def network(self, source, info, output, cache_dir=None):
        api, downloads = Mock(), []
        api.model_info.return_value = info

        def download(model, name, **options):
            self.assertEqual(model, qwen36.MODEL_ID)
            self.assertEqual(options, dict(revision=qwen36.REVISIONS[model], cache_dir=cache_dir))
            self.assertFalse(
                output.exists(), "Checkpoint was published before verification completed"
            )
            downloads.append(name)
            return source / name

        self.enterContext(patch("huggingface_hub.HfApi", return_value=api))
        self.enterContext(patch("huggingface_hub.hf_hub_download", side_effect=download))
        return api, downloads

    def test_preserves_all_bytes_config_eos_and_source_mapping_incrementally(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output, cache = root / "source", root / "text", root / "cache"
            source.mkdir()
            info, originals = self.source(source)
            api, downloads = self.network(source, info, output, cache)
            saved, previous = [], []

            def observe(weights, destination):
                self.assertFalse(output.exists())
                self.assertTrue(
                    all(reference() is None for reference in previous),
                    "Prior shard tensors retained",
                )
                previous[:] = [weakref.ref(value) for value in weights.values()]
                saved.append(set(weights))
                save_file(weights, destination)

            with patch("safetensors.torch.save_file", new=observe):
                manifest = qwen36.prepare(output, cache_dir=cache)
            self.assertEqual(len(saved), 3)
            self.assertTrue(all(len(names) < len(originals) for names in saved))
            self.assertTrue(all(reference() is None for reference in previous))
            self.assertNotIn("vision.safetensors", downloads)
            self.assertNotIn("mtp.safetensors", downloads)
            for name in qwen36.MODEL_FILES:
                self.assertEqual((output / name).read_bytes(), (source / name).read_bytes())
            index = json.loads((output / qwen36.INDEX_NAME).read_text())
            ledger = json.loads((output / qwen36.INVENTORY_NAME).read_text())
            inventory = ledger["tensors"]
            self.assertEqual(
                ledger["prepared_modules_to_not_convert"],
                [
                    "model.visual",
                    "lm_head",
                    "model.embed_tokens",
                    "model.layers.0.linear_attn.in_proj_a",
                    "model.layers.0.linear_attn.in_proj_ba",
                    "mtp.norm",
                ],
            )
            expected_map = {
                name: "model." + name.removeprefix(qwen36.PREFIX)
                if name.startswith(qwen36.PREFIX)
                else name
                for name in originals
            }
            self.assertEqual(set(index["weight_map"]), set(expected_map.values()))
            actual = {}
            for shard in set(index["weight_map"].values()):
                actual.update(load_file(output / shard))
            for name, prepared in expected_map.items():
                self.assertEqual(raw(actual[prepared]), raw(originals[name]))
                self.assertEqual(actual[prepared].dtype, originals[name].dtype)
                self.assertEqual(inventory[prepared]["source"], name)
                self.assertEqual(
                    inventory[prepared]["sha256"], hashlib.sha256(raw(originals[name])).hexdigest()
                )
            self.assertNotEqual(
                raw(actual["lm_head.weight"]), raw(actual["model.embed_tokens.weight"])
            )
            self.assertEqual(manifest["tensor_count"], len(originals))
            self.assertEqual(
                manifest["tensor_bytes"],
                sum(value.numel() * value.element_size() for value in originals.values()),
            )
            mapping_json = json.dumps(expected_map, sort_keys=True, separators=(",", ":"))
            self.assertEqual(
                manifest["source_to_prepared_sha256"],
                hashlib.sha256(mapping_json.encode()).hexdigest(),
            )
            self.assertEqual(
                set(manifest["files"]),
                {path.name for path in output.iterdir()} - {MODEL_MANIFEST_NAME},
            )
            api.model_info.reset_mock()
            self.assertEqual(qwen36.verify_model(output), manifest)
            api.model_info.assert_not_called()

    def test_rejects_missing_head_bad_scales_and_wrong_stored_dtypes_without_publish(self):
        gate = qwen36.PREFIX + "layers.0.mlp.experts.0.gate_proj"
        mutations = (
            lambda tensors, *_: tensors.pop("lm_head.weight"),
            lambda tensors, *_: tensors.__setitem__(
                gate + ".weight_scale_inv", torch.ones(2, 1).bfloat16()
            ),
            lambda tensors, *_: tensors.__setitem__(
                gate + ".weight", tensors[gate + ".weight"].bfloat16()
            ),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, output = root / "source", root / "text"
                source.mkdir()
                info, _ = self.source(source, mutation)
                self.network(source, info, output)
                with self.assertRaises(ValueError):
                    qwen36.prepare(output)
                self.assertEqual(set(root.iterdir()), {source})

    def test_hub_corruption_and_late_shard_failure_do_not_publish(self):
        for late in (False, True):
            with self.subTest(late=late), TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, output = root / "source", root / "text"
                source.mkdir()
                mutation = None
                if late:
                    mutation = lambda tensors, *_: tensors.__setitem__(
                        "lm_head.weight", torch.ones(1).bfloat16()
                    )
                info, _ = self.source(source, mutation)
                self.network(source, info, output)
                if not late:
                    path = source / "layers-0.safetensors"
                    data = path.read_bytes()
                    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
                with self.assertRaises(ValueError):
                    qwen36.prepare(output)
                self.assertEqual(set(root.iterdir()), {source})

    def test_verifier_rejects_file_corruption_and_inconsistent_index(self):
        for tamper_index in (False, True):
            with self.subTest(index=tamper_index), TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, output = root / "source", root / "text"
                source.mkdir()
                info, _ = self.source(source)
                self.network(source, info, output)
                manifest = qwen36.prepare(output)
                if tamper_index:
                    path = output / qwen36.INDEX_NAME
                    index = json.loads(path.read_text())
                    index["weight_map"]["lm_head.weight"] = index["weight_map"][
                        "model.layers.0.input_layernorm.weight"
                    ]
                    path.write_text(json.dumps(index))
                    manifest["files"][qwen36.INDEX_NAME] = sha256(path)
                    (output / MODEL_MANIFEST_NAME).write_text(json.dumps(manifest))
                else:
                    path = output / next(
                        name for name in manifest["files"] if name.endswith(".safetensors")
                    )
                    data = path.read_bytes()
                    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
                with self.assertRaises(ValueError):
                    qwen36.verify_model(output)

    def test_cli_registration_default_path_and_existing_output(self):
        self.assertIs(cli.model_hooks(qwen36.MODEL_ID), qwen36)
        with patch.object(qwen36, "prepare", return_value={}) as prepare, patch("builtins.print"):
            cli.main(["download", "--model", qwen36.MODEL_ID, "--cache-dir", "cache"])
        prepare.assert_called_once_with(qwen36.TEXT_DIR, qwen36.MODEL_ID, cache_dir=Path("cache"))
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch("huggingface_hub.HfApi") as api:
                with self.assertRaises(FileExistsError):
                    qwen36.prepare(output)
            api.assert_not_called()

    def test_verifier_checks_raw_digests_eos_and_original_provenance(self):
        for mutation in (
            "raw_digest",
            "payload",
            "eos",
            "source_shard",
            "source_descriptor",
            "skiplist",
        ):
            with self.subTest(mutation=mutation), TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, output = root / "source", root / "text"
                source.mkdir()
                info, _ = self.source(source)
                self.network(source, info, output)
                manifest = qwen36.prepare(output)
                name = "model.layers.0.mlp.experts.0.gate_proj.weight"
                filename = qwen36.INVENTORY_NAME
                data = json.loads((output / filename).read_text())
                if mutation == "raw_digest":
                    data["tensors"][name]["sha256"] = "0" * 64
                elif mutation == "source_shard":
                    data["tensors"][name]["source_shard"] = "outside.safetensors"
                elif mutation == "skiplist":
                    data["prepared_modules_to_not_convert"][3] = (
                        "model.language_model.layers.0.linear_attn.in_proj_a"
                    )
                elif mutation == "source_descriptor":
                    manifest["source_files"]["layers-0.safetensors"]["bytes"] = True
                elif mutation == "eos":
                    filename = "generation_config.json"
                    data = dict(eos_token_id=[3])
                if mutation == "payload":
                    index = json.loads((output / qwen36.INDEX_NAME).read_text())
                    filename = index["weight_map"][name]
                    path = output / filename
                    payload = path.read_bytes()
                    path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
                else:
                    (output / filename).write_text(json.dumps(data))
                manifest["files"][filename] = sha256(output / filename)
                (output / MODEL_MANIFEST_NAME).write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    qwen36.verify_model(output)


if __name__ == "__main__":
    unittest.main()
