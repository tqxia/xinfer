"""The same metadata contract describes both model families before loading."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from compiler.common.artifacts import ROOT, sha256
from compiler.common.metadata import (
    MAX_CACHE_LENGTH,
    METADATA_FILE,
    finalize_metadata,
    is_kv_state,
    make_metadata,
    read_package_metadata,
    state_spec,
    validate_metadata,
)


def bundle_metadata(model_type="qwen3", *, cache_length=1024):
    if model_type == "qwen3":
        shape = [8, 8, cache_length, 128]
        states = [
            state_spec(i, kind, shape, "bfloat16") for i in range(28) for kind in ("key", "value")
        ]
    else:
        states = [
            state_spec(0, "conv", [8, 6144, 4], "bfloat16"),
            state_spec(0, "recurrent", [8, 16, 128, 128], "float32"),
        ]
        states += [
            state_spec(1, kind, [8, 2, cache_length, 256], "bfloat16") for kind in ("key", "value")
        ]
    metadata = make_metadata(
        model_type=model_type,
        checkpoint=dict(model=model_type, revision="fixture-revision"),
        torch_version="2.13.0+cu126",
        cxx11_abi=True,
        cuda_compute_capability=[8, 9],
        cache_length=cache_length,
        vocab_size=151936,
        eos_token_ids=[151645],
        state=states,
    )
    metadata.update(
        entries=dict(
            prefill=dict(bytes=100, sha256="a" * 64),
            decode=dict(bytes=100, sha256="a" * 64),
        ),
        shared_weights=dict(
            identity="1" * 64,
            bytes=1000000,
            sha256="b" * 64,
        ),
        tokenizer={
            "tokenizer.json": dict(bytes=256, sha256="c" * 64),
            "tokenizer_config.json": dict(bytes=128, sha256="d" * 64),
        },
    )
    return metadata


class PackageMetadata(unittest.TestCase):
    def test_one_format_handles_all_model_types_with_unified_inplace_abi(self):
        for model_type in ("qwen3", "qwen35", "qwen38"):
            with self.subTest(model_type=model_type):
                metadata = bundle_metadata(model_type)
                validate_metadata(metadata, model_type)
                for other in ("qwen3", "qwen35", "qwen38"):
                    if other != model_type:
                        with self.assertRaisesRegex(ValueError, "Unsupported model type"):
                            validate_metadata(metadata, other)
                self.assertTrue(
                    all(
                        set(item) == {"layer", "kind", "shape", "dtype"}
                        for item in metadata["state"]
                    )
                )
                self.assertEqual(metadata["eos_token_ids"], [151645])
        qwen35 = bundle_metadata("qwen35")
        self.assertEqual(
            [is_kv_state(item) for item in qwen35["state"]], [False, False, True, True]
        )

    def test_both_models_reserve_eight_rows_and_reject_other_allocations(self):
        for model_type in ("qwen3", "qwen35", "qwen38"):
            metadata = bundle_metadata(model_type)
            validate_metadata(metadata)
            for batch_size in (1, 2, 4, 7, 9):
                invalid = copy.deepcopy(metadata)
                invalid["state"][0]["shape"][0] = batch_size
                with (
                    self.subTest(model=model_type, batch_size=batch_size),
                    self.assertRaisesRegex(ValueError, "state"),
                ):
                    validate_metadata(invalid)

    def test_capacity_is_separate_from_compiled_call_bounds_and_prompt_policy(self):
        for model_type in ("qwen3", "qwen35", "qwen38"):
            for capacity in (512, 1024, 16384):
                with self.subTest(model=model_type, capacity=capacity):
                    metadata = bundle_metadata(model_type, cache_length=capacity)
                    validate_metadata(metadata)
                    self.assertEqual(metadata["max_prefill_tokens"], 512)
                    self.assertEqual(metadata["cache_length"], capacity)
                    self.assertNotIn("max_prompt_length", metadata)

    def test_capacity_is_a_required_integer_contract(self):
        for invalid in (0, -1, 511, True, 1024.0, "1024", MAX_CACHE_LENGTH + 1, 2**63 - 1, 2**63):
            with self.subTest(value=invalid):
                metadata = bundle_metadata()
                metadata["cache_length"] = invalid
                with self.assertRaisesRegex(
                    ValueError, "Unsupported package metadata: cache_length"
                ):
                    validate_metadata(metadata)
        validate_metadata(bundle_metadata(cache_length=MAX_CACHE_LENGTH))
        metadata = bundle_metadata()
        metadata["max_prompt_length"] = 8192
        with self.assertRaisesRegex(ValueError, "Unsupported package metadata: max_prompt_length"):
            validate_metadata(metadata)
        metadata = bundle_metadata()
        metadata["cache_length"] = 16384
        with self.assertRaisesRegex(ValueError, "Unsupported package metadata: state"):
            validate_metadata(metadata)

    def test_larger_capacity_does_not_advertise_a_larger_compiled_graph(self):
        metadata = bundle_metadata(cache_length=16384)
        metadata["max_prefill_tokens"] = 8192
        with self.assertRaisesRegex(ValueError, "Unsupported package metadata: max_prefill_tokens"):
            validate_metadata(metadata)

    def test_legacy_aliases_retired_fields_and_missing_fields_are_rejected(self):
        for field in (
            "entry_point",
            "token_range",
            "export_options",
            "tf32",
            "triton_enable_fp_fusion",
            "inductor_configs",
            "package_bytes",
            "package_sha256",
            "model_specific",
            "torch",
            "sm",
            "vocabulary",
            "kind",
            "generation_config",
            "model_revision",
            "input_order",
            "output_order",
            "logits",
            "provenance",
            "diagnostics",
            "torch_cuda_version",
            "batch_size",
            "dtype",
            "input_mutation",
            "attention_backend",
            "custom_attention",
            "cache_input",
            "kv_extent",
            "max_prompt_length",
        ):
            with self.subTest(alias=field):
                metadata = bundle_metadata()
                metadata[field] = "legacy"
                with self.assertRaisesRegex(ValueError, f"Unsupported package metadata: {field}"):
                    validate_metadata(metadata)
        for field in (
            "entries",
            "state",
            "checkpoint",
            "cache_length",
            "max_prefill_tokens",
            "shared_weights",
            "tokenizer",
        ):
            with self.subTest(missing=field):
                metadata = bundle_metadata()
                metadata.pop(field)
                with self.assertRaisesRegex(ValueError, f"Unsupported package metadata: {field}"):
                    validate_metadata(metadata)

    def test_checkpoint_entries_and_shared_weights_contracts(self):
        for bad_checkpoint in (
            {},
            {"model_id": "Qwen/Qwen3-0.6B", "revision": "abc", "manifest": {}},
            {"model": "", "revision": "abc"},
            {"model": "Qwen/Qwen3-0.6B", "revision": ""},
        ):
            with self.subTest(checkpoint=bad_checkpoint):
                metadata = bundle_metadata()
                metadata["checkpoint"] = bad_checkpoint
                with self.assertRaisesRegex(ValueError, "Unsupported package metadata: checkpoint"):
                    validate_metadata(metadata)
        for bad_entries in (
            {},
            {"prefill": dict(bytes=1, sha256="a" * 64)},
            {
                "prefill": dict(bytes=0, sha256="a" * 64),
                "decode": dict(bytes=1, sha256="a" * 64),
            },
            {
                "prefill": dict(bytes=1, sha256="bad"),
                "decode": dict(bytes=1, sha256="a" * 64),
            },
        ):
            with self.subTest(entries=bad_entries):
                metadata = bundle_metadata()
                metadata["entries"] = bad_entries
                with self.assertRaisesRegex(ValueError, "Unsupported package metadata: entries"):
                    validate_metadata(metadata)
        metadata = bundle_metadata()
        metadata["shared_weights"]["file"] = "weights.safetensors"
        with self.assertRaisesRegex(ValueError, "Unsupported package metadata: shared_weights"):
            validate_metadata(metadata)
        optional = bundle_metadata()
        optional["tokenizer"]["chat_template.jinja"] = dict(bytes=64, sha256="e" * 64)
        validate_metadata(optional)
        for bad_tokenizer in (
            {},
            {"tokenizer.json": dict(bytes=1, sha256="a" * 64)},
            {
                "tokenizer.json": dict(bytes=1, sha256="a" * 64),
                "tokenizer_config.json": dict(bytes=1, sha256="a" * 64),
                "merges.txt": dict(bytes=1, sha256="a" * 64),
            },
            {
                "tokenizer.json": dict(bytes=0, sha256="a" * 64),
                "tokenizer_config.json": dict(bytes=1, sha256="a" * 64),
            },
            {
                "tokenizer.json": dict(bytes=1, sha256="bad"),
                "tokenizer_config.json": dict(bytes=1, sha256="a" * 64),
            },
            {
                "tokenizer.json": dict(bytes=True, sha256="a" * 64),
                "tokenizer_config.json": dict(bytes=1, sha256="a" * 64),
            },
        ):
            with self.subTest(tokenizer=bad_tokenizer):
                metadata = bundle_metadata()
                metadata["tokenizer"] = bad_tokenizer
                with self.assertRaisesRegex(ValueError, "Unsupported package metadata: tokenizer"):
                    validate_metadata(metadata)

    def test_state_contract_rejects_bad_bounds_legacy_layout_and_mutation(self):
        changes = [
            lambda s: s.update(shape=[1, 8, 1024, 2**62]),
            lambda s: s["shape"].__setitem__(1, 0),
            lambda s: s.update(stride=[1]),
            lambda s: s.update(dtype="float32"),
            lambda s: s.update(update="inplace"),
            lambda s: s.update(update="replace"),
            lambda s: s.update(name="other.key"),
            lambda s: s.update(layer=True),
        ]
        for change in changes:
            metadata = bundle_metadata()
            change(metadata["state"][0])
            with self.assertRaisesRegex(ValueError, "Unsupported package metadata: state"):
                validate_metadata(metadata)
        for index in (0, 1, 2, 3):
            metadata = bundle_metadata("qwen35")
            metadata["state"][index]["update"] = "replace"
            with self.subTest(index=index):
                with self.assertRaisesRegex(ValueError, "Unsupported package metadata: state"):
                    validate_metadata(metadata)

    def test_duplicate_state_identity_is_rejected(self):
        metadata = bundle_metadata()
        metadata["state"][1] = copy.deepcopy(metadata["state"][0])
        with self.assertRaisesRegex(ValueError, "Unsupported package metadata: state"):
            validate_metadata(metadata)

    def test_json_booleans_are_not_accepted_as_numbers(self):
        for name, mutate in (
            ("entries", lambda m: m["entries"]["decode"].update(bytes=True)),
            ("max_prefill_tokens", lambda m: m.update(max_prefill_tokens=True)),
            ("eos_token_ids", lambda m: m.update(eos_token_ids=[True])),
            ("cuda_compute_capability", lambda m: m.update(cuda_compute_capability=[True, 9])),
        ):
            with self.subTest(field=name):
                metadata = bundle_metadata()
                mutate(metadata)
                with self.assertRaisesRegex(ValueError, f"Unsupported package metadata: {name}"):
                    validate_metadata(metadata)

    def test_writer_owns_file_facts_and_reader_is_cpu_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = bundle_metadata()
            for kind in ("prefill", "decode"):
                (root / f"{kind}.pt2").write_bytes(kind.encode())
            for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
                (root / name).write_bytes(name.encode())
            complete = finalize_metadata(
                metadata,
                directory=root,
                shared_weights=metadata["shared_weights"],
            )
            for kind in ("prefill", "decode"):
                package = root / f"{kind}.pt2"
                self.assertEqual(complete["entries"][kind]["bytes"], len(kind))
                self.assertEqual(complete["entries"][kind]["sha256"], sha256(package))
            for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
                path = root / name
                self.assertEqual(complete["tokenizer"][name]["bytes"], len(name))
                self.assertEqual(complete["tokenizer"][name]["sha256"], sha256(path))
            (root / METADATA_FILE).write_text(json.dumps(complete))
            loaded = read_package_metadata(root)
            self.assertEqual(loaded, complete)
            code = (
                "import sys\nfrom compiler.common.metadata import read_package_metadata\n"
                "read_package_metadata(sys.argv[1])\n"
                "if 'torch' in sys.modules: raise RuntimeError('reader imported Torch')\n"
            )
            subprocess.run([sys.executable, "-O", "-c", code, str(root)], cwd=ROOT, check=True)


if __name__ == "__main__":
    unittest.main()
