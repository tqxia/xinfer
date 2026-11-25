"""CPU preflight rejects incompatible metadata before archive or GPU loading."""

import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from compiler.common.metadata import METADATA_FILE, make_metadata, state_spec
from tools.validation.common.runner import load_aoti_runtime


class ReachedArchiveValidation(Exception):
    pass


class AOTIRuntimeContract(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def metadata(self):
        state = [
            state_spec(0, "conv", [8, 6144, 4], "bfloat16", "replace"),
            state_spec(0, "recurrent", [8, 16, 128, 128], "float32", "replace"),
        ]
        value = make_metadata(
            model_type="qwen35",
            checkpoint=dict(model="Qwen/Qwen3.5-0.8B", revision="fixture"),
            torch_version="2.13.0",
            cxx11_abi=True,
            cuda_compute_capability=[8, 9],
            cache_length=1024,
            vocab_size=100,
            eos_token_ids=[99],
            state=state,
        )
        value.update(
            entries=dict(
                prefill=dict(bytes=1, sha256="1" * 64),
                decode=dict(bytes=1, sha256="1" * 64),
            ),
            shared_weights=dict(
                identity="2" * 64,
                bytes=1,
                sha256="3" * 64,
            ),
        )
        return value

    def write_metadata(self, metadata):
        (self.directory / METADATA_FILE).write_text(json.dumps(metadata))

    def reject(self, metadata, error):
        self.write_metadata(metadata)
        with (
            patch("tools.validation.common.runner.validate_shared_weights") as validate,
            patch("tools.validation.common.runner.SharedWeights") as weights,
        ):
            with self.assertRaisesRegex(ValueError, "^" + re.escape(error)):
                load_aoti_runtime(self.directory, model_type="qwen35", specs=[])
            validate.assert_not_called()
            weights.assert_not_called()

    def test_model_tags_are_required_and_checked_before_loading(self):
        for tag in ("qwen3", "unknown", None):
            with self.subTest(tag=tag):
                metadata = self.metadata()
                metadata["model_type"] = tag
                self.reject(metadata, "Unsupported model type")
        metadata = self.metadata()
        metadata.pop("model_type")
        self.reject(metadata, "Unsupported model type")

    def test_invalid_or_missing_prefill_bounds_are_rejected_before_loading(self):
        metadata = self.metadata()
        metadata["max_prefill_tokens"] = 513
        self.reject(metadata, "Unsupported package metadata: max_prefill_tokens")
        metadata = self.metadata()
        metadata.pop("max_prefill_tokens")
        self.reject(metadata, "Unsupported package metadata: max_prefill_tokens")

    def test_old_aliases_and_retired_fields_are_not_accepted(self):
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
            "eager_numerics",
            "torch",
            "sm",
            "vocabulary",
            "kind",
            "tokens",
            "generation_config",
            "batch_size",
            "dtype",
            "input_mutation",
            "attention_backend",
            "custom_attention",
            "cache_input",
            "kv_extent",
        ):
            with self.subTest(field=field):
                metadata = self.metadata()
                metadata[field] = False
                self.reject(metadata, f"Unsupported package metadata: {field}")

    def test_state_order_count_and_kv_shapes_are_checked_before_loading(self):
        def reverse_state(metadata):
            metadata["state"].reverse()

        def mismatched_kv_shapes(metadata):
            metadata["state"] = [
                state_spec(0, "key", [8, 2, 1024, 256], "bfloat16", "inplace"),
                state_spec(0, "value", [8, 3, 1024, 256], "bfloat16", "inplace"),
            ]

        for mutate, error in (
            (lambda m: m["state"].pop(), "Invalid state count"),
            (reverse_state, "Invalid state order or dtype"),
            (mismatched_kv_shapes, "K/V shapes differ"),
            (
                lambda m: [s.update(layer=1) for s in m["state"]],
                "Invalid state order or dtype",
            ),
        ):
            metadata = self.metadata()
            mutate(metadata)
            self.reject(metadata, error)

    def test_valid_bundle_reaches_archive_validation(self):
        metadata = self.metadata()
        self.write_metadata(metadata)
        with (
            patch(
                "tools.validation.common.runner.validate_shared_weights",
                side_effect=ReachedArchiveValidation,
            ) as validate,
            patch("tools.validation.common.runner.SharedWeights") as weights,
        ):
            with self.assertRaises(ReachedArchiveValidation):
                load_aoti_runtime(self.directory, model_type="qwen35", specs=[])
            validate.assert_called_once_with(self.directory, metadata)
            weights.assert_not_called()


if __name__ == "__main__":
    unittest.main()
