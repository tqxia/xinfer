"""CPU preflight rejects incompatible metadata before archive or GPU loading."""

import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from compiler.common.metadata import make_metadata, state_spec
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
        pair = {}
        for kind in ("prefill", "decode"):
            value = make_metadata(
                model_type="qwen35",
                entry_point=kind,
                token_range=[2, 512] if kind == "prefill" else [1, 1],
                checkpoint=dict(model_id="Qwen/Qwen3.5-0.8B", revision="fixture", manifest={}),
                torch_version="2.13.0",
                cxx11_abi=True,
                cuda_compute_capability=[8, 9],
                cache_length=1024,
                vocab_size=100,
                eos_token_ids=[99],
                state=state,
                model_specific=dict(delta_backend="triton_op"),
            )
            value.update(
                package_bytes=1,
                package_sha256="1" * 64,
                shared_weights=dict(
                    format=3,
                    file="weights.safetensors",
                    identity="2" * 64,
                    bytes=1,
                    sha256="3" * 64,
                ),
                inductor_configs={
                    "emulate_precision_casts": True,
                    "aot_inductor.package_constants_in_so": False,
                    "always_keep_tensor_constants": True,
                },
            )
            pair[kind] = value
        return pair

    def write_metadata(self, metadata):
        for kind, value in metadata.items():
            (self.directory / f"{kind}.json").write_text(json.dumps(value))

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

    def test_missing_export_options_are_rejected_before_loading(self):
        metadata = self.metadata()
        for value in metadata.values():
            value.pop("export_options")
        self.reject(metadata, "Unsupported package metadata: export_options")

    def test_model_tags_are_required_and_checked_before_loading(self):
        for tags, message in (
            (("qwen3", "qwen3"), "Unsupported model type"),
            (("unknown", "unknown"), "Unsupported model type"),
            ((None, None), "Unsupported model type"),
            (("qwen3", "qwen35"), "Package model types differ"),
        ):
            with self.subTest(tags=tags):
                metadata = self.metadata()
                for kind, tag in zip(("prefill", "decode"), tags):
                    metadata[kind]["model_type"] = tag
                self.reject(metadata, message)
        metadata = self.metadata()
        metadata["prefill"].pop("model_type")
        self.reject(metadata, "Package model types differ")

    def test_invalid_matching_export_options_are_rejected_before_loading(self):
        metadata = self.metadata()
        for value in metadata.values():
            value["export_options"] = {"backed_size_oblivious": False}
        self.reject(metadata, "Unsupported package metadata: export_options")

    def test_mismatched_export_options_identify_the_field(self):
        for kind in ("prefill", "decode"):
            with self.subTest(kind=kind):
                metadata = self.metadata()
                metadata[kind]["export_options"] = {"backed_size_oblivious": False}
                self.reject(metadata, "Package metadata differs: export_options")

    def test_missing_export_options_in_one_entry_is_rejected(self):
        for kind in ("prefill", "decode"):
            with self.subTest(kind=kind):
                metadata = self.metadata()
                metadata[kind].pop("export_options")
                self.reject(metadata, "Package metadata differs: export_options")

    def test_old_aliases_are_not_accepted(self):
        for field in (
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
                for value in metadata.values():
                    value[field] = False
                self.reject(metadata, f"Unsupported package metadata: {field}")

    def test_model_specific_backend_and_state_order_are_checked_before_loading(self):
        def reverse_state(metadata):
            for entry in metadata.values():
                entry["state"].reverse()

        def mismatched_kv_shapes(metadata):
            for entry in metadata.values():
                entry["state"] = [
                    state_spec(0, "key", [8, 2, 1024, 256], "bfloat16", "inplace"),
                    state_spec(0, "value", [8, 3, 1024, 256], "bfloat16", "inplace"),
                ]

        for mutate, error in (
            (
                lambda m: [v["model_specific"].update(delta_backend="unknown") for v in m.values()],
                "Unsupported DeltaNet backend",
            ),
            (
                lambda m: [v["state"].pop() for v in m.values()],
                "Invalid state count",
            ),
            (reverse_state, "Invalid state order or dtype"),
            (mismatched_kv_shapes, "K/V shapes differ"),
            (
                lambda m: [s.update(layer=1) for v in m.values() for s in v["state"]],
                "Invalid state order or dtype",
            ),
        ):
            metadata = self.metadata()
            mutate(metadata)
            self.reject(metadata, error)

    def test_valid_pair_reaches_archive_validation(self):
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
            validate.assert_called_once_with(
                self.directory, metadata["prefill"], metadata["decode"]
            )
            weights.assert_not_called()


if __name__ == "__main__":
    unittest.main()
