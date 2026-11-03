"""CPU preflight rejects incompatible metadata before archive or GPU loading."""

import copy
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from compiler.common.metadata import make_metadata, state_spec
from compiler.qwen35.reference import PythonRuntime


ATTENTION_FIELDS = ("attention_backend", "input_mutation", "cache_input", "kv_extent", "custom_attention", "export_options")
INVALID_VALUES = {
    "attention_backend": "efficient_sdpa",
    "input_mutation": False,
    "cache_input": "full_capacity",
    "kv_extent": {"device": "cpu", "dtype": "uint8", "shape": "[valid_length + 1]", "data_read": True},
    "custom_attention": {"name": "xinfer::flash_attention", "abi_version": 0, "read_only": True},
    "export_options": {"backed_size_oblivious": True},
}


class ReachedArchiveValidation(Exception):
    pass


class PythonRuntimeContract(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def metadata(self):
        state = [state_spec(0, "conv", [1, 6144, 4], "bfloat16", "replace"),
                 state_spec(0, "recurrent", [1, 16, 128, 128], "float32", "replace")]
        pair = {}
        for kind in ("prefill", "decode"):
            value = make_metadata(
                model_type="qwen35", entry_point=kind,
                token_range=[2, 512] if kind == "prefill" else [1, 1],
                checkpoint=dict(model_id="Qwen/Qwen3.5-0.8B", revision="fixture", manifest={}),
                torch_version="2.13.0", torch_cuda_version="12.6", cxx11_abi=True,
                cuda_compute_capability=[8, 9], cache_length=1024, vocab_size=100,
                eos_token_ids=[99], state=state, return_state=True,
                model_specific=dict(delta_backend="triton_op", layer_types=["linear_attention"]))
            value.update(package_bytes=1, package_sha256="1" * 64,
                         shared_weights=dict(format=2, file="weights.pt", identity="2" * 64, bytes=1, sha256="3" * 64),
                         inductor_configs={"emulate_precision_casts": True,
                                           "aot_inductor.package_constants_in_so": False,
                                           "always_keep_tensor_constants": True})
            pair[kind] = value
        return pair

    def write_metadata(self, metadata):
        for kind, value in metadata.items():
            (self.directory / f"{kind}.json").write_text(json.dumps(value))

    def reject(self, metadata, error):
        self.write_metadata(metadata)
        with patch("compiler.qwen35.reference.validate_shared_weights") as validate, \
                patch("compiler.qwen35.reference.load_attention_ops") as load, \
                patch("compiler.qwen35.reference.SharedWeights") as weights:
            with self.assertRaisesRegex(ValueError, "^" + re.escape(error)):
                PythonRuntime(self.directory, [])
            validate.assert_not_called()
            load.assert_not_called()
            weights.assert_not_called()

    def test_missing_contract_fields_are_rejected_before_loading(self):
        for field in ATTENTION_FIELDS:
            with self.subTest(field=field):
                metadata = self.metadata()
                for value in metadata.values():
                    value.pop(field)
                self.reject(metadata, f"Unsupported package metadata: {field}")

    def test_model_tags_are_required_and_checked_before_loading(self):
        for tags, message in ((("qwen3", "qwen3"), "Unsupported model type"),
                              (("unknown", "unknown"), "Unsupported model type"),
                              ((None, None), "Unsupported model type"),
                              (("qwen3", "qwen35"), "Package model types differ")):
            with self.subTest(tags=tags):
                metadata = self.metadata()
                for kind, tag in zip(("prefill", "decode"), tags):
                    metadata[kind]["model_type"] = tag
                self.reject(metadata, message)
        metadata = self.metadata()
        metadata["prefill"].pop("model_type")
        self.reject(metadata, "Package model types differ")

    def test_invalid_matching_contracts_are_rejected_before_loading(self):
        for field, invalid in INVALID_VALUES.items():
            with self.subTest(field=field):
                metadata = self.metadata()
                for value in metadata.values():
                    value[field] = copy.deepcopy(invalid)
                self.reject(metadata, f"Unsupported package metadata: {field}")

    def test_mismatched_contract_fields_identify_the_field(self):
        for field, invalid in INVALID_VALUES.items():
            for kind in ("prefill", "decode"):
                with self.subTest(field=field, kind=kind):
                    metadata = self.metadata()
                    metadata[kind][field] = copy.deepcopy(invalid)
                    self.reject(metadata, f"Package metadata differs: {field}")

    def test_missing_field_in_one_entry_is_a_pair_mismatch(self):
        for field in ATTENTION_FIELDS:
            for kind in ("prefill", "decode"):
                with self.subTest(field=field, kind=kind):
                    metadata = self.metadata()
                    metadata[kind].pop(field)
                    self.reject(metadata, f"Package metadata differs: {field}")

    def test_unversioned_unknown_and_mixed_versions_are_rejected(self):
        for versions in ((None, None), (999, 999), (1, 999), (True, True)):
            with self.subTest(versions=versions):
                metadata = self.metadata()
                for kind, version in zip(("prefill", "decode"), versions):
                    if version is None:
                        metadata[kind].pop("schema_version")
                    else:
                        metadata[kind]["schema_version"] = version
                self.reject(metadata, "Unsupported package metadata schema version")

    def test_old_aliases_are_not_accepted_in_v1(self):
        for field in ("eager_numerics", "torch", "sm", "vocabulary", "kind", "tokens", "generation_config"):
            with self.subTest(field=field):
                metadata = self.metadata()
                for value in metadata.values():
                    value[field] = False
                self.reject(metadata, f"Unsupported package metadata: {field}")

    def test_model_specific_backend_and_state_order_are_checked_before_loading(self):
        def reverse_state(metadata):
            for entry in metadata.values():
                entry["state"].reverse()
                names = [s["name"] for s in entry["state"]]
                entry["input_order"] = ["input_ids", "cache_position", *names, "kv_extent"]
                entry["output_order"] = ["last_logits", *names]

        for mutate, error in (
            (lambda m: [v["model_specific"].update(delta_backend="unknown") for v in m.values()], "Unsupported DeltaNet backend"),
            (lambda m: [v["model_specific"].update(layer_types=[]) for v in m.values()], "Invalid state count"),
            (reverse_state, "Invalid state order or dtype"),
        ):
            metadata = self.metadata()
            mutate(metadata)
            self.reject(metadata, error)

    def test_valid_versioned_pair_reaches_archive_validation(self):
        metadata = self.metadata()
        self.write_metadata(metadata)
        with patch("compiler.qwen35.reference.validate_shared_weights",
                   side_effect=ReachedArchiveValidation) as validate, \
                patch("compiler.qwen35.reference.load_attention_ops") as load, \
                patch("compiler.qwen35.reference.SharedWeights") as weights:
            with self.assertRaises(ReachedArchiveValidation):
                PythonRuntime(self.directory, [])
            validate.assert_called_once_with(self.directory, metadata["prefill"], metadata["decode"])
            load.assert_not_called()
            weights.assert_not_called()


if __name__ == "__main__":
    unittest.main()
