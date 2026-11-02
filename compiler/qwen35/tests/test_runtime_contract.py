"""CPU preflight checks reject incompatible packages before native/GPU loading."""

import copy
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from compiler.qwen35.reference import PythonRuntime


ATTENTION_CONTRACT = {
    "attention_backend": "flash",
    "input_mutation": "kv_only",
    "cache_input": "full_capacity_with_extent",
    "kv_extent": {"device": "cpu", "dtype": "uint8", "shape": "[valid_length + 1]", "data_read": False},
    "custom_attention": {"name": "xinfer::flash_attention", "abi_version": 1, "read_only": True},
    "export_options": {"backed_size_oblivious": False},
}
INVALID_VALUES = {
    "attention_backend": "efficient_sdpa",
    "input_mutation": False,
    "cache_input": "full_capacity",
    "kv_extent": dict(ATTENTION_CONTRACT["kv_extent"], data_read=True),
    "custom_attention": dict(ATTENTION_CONTRACT["custom_attention"], abi_version=0),
    "export_options": {"backed_size_oblivious": True},
}
LEGACY_ATTENTION = "Legacy attention or state contract is unsupported"
LEGACY_EAGER = "Legacy eager-numerics packages are unsupported"


class ReachedArchiveValidation(Exception):
    pass


class PythonRuntimeContract(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def metadata(self):
        return {
            kind: dict(copy.deepcopy(ATTENTION_CONTRACT), kind=kind, delta_backend="triton_op",
                       tokens=None if kind == "prefill" else 1,
                       token_range=[2, 512] if kind == "prefill" else [1, 1], shared_weights={})
            for kind in ("prefill", "decode")
        }

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
        for field in ATTENTION_CONTRACT:
            with self.subTest(field=field):
                metadata = self.metadata()
                for value in metadata.values():
                    value.pop(field)
                self.reject(metadata, LEGACY_ATTENTION)

    def test_invalid_matching_contracts_are_rejected_before_loading(self):
        for field, invalid in INVALID_VALUES.items():
            with self.subTest(field=field):
                metadata = self.metadata()
                for value in metadata.values():
                    value[field] = copy.deepcopy(invalid)
                self.reject(metadata, LEGACY_ATTENTION)

    def test_mismatched_contract_fields_identify_the_field(self):
        for field, invalid in INVALID_VALUES.items():
            for kind in ("prefill", "decode"):
                with self.subTest(field=field, kind=kind):
                    metadata = self.metadata()
                    metadata[kind][field] = copy.deepcopy(invalid)
                    self.reject(metadata, f"Package metadata differs: {field}")

    def test_missing_field_in_one_entry_is_a_pair_mismatch(self):
        for field in ATTENTION_CONTRACT:
            for kind in ("prefill", "decode"):
                with self.subTest(field=field, kind=kind):
                    metadata = self.metadata()
                    metadata[kind].pop(field)
                    self.reject(metadata, f"Package metadata differs: {field}")

    def test_eager_numerics_is_rejected_in_either_entry_or_both(self):
        for prefill, decode in ((True, False), (False, True), (True, True)):
            with self.subTest(prefill=prefill, decode=decode):
                metadata = self.metadata()
                metadata["prefill"]["eager_numerics"] = prefill
                metadata["decode"]["eager_numerics"] = decode
                self.reject(metadata, LEGACY_EAGER)

    def test_valid_contract_reaches_archive_validation_with_absent_or_false_eager_flag(self):
        for prefill, decode in ((None, None), (False, None), (None, False), (False, False)):
            with self.subTest(prefill=prefill, decode=decode):
                metadata = self.metadata()
                for kind, eager in (("prefill", prefill), ("decode", decode)):
                    if eager is not None:
                        metadata[kind]["eager_numerics"] = eager
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
