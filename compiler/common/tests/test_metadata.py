"""The same versioned contract describes both model families before loading."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from compiler.common.artifacts import ROOT, sha256
from compiler.common.metadata import (
    finalize_metadata, make_metadata, read_package_metadata, state_spec,
    validate_metadata, validate_metadata_pair,
)


def metadata_pair(model_type="qwen3"):
    if model_type == "qwen3":
        shape = [1, 8, 1024, 128]
        states = [state_spec(i, kind, shape, "bfloat16", "inplace")
                  for i in range(28) for kind in ("key", "value")]
        specific = dict(layers=28, kv_heads=8, head_dim=128, cache_shape=shape,
                        cache_stride=states[0]["stride"], cache_padding=0, flat_input_count=59)
    else:
        kinds = ["linear_attention", "full_attention"]
        states = [state_spec(0, "conv", [1, 6144, 4], "bfloat16", "replace"),
                  state_spec(0, "recurrent", [1, 16, 128, 128], "float32", "replace")]
        states += [state_spec(1, kind, [1, 2, 1024, 256], "bfloat16", "inplace")
                   for kind in ("key", "value")]
        specific = dict(layer_types=kinds, hidden_size=1024, delta_backend="triton_op")
    result = {}
    for kind in ("prefill", "decode"):
        metadata = make_metadata(
            model_type=model_type, entry_point=kind, token_range=[2, 512] if kind == "prefill" else [1, 1],
            checkpoint=dict(model_id=model_type, revision="fixture-revision", manifest={}),
            torch_version="2.13.0+cu126", torch_cuda_version="12.6", cxx11_abi=True,
            cuda_compute_capability=[8, 9], cache_length=1024, vocab_size=151936,
            eos_token_ids=[151645], state=states, return_state=model_type == "qwen35", model_specific=specific)
        metadata.update(package_bytes=100, package_sha256="a" * 64,
                        inductor_configs={"emulate_precision_casts": True,
                                          "aot_inductor.package_constants_in_so": False,
                                          "always_keep_tensor_constants": True},
                        shared_weights=dict(format=2, file="weights.pt", identity="1" * 64,
                                            bytes=1000000, sha256="b" * 64))
        result[kind] = metadata
    return result


class PackageMetadata(unittest.TestCase):
    def test_one_format_handles_both_output_state_abis(self):
        for model_type in ("qwen3", "qwen35"):
            with self.subTest(model_type=model_type):
                pair = metadata_pair(model_type)
                validate_metadata_pair(pair["prefill"], pair["decode"], model_type)
                count = 1 if model_type == "qwen3" else 1 + len(pair["decode"]["state"])
                self.assertEqual(len(pair["decode"]["output_order"]), count)
                self.assertEqual(pair["decode"]["eos_token_ids"], [151645])

    def test_schema_versions_are_required_and_do_not_alias_weight_format(self):
        for version in (None, 0, 2, True, 1.0, "1"):
            with self.subTest(version=version):
                pair = metadata_pair()
                if version is None:
                    pair["decode"].pop("schema_version")
                else:
                    pair["decode"]["schema_version"] = version
                with self.assertRaisesRegex(ValueError, "Unsupported package metadata schema version"):
                    validate_metadata_pair(pair["prefill"], pair["decode"])

    def test_model_identity_and_stable_extras_must_match(self):
        for field, value in (("model_type", "qwen35"), ("model_specific", {}),
                             ("checkpoint", {}), ("provenance", {"source": "changed"})):
            with self.subTest(field=field):
                pair = metadata_pair()
                pair["decode"][field] = value
                message = "Package model types differ" if field == "model_type" else f"Package metadata differs: {field}"
                with self.assertRaisesRegex(ValueError, message):
                    validate_metadata_pair(pair["prefill"], pair["decode"])

    def test_only_entry_facts_may_differ(self):
        pair = metadata_pair("qwen35")
        pair["prefill"].update(package_bytes=200, package_sha256="c" * 64,
                                diagnostics={"delta_graph": {"cond": 18}})
        pair["decode"]["diagnostics"] = {"delta_graph": {"cond": 0}}
        validate_metadata_pair(pair["prefill"], pair["decode"])

    def test_legacy_aliases_and_incomplete_common_fields_are_rejected(self):
        for field in ("torch", "sm", "vocabulary", "kind", "generation_config", "model_revision"):
            with self.subTest(alias=field):
                metadata = metadata_pair()["decode"]
                metadata[field] = "legacy"
                with self.assertRaisesRegex(ValueError, f"Unsupported package metadata: {field}"):
                    validate_metadata(metadata)
        for field in ("package_bytes", "state", "model_specific", "checkpoint", "input_order"):
            with self.subTest(missing=field):
                metadata = metadata_pair()["decode"]
                metadata.pop(field)
                with self.assertRaisesRegex(ValueError, f"Unsupported package metadata: {field}"):
                    validate_metadata(metadata)

    def test_tensor_storage_contract_rejects_bad_bounds_strides_and_mutation(self):
        changes = [lambda s: s.update(shape=[1, 8, 1024, 2**62]),
                   lambda s: s["shape"].__setitem__(1, 0),
                   lambda s: s["stride"].__setitem__(0, 1),
                   lambda s: s.update(dtype="float32"), lambda s: s.update(update="replace"),
                   lambda s: s.update(name="other.key"), lambda s: s.update(layer=True)]
        for change in changes:
            metadata = metadata_pair()["decode"]
            change(metadata["state"][0])
            with self.assertRaisesRegex(ValueError, "Unsupported package metadata: state"):
                validate_metadata(metadata)

    def test_json_booleans_are_not_accepted_as_numbers(self):
        changes = [("batch_size", True), ("package_bytes", True), ("eos_token_ids", [True]),
                   ("cuda_compute_capability", [True, 9]),
                   ("custom_attention", dict(name="xinfer::flash_attention", abi_version=True, read_only=True)),
                   ("kv_extent", dict(device="cpu", dtype="uint8", shape="[valid_length + 1]", data_read=0))]
        for field, value in changes:
            with self.subTest(field=field):
                metadata = metadata_pair()["decode"]
                metadata[field] = value
                with self.assertRaisesRegex(ValueError, f"Unsupported package metadata: {field}"):
                    validate_metadata(metadata)

    def test_writer_owns_file_facts_and_pair_reader_is_cpu_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair = metadata_pair()
            for kind, metadata in pair.items():
                package = root / f"{kind}.pt2"
                package.write_bytes(kind.encode())
                complete = finalize_metadata(metadata, package_path=package,
                                             inductor_configs=metadata["inductor_configs"],
                                             shared_weights=metadata["shared_weights"])
                self.assertEqual(complete["package_bytes"], len(kind))
                self.assertEqual(complete["package_sha256"], sha256(package))
                (root / f"{kind}.json").write_text(json.dumps(complete))
            self.assertEqual(set(read_package_metadata(root)), {"prefill", "decode"})
            code = ("import sys\nfrom compiler.common.metadata import read_package_metadata\n"
                    "read_package_metadata(sys.argv[1])\n"
                    "if 'torch' in sys.modules: raise RuntimeError('reader imported Torch')\n")
            subprocess.run([sys.executable, "-O", "-c", code, str(root)], cwd=ROOT, check=True)


if __name__ == "__main__":
    unittest.main()
