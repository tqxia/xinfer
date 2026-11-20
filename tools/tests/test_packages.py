"""Reject incompatible model ABIs and shared-weight bundles before native loading."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from compiler.common.artifacts import json_sha256, sha256
from compiler.common.metadata import make_metadata, state_spec, validate_metadata
from compiler.common.tests.test_metadata import metadata_pair
from tools.packages import validate_model_metadata, validate_shared_weights


def metadata(*, cache_length=1024):
    shape = [8, 8, cache_length, 128]
    state = [state_spec(0, kind, shape, "bfloat16", "inplace") for kind in ("key", "value")]
    result = make_metadata(
        model_type="qwen3",
        entry_point="decode",
        token_range=[1, 1],
        checkpoint=dict(model_id="Qwen/Qwen3-0.6B", revision="pinned", manifest={}),
        torch_version="2.13.0+cu126",
        cxx11_abi=True,
        cuda_compute_capability=[8, 9],
        cache_length=cache_length,
        vocab_size=151936,
        eos_token_ids=[151645],
        state=state,
        model_specific={},
    )
    result.update(
        package_bytes=128,
        package_sha256="a" * 64,
        inductor_configs={
            "emulate_precision_casts": True,
            "aot_inductor.package_constants_in_so": False,
            "always_keep_tensor_constants": True,
        },
        shared_weights=dict(
            format=2, file="weights.pt", identity="b" * 64, bytes=128, sha256="c" * 64
        ),
    )
    return result


class Qwen3PackageContract(unittest.TestCase):
    def reject(self, value, message):
        with self.assertRaisesRegex(ValueError, "^" + message):
            validate_metadata(value, model_type="qwen3")
            validate_model_metadata(value)

    def test_cache_capacity_is_derived_from_metadata(self):
        for capacity in (512, 1024, 16384):
            item = metadata(cache_length=capacity)
            validate_metadata(item, model_type="qwen3")
            validate_model_metadata(item)
        item = metadata(cache_length=16384)
        item["state"][0]["shape"][2] = 1024
        self.reject(item, "Unsupported package metadata: state")

    def test_invalid_geometry_is_rejected(self):
        for shape in ([8, 4, 1024, 128], [8, 8, 1024, 64]):
            with self.subTest(shape=shape):
                item = metadata()
                for state in item["state"]:
                    state["shape"] = shape
                self.reject(item, "Unsupported Flash cache shape")

    def test_removed_geometry_fields_are_rejected(self):
        for field in (
            "layers",
            "kv_heads",
            "head_dim",
            "cache_padding",
            "cache_shape",
            "cache_stride",
        ):
            with self.subTest(field=field):
                item = metadata()
                item["model_specific"][field] = True
                self.reject(item, "Unsupported package metadata: model_specific")

    def test_qwen3_tag_does_not_accept_functional_deltanet_state(self):
        item = metadata()
        item["state"] = [
            state_spec(0, "conv", [8, 6144, 4], "bfloat16", "replace"),
            state_spec(0, "recurrent", [8, 16, 128, 128], "float32", "replace"),
        ]
        self.reject(item, "Unsupported package metadata: state")

    def test_state_order_count_and_layer_indices_are_model_specific(self):
        for mutate in (
            lambda state: state.reverse(),
            lambda state: state.pop(),
            lambda state: [s.update(layer=1) for s in state],
        ):
            item = metadata()
            mutate(item["state"])
            self.reject(item, "Unsupported package metadata: state")
        item = metadata()
        item["output_order"] = ["last_logits"]
        self.reject(item, "Unsupported package metadata: output_order")


class SharedWeightBundleContract(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.bundle = Path(temporary.name)
        for kind in ("prefill", "decode"):
            (self.bundle / f"{kind}.pt2").write_bytes(f"{kind} package fixture".encode())
        self.write_archive("cpu-tied-view-fixture")
        self.metadata = {
            key: value
            for key, value in self.model_metadata()["decode"].items()
            if key in ("shared_weights", "inductor_configs")
        }

    def write_archive(self, model):
        # Preflight reads ZIP metadata and file bytes; it does not load tensors or runners.
        identity = dict(model=model, revision="cpu-fixture")
        self.manifest = dict(
            format=2,
            identity=identity,
            identity_sha256=json_sha256(identity),
            entries={
                kind: dict(package_sha256=sha256(self.bundle / f"{kind}.pt2"), constants={})
                for kind in ("prefill", "decode")
            },
            tensors={},
            storages={},
            sources={},
        )
        path = self.bundle / "weights.pt"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("weights/extra/weights.json", json.dumps(self.manifest))
        self.contract = dict(
            format=2,
            file="weights.pt",
            identity=self.manifest["identity_sha256"],
            bytes=path.stat().st_size,
            sha256=sha256(path),
        )

    def model_metadata(self, model_type="qwen3"):
        pair = metadata_pair(model_type)
        for kind, value in pair.items():
            package = self.bundle / f"{kind}.pt2"
            value.update(
                package_bytes=package.stat().st_size,
                package_sha256=sha256(package),
                shared_weights=self.contract,
            )
        return pair

    def test_complete_bundles_return_the_verified_manifest(self):
        self.assertEqual(
            validate_shared_weights(self.bundle, self.metadata, self.metadata), self.manifest
        )
        for model, model_type in (("qwen3-0.6b", "qwen3"), ("qwen3.5", "qwen35")):
            with self.subTest(model=model):
                self.write_archive(model)
                pair = self.model_metadata(model_type)
                self.assertEqual(
                    validate_shared_weights(self.bundle, pair["prefill"], pair["decode"]),
                    self.manifest,
                )

    def test_model_file_facts_fail_before_archive_access(self):
        for kind in ("prefill", "decode"):
            for field, value, error in (
                ("package_bytes", 1, "Package file size mismatch"),
                ("package_sha256", "0" * 64, "Package checksum mismatch"),
            ):
                with self.subTest(kind=kind, field=field):
                    pair = self.model_metadata()
                    pair[kind][field] = value
                    with patch("compiler.common.weights.read_shared_weight_manifest") as archive:
                        with self.assertRaisesRegex(ValueError, error):
                            validate_shared_weights(self.bundle, pair["prefill"], pair["decode"])
                        archive.assert_not_called()

    def test_model_metadata_rejects_a_different_archive_model(self):
        pair = self.model_metadata()
        with self.assertRaisesRegex(ValueError, "Package model identity mismatch"):
            validate_shared_weights(self.bundle, pair["prefill"], pair["decode"])

    def test_contract_and_compiler_options_cannot_select_another_pool(self):
        altered = copy.deepcopy(self.metadata)
        altered["shared_weights"]["identity"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "contracts differ"):
            validate_shared_weights(self.bundle, altered, self.metadata)
        with self.assertRaisesRegex(ValueError, "metadata is required"):
            validate_shared_weights(self.bundle, {}, self.metadata)
        altered = copy.deepcopy(self.metadata)
        altered["inductor_configs"]["aot_inductor.package_constants_in_so"] = True
        with self.assertRaisesRegex(ValueError, "weightless package compiler options"):
            validate_shared_weights(self.bundle, altered, altered)


if __name__ == "__main__":
    unittest.main()
