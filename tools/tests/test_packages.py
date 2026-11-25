"""Reject incompatible model ABIs and shared-weight bundles before native loading."""

import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from compiler.common.artifacts import json_sha256, sha256
from compiler.common.metadata import make_metadata, state_spec, validate_metadata
from compiler.common.tests.test_metadata import bundle_metadata
from tools.packages import validate_model_metadata, validate_shared_weights


def metadata(*, cache_length=1024):
    shape = [8, 8, cache_length, 128]
    state = [state_spec(0, kind, shape, "bfloat16", "inplace") for kind in ("key", "value")]
    result = make_metadata(
        model_type="qwen3",
        checkpoint=dict(model="Qwen/Qwen3-0.6B", revision="pinned"),
        torch_version="2.13.0+cu126",
        cxx11_abi=True,
        cuda_compute_capability=[8, 9],
        cache_length=cache_length,
        vocab_size=151936,
        eos_token_ids=[151645],
        state=state,
    )
    result.update(
        entries=dict(
            prefill=dict(bytes=128, sha256="a" * 64),
            decode=dict(bytes=128, sha256="a" * 64),
        ),
        shared_weights=dict(
            identity="b" * 64,
            bytes=128,
            sha256="c" * 64,
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

    def test_geometry_accepts_additional_sizes_and_rejects_mismatched_kv_shapes(self):
        for shape in ([8, 4, 1024, 128], [8, 8, 1024, 64], [8, 8, 1024, 256]):
            with self.subTest(shape=shape):
                item = metadata()
                for state in item["state"]:
                    state["shape"] = shape
                validate_metadata(item, model_type="qwen3")
                validate_model_metadata(item)
        item = metadata()
        item["state"][0]["shape"] = [8, 4, 1024, 128]
        self.reject(item, "K/V shapes differ")

    def test_removed_geometry_and_model_specific_fields_are_rejected(self):
        for field in (
            "model_specific",
            "layers",
            "kv_heads",
            "head_dim",
            "cache_padding",
            "cache_shape",
            "cache_stride",
        ):
            with self.subTest(field=field):
                item = metadata()
                item[field] = True
                self.reject(item, f"Unsupported package metadata: {field}")

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
        self.metadata = dict(shared_weights=self.contract)

    def write_archive(self, model):
        # Preflight reads safetensors header metadata and file bytes without loading tensors.
        identity = dict(model=model, revision="cpu-fixture")
        self.manifest = dict(
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
        path = self.bundle / "weights.safetensors"
        header = json.dumps(
            {"__metadata__": {"xinfer_manifest": json.dumps(self.manifest)}}
        ).encode("utf-8")
        path.write_bytes(struct.pack("<Q", len(header)) + header)
        self.contract = dict(
            identity=self.manifest["identity_sha256"],
            bytes=path.stat().st_size,
            sha256=sha256(path),
        )

    def model_metadata(self, model_type="qwen3"):
        value = bundle_metadata(model_type)
        value["entries"] = {
            kind: dict(
                bytes=(self.bundle / f"{kind}.pt2").stat().st_size,
                sha256=sha256(self.bundle / f"{kind}.pt2"),
            )
            for kind in ("prefill", "decode")
        }
        value["shared_weights"] = self.contract
        return value

    def test_complete_bundles_return_the_verified_manifest(self):
        self.assertEqual(validate_shared_weights(self.bundle, self.metadata), self.manifest)
        for model, model_type in (("qwen3", "qwen3"), ("qwen3.5", "qwen35")):
            with self.subTest(model=model):
                self.write_archive(model)
                meta = self.model_metadata(model_type)
                self.assertEqual(validate_shared_weights(self.bundle, meta), self.manifest)

    def test_model_file_facts_fail_before_archive_access(self):
        for kind in ("prefill", "decode"):
            for field, value, error in (
                ("bytes", 1, "Package file size mismatch"),
                ("sha256", "0" * 64, "Package checksum mismatch"),
            ):
                with self.subTest(kind=kind, field=field):
                    meta = self.model_metadata()
                    meta["entries"][kind][field] = value
                    with patch("compiler.common.weights.read_shared_weight_manifest") as archive:
                        with self.assertRaisesRegex(ValueError, error):
                            validate_shared_weights(self.bundle, meta)
                        archive.assert_not_called()

    def test_model_metadata_rejects_a_different_archive_model(self):
        for model in ("cpu-tied-view-fixture", "qwen3-0.6b"):
            with self.subTest(model=model):
                self.write_archive(model)
                meta = self.model_metadata()
                with self.assertRaisesRegex(ValueError, "Package model identity mismatch"):
                    validate_shared_weights(self.bundle, meta)

    def test_missing_shared_weights_contract_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "metadata is required"):
            validate_shared_weights(self.bundle, {})


if __name__ == "__main__":
    unittest.main()
