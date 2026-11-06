"""Reject incompatible Qwen3 tensor ABIs without native-library or CUDA work."""

import unittest

from compiler.common.metadata import make_metadata, state_spec, validate_metadata
from compiler.qwen3.packages import validate_model_metadata


def metadata():
    shape = [1, 8, 1024, 128]
    state = [state_spec(0, kind, shape, "bfloat16", "inplace") for kind in ("key", "value")]
    result = make_metadata(
        model_type="qwen3",
        entry_point="decode",
        token_range=[1, 1],
        checkpoint=dict(model_id="Qwen/Qwen3-0.6B", revision="pinned", manifest={}),
        torch_version="2.13.0+cu126",
        cxx11_abi=True,
        cuda_compute_capability=[8, 9],
        cache_length=1024,
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

    def test_invalid_geometry_is_rejected(self):
        for shape in ([1, 4, 1024, 128], [1, 8, 1024, 64]):
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
            state_spec(0, "conv", [1, 6144, 4], "bfloat16", "replace"),
            state_spec(0, "recurrent", [1, 16, 128, 128], "float32", "replace"),
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


if __name__ == "__main__":
    unittest.main()
