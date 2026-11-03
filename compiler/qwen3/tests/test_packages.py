"""Reject incompatible Qwen3 tensor ABIs before native-library or CUDA work."""

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from compiler.common.metadata import make_metadata, state_spec
from compiler.qwen3.packages import load_package_attention_ops


def metadata():
    shape = [1, 8, 1024, 128]
    state = [state_spec(0, kind, shape, "bfloat16", "inplace") for kind in ("key", "value")]
    result = make_metadata(
        model_type="qwen3", entry_point="decode", token_range=[1, 1],
        checkpoint=dict(model_id="Qwen/Qwen3-0.6B", revision="pinned", manifest={}),
        torch_version="2.13.0+cu126", torch_cuda_version="12.6", cxx11_abi=True,
        cuda_compute_capability=[8, 9], cache_length=1024, vocab_size=151936,
        eos_token_ids=[151645], state=state,
        model_specific=dict(layers=1, kv_heads=8, head_dim=128, cache_shape=shape,
                            cache_stride=state[0]["stride"], cache_padding=0))
    result.update(package_bytes=128, package_sha256="a" * 64, inductor_configs={},
                  shared_weights=dict(format=2, file="weights.pt", identity="b" * 64,
                                      bytes=128, sha256="c" * 64))
    return result


class Qwen3PackageContract(unittest.TestCase):
    def reject(self, value, message):
        with patch("compiler.qwen3.packages.load_attention_ops") as load, \
                patch("compiler.qwen3.packages.native_attention_library") as resolve:
            with self.assertRaisesRegex(ValueError, "^" + message):
                load_package_attention_ops(value, native=Path("xinfer_aoti"))
            resolve.assert_not_called()
            load.assert_not_called()

    def test_valid_tensor_abi_loads_the_native_dependency(self):
        native, library = Path("xinfer_aoti"), Path("libxinfer_attention_ops.so")
        with patch("compiler.qwen3.packages.load_attention_ops") as load, \
                patch("compiler.qwen3.packages.native_attention_library", return_value=library) as resolve:
            load_package_attention_ops(metadata(), native=native)
            resolve.assert_called_once_with(native)
            load.assert_called_once_with(library)

    def test_geometry_validation_precedes_loading(self):
        cases = (("layers", 0, "Invalid model dimensions"),
                 ("kv_heads", 0, "Invalid model dimensions"),
                 ("kv_heads", 4, "Invalid KV cache shape"),
                 ("cache_padding", 1, "Unsupported cache padding"),
                 ("cache_shape", [1, 8, 1024, 64], "Unsupported Flash cache shape"),
                 ("cache_stride", [1048576, 131072, 129, 1], "Unexpected KV cache strides"))
        for field, value, message in cases:
            with self.subTest(field=field, value=value):
                item = metadata()
                item["model_specific"][field] = value
                self.reject(item, message)

    def test_missing_and_boolean_geometry_fields_are_rejected(self):
        for field in ("layers", "kv_heads", "head_dim", "cache_padding", "cache_shape", "cache_stride"):
            for invalid in (None, True):
                with self.subTest(field=field, invalid=invalid):
                    item = metadata()
                    if invalid is None:
                        item["model_specific"].pop(field)
                    else:
                        item["model_specific"][field] = invalid
                    self.reject(item, "Unsupported package metadata: model_specific")

    def test_qwen3_tag_does_not_accept_functional_deltanet_state(self):
        item = metadata()
        item["state"] = [state_spec(0, "conv", [1, 6144, 4], "bfloat16", "replace"),
                         state_spec(0, "recurrent", [1, 16, 128, 128], "float32", "replace")]
        item["input_order"] = ["input_ids", "cache_position", *[s["name"] for s in item["state"]], "kv_extent"]
        self.reject(item, "Unsupported package metadata: state")

    def test_state_order_and_output_abi_are_model_specific(self):
        item = metadata()
        item["state"] = list(reversed(copy.deepcopy(item["state"])))
        item["input_order"] = ["input_ids", "cache_position", *[s["name"] for s in item["state"]], "kv_extent"]
        self.reject(item, "Unsupported package metadata: state")
        item = metadata()
        item["output_order"] += [s["name"] for s in item["state"]]
        self.reject(item, "Unsupported package metadata: output_order")


if __name__ == "__main__":
    unittest.main()
