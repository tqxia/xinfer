"""Reject partial/wrong Qwen3.6 FP8 packages before archive or device access."""

from copy import deepcopy
import unittest

from compiler.common.metadata import make_metadata, state_spec
from compiler.qwen36.config import MODEL_ID, REVISION
from compiler.qwen36.packages import validate_model_metadata


def metadata(capacity=1024, kind="decode"):
    states = []
    for layer in range(40):
        if layer % 4 == 3:
            states += [
                state_spec(layer, name, [1, 2, capacity, 256], "bfloat16", "inplace")
                for name in ("key", "value")
            ]
        else:
            states += [
                state_spec(layer, "conv", [1, 8192, 4], "bfloat16", "replace"),
                state_spec(layer, "recurrent", [1, 32, 128, 128], "float32", "replace"),
            ]
    return make_metadata(
        model_type="qwen36",
        entry_point=kind,
        token_range=[2, 512] if kind == "prefill" else [1, 1],
        checkpoint=dict(model_id=MODEL_ID, revision=REVISION, manifest={}),
        torch_version="2.13.0+cu130",
        cxx11_abi=True,
        cuda_compute_capability=[12, 1],
        cache_length=capacity,
        vocab_size=248320,
        eos_token_ids=[248046, 248044],
        state=states,
        model_specific=dict(
            delta_backend="triton_op",
            attention_backend="flash_attention",
            fp8_backend="triton_op",
            fp8_format="e4m3fn",
            fp8_block_size=[128, 128],
            fp8_scale_dtype="bfloat16",
            fp8_kernel=dict(
                repo="kernels-community/finegrained-fp8",
                revision="03ae31933313f38fae2b050dbe2731024386c369",
            ),
            linear_packing=dict(enabled=False),
        ),
    )


class Qwen36Packages(unittest.TestCase):
    def test_complete_model_accepts_fixture_and_config_derived_capacities(self):
        for capacity in (512, 1024, 16384, 262144):
            for kind in ("prefill", "decode"):
                with self.subTest(capacity=capacity, kind=kind):
                    validate_model_metadata(metadata(capacity, kind))

    def test_rejects_wrong_quantized_backend_or_bf16_packing(self):
        for name, value in dict(
            attention_backend="sdpa",
            delta_backend="native",
            fp8_backend="hf",
            fp8_format="e5m2",
            fp8_block_size=[64, 128],
            fp8_scale_dtype="float32",
            fp8_kernel=dict(repo="other", revision="pinned"),
            linear_packing=dict(enabled=True),
        ).items():
            item = metadata()
            item["model_specific"][name] = value
            with self.subTest(field=name), self.assertRaises(ValueError):
                validate_model_metadata(item)

    def test_rejects_prefix_and_wrong_hybrid_state_geometry(self):
        item = metadata()
        item["state"] = item["state"][:8]
        with self.assertRaisesRegex(ValueError, "all 40"):
            validate_model_metadata(item)
        for index, shape in (
            (0, [1, 6144, 4]),
            (1, [1, 16, 128, 128]),
            (6, [1, 4, 1024, 256]),
            (7, [1, 2, 1024, 128]),
        ):
            item = metadata()
            item["state"][index]["shape"] = shape
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_model_metadata(item)
        item = metadata()
        item["state"][0]["layer"] = 1
        with self.assertRaisesRegex(ValueError, "state order"):
            validate_model_metadata(item)

    def test_rejects_wrong_checkpoint_eos_vocabulary_and_call_bounds(self):
        template = metadata()
        for field, value in (
            ("model_type", "qwen35"),
            ("vocab_size", 151936),
            ("eos_token_ids", [248044]),
            ("token_range", [2, 512]),
            ("cache_length", 262145),
        ):
            item = deepcopy(template)
            item[field] = value
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "checkpoint or execution bounds"),
            ):
                validate_model_metadata(item)
        for field in ("model_id", "revision"):
            item = deepcopy(template)
            item["checkpoint"][field] = "unrelated"
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "checkpoint or execution bounds"),
            ):
                validate_model_metadata(item)


if __name__ == "__main__":
    unittest.main()
