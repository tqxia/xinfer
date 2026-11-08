"""CPU checks for shared validation-runtime loading and constant ownership."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import call, patch, sentinel

from compiler.common import attention
from compiler.common.metadata import make_metadata, state_spec
from tools.validation.common import runner


class AOTIRuntimeLoading(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def metadata(self, model_type):
        states = (
            [
                state_spec(0, kind, [1, 8, 1024, 128], "bfloat16", "inplace")
                for kind in ("key", "value")
            ]
            if model_type == "qwen3"
            else [
                state_spec(0, "conv", [1, 6144, 4], "bfloat16", "replace"),
                state_spec(0, "recurrent", [1, 16, 128, 128], "float32", "replace"),
            ]
        )
        pair = {}
        for kind in ("prefill", "decode"):
            entry = make_metadata(
                model_type=model_type,
                entry_point=kind,
                token_range=[2, 512] if kind == "prefill" else [1, 1],
                checkpoint=dict(model_id="fixture", revision="fixture", manifest={}),
                torch_version="2.13.0",
                cxx11_abi=True,
                cuda_compute_capability=[8, 9],
                cache_length=1024,
                vocab_size=100,
                eos_token_ids=[99],
                state=states,
                model_specific={} if model_type == "qwen3" else dict(delta_backend="triton_op"),
            )
            entry.update(
                package_bytes=1,
                package_sha256="1" * 64,
                shared_weights=dict(
                    format=2, file="weights.pt", identity="2" * 64, bytes=1, sha256="3" * 64
                ),
                inductor_configs={
                    "emulate_precision_casts": True,
                    "aot_inductor.package_constants_in_so": False,
                    "always_keep_tensor_constants": True,
                },
            )
            pair[kind] = entry
            (self.root / f"{kind}.json").write_text(json.dumps(entry))
        return pair

    def test_both_model_abis_retain_one_owner_for_both_entries(self):
        for model_type in ("qwen3", "qwen35"):
            with self.subTest(model_type=model_type):
                pair = self.metadata(model_type)
                with (
                    patch.object(runner, "validate_shared_weights") as validate,
                    patch.object(runner, "SharedWeights") as weights,
                ):
                    owner = weights.return_value
                    owner.load.side_effect = [sentinel.prefill, sentinel.decode]
                    runtime = runner.load_aoti_runtime(
                        self.root, model_type=model_type, specs=pair["decode"]["state"]
                    )
                    validate.assert_called_once_with(self.root, pair["prefill"], pair["decode"])
                    weights.assert_called_once_with(self.root, pair["decode"]["shared_weights"])
                    self.assertEqual(owner.load.call_args_list, [call("prefill"), call("decode")])
                    self.assertIs(runtime.owner, owner)
                    self.assertEqual(
                        runtime.functions, dict(prefill=sentinel.prefill, decode=sentinel.decode)
                    )
                    self.assertEqual(runtime.specs, pair["decode"]["state"])
                    self.assertEqual(runtime.vocab_size, 100)
                    self.assertEqual(runtime.capacity, 1024)
                    self.assertEqual(runtime.logits_only, model_type == "qwen3")
                    self.assertEqual(runtime.require_write, model_type == "qwen3")
                    self.assertEqual(runtime.state, [])
                    runtime.close()
                    self.assertIsNone(runtime.owner)
                    self.assertEqual(runtime.functions, {})

    def test_expected_state_mismatch_fails_before_operator_or_weight_loading(self):
        pair = self.metadata("qwen35")
        with (
            patch.object(runner, "validate_shared_weights") as validate,
            patch.object(runner, "SharedWeights") as weights,
        ):
            with self.assertRaisesRegex(ValueError, "Runtime state descriptors differ"):
                runner.load_aoti_runtime(self.root, model_type="qwen35", specs=[])
            validate.assert_called_once_with(self.root, pair["prefill"], pair["decode"])
            weights.assert_not_called()

    def test_factory_leaves_operator_loading_to_the_compiled_packages(self):
        pair = self.metadata("qwen35")
        override = str(self.root / "unrelated-attention.so")
        with (
            patch.object(runner, "validate_shared_weights"),
            patch.object(runner, "SharedWeights") as weights,
            patch.object(attention, "load_attention_ops") as load,
            patch.dict(os.environ, XINFER_ATTENTION_OPS_LIBRARY=override),
        ):
            runtime = runner.load_aoti_runtime(
                self.root, model_type="qwen35", specs=pair["decode"]["state"]
            )
            load.assert_not_called()
            self.assertEqual(
                weights.return_value.load.call_args_list, [call("prefill"), call("decode")]
            )
            self.assertEqual(os.environ["XINFER_ATTENTION_OPS_LIBRARY"], override)
            runtime.close()


if __name__ == "__main__":
    unittest.main()
