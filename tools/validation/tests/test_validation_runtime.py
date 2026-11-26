"""CPU checks for shared validation-runtime loading and constant ownership."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import call, patch, sentinel

import torch

from compiler.common.kernels.flash_attention import attention
from compiler.common.metadata import METADATA_FILE, make_metadata, state_spec
from compiler.common.kernels.gated_delta_net import deltanet
from tools.validation.common import runner
from tools.validation.common.pipeline import _contract_probes


class AOTIRuntimeLoading(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def metadata(self, model_type):
        states = (
            [state_spec(0, kind, [8, 8, 1024, 128], "bfloat16") for kind in ("key", "value")]
            if model_type == "qwen3"
            else [
                state_spec(0, "conv", [8, 6144, 4], "bfloat16"),
                state_spec(0, "recurrent", [8, 16, 128, 128], "float32"),
            ]
        )
        entry = make_metadata(
            model_type=model_type,
            checkpoint=dict(model="fixture", revision="fixture"),
            torch_version="2.13.0",
            cxx11_abi=True,
            cuda_compute_capability=[8, 9],
            cache_length=1024,
            vocab_size=100,
            eos_token_ids=[99],
            state=states,
        )
        entry.update(
            entries=dict(
                prefill=dict(bytes=1, sha256="1" * 64),
                decode=dict(bytes=1, sha256="1" * 64),
            ),
            shared_weights=dict(
                identity="2" * 64,
                bytes=1,
                sha256="3" * 64,
            ),
        )
        (self.root / METADATA_FILE).write_text(json.dumps(entry))
        return entry

    def test_both_model_abis_retain_one_owner_for_both_entries(self):
        for model_type in ("qwen3", "qwen35", "qwen38"):
            with self.subTest(model_type=model_type):
                meta = self.metadata(model_type)
                with (
                    patch.object(runner, "validate_shared_weights") as validate,
                    patch.object(runner, "SharedWeights") as weights,
                ):
                    owner = weights.return_value
                    owner.load.side_effect = [sentinel.prefill, sentinel.decode]
                    runtime = runner.load_aoti_runtime(
                        self.root, model_type=model_type, specs=meta["state"]
                    )
                    validate.assert_called_once_with(self.root, meta)
                    weights.assert_called_once_with(self.root, meta["shared_weights"])
                    self.assertEqual(owner.load.call_args_list, [call("prefill"), call("decode")])
                    self.assertIs(runtime.owner, owner)
                    self.assertEqual(
                        runtime.functions, dict(prefill=sentinel.prefill, decode=sentinel.decode)
                    )
                    self.assertEqual(runtime.specs, meta["state"])
                    self.assertEqual(runtime.vocab_size, 100)
                    self.assertEqual(runtime.capacity, 1024)
                    self.assertTrue(runtime.logits_only)
                    self.assertTrue(runtime.require_write)
                    self.assertEqual(runtime.state, [])
                    runtime.close()
                    self.assertIsNone(runtime.owner)
                    self.assertEqual(runtime.functions, {})

    def test_expected_state_mismatch_fails_before_operator_or_weight_loading(self):
        meta = self.metadata("qwen35")
        with (
            patch.object(runner, "validate_shared_weights") as validate,
            patch.object(runner, "SharedWeights") as weights,
        ):
            with self.assertRaisesRegex(ValueError, "Runtime state descriptors differ"):
                runner.load_aoti_runtime(self.root, model_type="qwen35", specs=[])
            validate.assert_called_once_with(self.root, meta)
            weights.assert_not_called()

    def test_factory_leaves_operator_loading_to_the_compiled_packages(self):
        meta = self.metadata("qwen35")
        attn_override = str(self.root / "unrelated-attention.so")
        delta_override = str(self.root / "unrelated-deltanet.so")
        with (
            patch.object(runner, "validate_shared_weights"),
            patch.object(runner, "SharedWeights") as weights,
            patch.object(attention, "load_attention_ops") as load_attn,
            patch.object(deltanet, "load_deltanet_ops") as load_delta,
            patch.dict(
                os.environ,
                XINFER_ATTENTION_OPS_LIBRARY=attn_override,
                XINFER_DELTANET_OPS_LIBRARY=delta_override,
            ),
        ):
            runtime = runner.load_aoti_runtime(self.root, model_type="qwen35", specs=meta["state"])
            load_attn.assert_not_called()
            load_delta.assert_not_called()
            self.assertEqual(
                weights.return_value.load.call_args_list, [call("prefill"), call("decode")]
            )
            self.assertEqual(os.environ["XINFER_ATTENTION_OPS_LIBRARY"], attn_override)
            self.assertEqual(os.environ["XINFER_DELTANET_OPS_LIBRARY"], delta_override)
            runtime.close()


class BatchAdmission(unittest.TestCase):
    def test_invalid_rows_are_rejected_before_execution(self):
        runtime = runner.ExplicitRuntime(
            {}, [{"shape": [8, 8, 1024, 128]}], 128, 1024, logits_only=True
        )
        for rows in (
            [],
            [[1]] * 9,
            [[], []],
            [[], [1]],
            [[-1], [1]],
            [[1], [128]],
            [[1] * 513, [1]],
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                runtime.step_batch(rows)
            self.assertEqual(runtime.positions, [0] * 8)
            self.assertEqual(runtime.state, [])
        for positions in ([1024, 9] + [0] * 6, [9, 1024] + [0] * 6):
            runtime.positions = positions.copy()
            with self.assertRaisesRegex(ValueError, "capacity"):
                runtime.step_batch([[1], [2]])
            self.assertEqual(runtime.positions, positions)


class ReservedRows(unittest.TestCase):
    class Runtime:
        capacity = 8

        def __init__(self, corrupt_unused=False):
            self.specs = [state_spec(0, "key", [8, 1, 8, 1], "bfloat16")]
            self.state = [torch.empty(self.specs[0]["shape"], dtype=torch.bfloat16)]
            self.corrupt_unused = corrupt_unused

        def reset(self, fill=0):
            self.state[0].fill_(fill.get("key", 0) if isinstance(fill, dict) else fill)
            self.position = 0

        def step(self, tokens):
            end = self.position + len(tokens)
            self.state[0][0, :, self.position : end] = 1
            if self.corrupt_unused:
                self.state[0][1, :, :end] = 1
            self.position = end
            return torch.ones((1, 2), dtype=torch.bfloat16)

    def test_single_request_poison_checks_preserve_reserved_rows(self):
        runtime = self.Runtime()
        _contract_probes(runtime, runtime.specs, [1, 2], [2])
        runner.teacher_trace(
            runtime,
            [
                dict(name="clean", steps=[[1, 2]]),
                dict(name="poison", steps=[[1, 2]], fill=3, same_as="clean"),
            ],
            runtime.specs,
            lambda case, step: True,
        )

    def test_poison_probe_rejects_unused_row_writes(self):
        runtime = self.Runtime(corrupt_unused=True)
        with self.assertRaisesRegex(ValueError, "Unused KV row overwritten"):
            _contract_probes(runtime, runtime.specs, [1, 2], [2])


if __name__ == "__main__":
    unittest.main()
