"""CPU contracts for the combined eager validation stage."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch, sentinel

import torch

from compiler.qwen35 import export
from compiler.qwen35.adapters import delta
from tools.validation.qwen35 import model


class ValidationStages(unittest.TestCase):
    def setUp(self):
        self.suite = model.Suite.__new__(model.Suite)
        self.suite.model_dir = Path("unused-checkpoint")
        self.suite.specs = []
        self.suite.capacity = 1024
        self.suite.max_prefill = 512
        self.suite.vocab_size = 16
        self.suite.pair = {"prefill": {"model_specific": {}}}
        self.original = SimpleNamespace(config=object())
        self.events = []
        self.loader = self.enterContext(
            patch.object(model, "load_model", return_value=self.original)
        )
        self.enterContext(patch.object(model, "state_specs", return_value=self.suite.specs))
        self.hf = self.enterContext(
            patch.object(model, "HybridHFStage", side_effect=[sentinel.hf, sentinel.hf_dynamic])
        )
        self.install = self.enterContext(
            patch.object(
                delta, "install_delta", side_effect=lambda _: self.events.append("adapter")
            )
        )
        self.wrapper = self.enterContext(patch.object(model, "TextStep"))
        self.export_step = self.enterContext(patch.object(export, "export_step"))
        self.runtime = self.enterContext(
            patch.object(model, "ExplicitRuntime", return_value=sentinel.eager)
        )
        self.enterContext(patch.object(model.torch.cuda, "empty_cache"))

    def begin(self):
        stages = self.suite.stages()
        self.addCleanup(stages.close)
        for name, runtime in (("hf", sentinel.hf), ("hf_dynamic", sentinel.hf_dynamic)):
            stage = next(stages)
            self.assertEqual(stage.name, name)
            self.assertIs(stage.runtime, runtime)
            self.install.assert_not_called()
            self.wrapper.assert_not_called()
            self.export_step.assert_not_called()
        return stages

    def check_eager(self, stages, functions):
        stage = next(stages)
        self.assertEqual(stage.name, "eager")
        self.assertIs(stage.runtime, sentinel.eager)
        self.runtime.assert_called_once_with(
            functions,
            self.suite.specs,
            self.suite.vocab_size,
            self.suite.capacity,
            owner=self.original,
            max_prefill=512,
        )
        self.install.assert_called_once_with(self.original)
        with self.assertRaises(StopIteration):
            next(stages)
        self.loader.assert_called_once_with(self.suite.model_dir)

    def test_packed_packages_apply_adapters_before_exporting_both_eager_entries(self):
        self.suite.pair["prefill"]["model_specific"]["linear_packing"] = {"enabled": True}
        functions = {"prefill": Mock(), "decode": Mock()}
        exported = {
            kind: SimpleNamespace(module=Mock(return_value=function))
            for kind, function in functions.items()
        }

        def export_entry(original, kind, capacity):
            self.assertEqual(capacity, self.suite.capacity)
            self.assertIs(original, self.original)
            self.assertEqual(self.events[0], "adapter")
            self.events.append(kind)
            return exported[kind], {}

        self.export_step.side_effect = export_entry
        self.check_eager(self.begin(), functions)
        self.assertEqual(self.events, ["adapter", "prefill", "decode"])
        self.export_step.assert_has_calls(
            [
                call(self.original, "prefill", self.suite.capacity),
                call(self.original, "decode", self.suite.capacity),
            ]
        )
        self.assertEqual(self.export_step.call_count, 2)
        for entry in exported.values():
            entry.module.assert_called_once_with()
        self.wrapper.assert_not_called()

    def test_unpacked_packages_use_one_adapted_wrapper_without_exporting(self):
        self.suite.pair["prefill"]["model_specific"]["linear_packing"] = {"enabled": False}

        def wrap(original):
            self.assertIs(original, self.original)
            self.assertEqual(self.events, ["adapter"])
            self.events.append("wrapper")
            return sentinel.wrapper

        self.wrapper.side_effect = wrap
        self.check_eager(self.begin(), {"prefill": sentinel.wrapper, "decode": sentinel.wrapper})
        self.assertEqual(self.events, ["adapter", "wrapper"])
        self.wrapper.assert_called_once_with(self.original)
        self.export_step.assert_not_called()

    def test_greedy_replay_reference_uses_default_bf16_model(self):
        runtime = Mock(spec=["close"])
        self.hf.side_effect = None
        self.hf.return_value = runtime
        with self.suite.reference() as actual:
            self.assertIs(actual, runtime)
        self.loader.assert_called_once_with(self.suite.model_dir)
        self.install.assert_not_called()
        runtime.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
