"""CPU tests for the Torch 2.13 strict export global-cleanup patch."""

import gc
import unittest
from unittest.mock import patch
import weakref

import torch
from torch._dynamo import convert_frame
from torch._dynamo.utils import CleanupManager

from compiler.patches.torch_2_13 import export_cleanup
from compiler.patches.torch_2_13.export_cleanup import cleanup_export_globals


class LinearModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(64, dtype=torch.float32).reshape(8, 8) / 8)

    def forward(self, x):
        return x @ self.weight


class ExportCleanupPatch(unittest.TestCase):
    def test_strict_export_releases_model_weights_and_dynamo_cleanups(self):
        original = convert_frame.DynamoOutput.graph_capture_output
        before_count = CleanupManager.count
        model = LinearModule().eval()
        weight_ref = weakref.ref(model.weight)
        sample = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 4

        with cleanup_export_globals():
            exported = torch.export.export(model, (sample,), strict=True)

        self.assertIs(convert_frame.DynamoOutput.graph_capture_output, original)
        gc.collect()
        self.assertEqual(CleanupManager.count, before_count)
        with torch.inference_mode():
            torch.testing.assert_close(exported.module()(sample), model(sample), rtol=0, atol=0)

        del exported, model
        gc.collect()
        self.assertIsNone(weight_ref(), "Strict export leaked parameter reference")
        self.assertEqual(CleanupManager.count, before_count)

    def test_restores_hook_on_exception(self):
        original = convert_frame.DynamoOutput.graph_capture_output
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with cleanup_export_globals():
                self.assertIsNot(convert_frame.DynamoOutput.graph_capture_output, original)
                raise RuntimeError("boom")
        self.assertIs(convert_frame.DynamoOutput.graph_capture_output, original)

    def test_rejects_unsupported_torch_version_source_or_nested_hook(self):
        original = convert_frame.DynamoOutput.graph_capture_output
        with (
            patch.object(torch, "__version__", "2.14.0"),
            self.assertRaisesRegex(ValueError, "pinned Torch 2.13"),
        ):
            with cleanup_export_globals():
                pass
        with (
            patch.object(torch.version, "git_version", "0" * 40),
            self.assertRaisesRegex(ValueError, "pinned Torch 2.13"),
        ):
            with cleanup_export_globals():
                pass
        with (
            patch.object(export_cleanup, "CONVERT_FRAME_SHA256", "0" * 64),
            self.assertRaisesRegex(ValueError, "convert_frame source"),
        ):
            with cleanup_export_globals():
                pass
        with cleanup_export_globals():
            with self.assertRaisesRegex(ValueError, "graph capture hook"):
                with cleanup_export_globals():
                    pass
        self.assertIs(convert_frame.DynamoOutput.graph_capture_output, original)


if __name__ == "__main__":
    unittest.main()
