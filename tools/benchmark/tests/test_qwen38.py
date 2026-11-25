"""Qwen3.8 benchmark defaults and FP8 worker routing by package model type."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch, sentinel

from compiler.common.artifacts import MODEL_MANIFEST_NAME, default_build_dir
from compiler.common.tests.test_metadata import bundle_metadata
from compiler.qwen38.config import TEXT_DIR
from tools.benchmark import cli, qwen35, qwen38

MODEL_ID = "Qwen/Qwen3.8-27B-FP8"


class Frontend(unittest.TestCase):
    def test_worker_selects_qwen38_original_hf_loader_by_model_type(self):
        with (
            patch("compiler.qwen38.model.load_model") as load,
            patch("compiler.qwen35.config.configure"),
            patch.object(qwen38, "hf_eager_loader", return_value=sentinel.loader) as wrapper,
            patch.object(qwen35, "eager_loader") as unused,
            patch.object(cli, "eager_worker") as worker,
        ):
            cli.main(
                [
                    "worker",
                    "--model-type",
                    "qwen38",
                    "--model-dir",
                    "checkpoint",
                    "--workload",
                    "workload.json",
                    "--output",
                    "result.json",
                ]
            )
        wrapper.assert_called_once_with(load)
        load.assert_not_called()
        unused.assert_not_called()
        self.assertIs(worker.call_args.args[1], sentinel.loader)
        self.assertEqual(worker.call_args.args[0].model_type, "qwen38")

    def test_worker_does_not_reroute_qwen35_from_checkpoint_id(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / MODEL_MANIFEST_NAME).write_text(json.dumps(dict(model=MODEL_ID)))
            with (
                patch.object(qwen35, "eager_loader", return_value=sentinel.loader) as loader,
                patch.object(qwen38, "eager_loader") as unused,
                patch.object(cli, "eager_worker") as worker,
            ):
                cli.main(
                    [
                        "worker",
                        "--model-type",
                        "qwen35",
                        "--model-dir",
                        str(root),
                        "--workload",
                        "workload.json",
                        "--output",
                        str(root / "result.json"),
                    ]
                )
            loader.assert_called_once_with()
            unused.assert_not_called()
            self.assertIs(worker.call_args.args[1], sentinel.loader)

    def test_controller_uses_package_frontend_and_build(self):
        for model_type, hooks in (("qwen35", qwen35), ("qwen38", qwen38)):
            with self.subTest(model_type=model_type):
                metadata = bundle_metadata(model_type)
                metadata["checkpoint"]["model"] = MODEL_ID
                argv = ["run", "--packages", "bundle", "--output", "output"]
                if model_type == "qwen35":
                    argv += ["--model-dir", "checkpoint"]
                args = cli.build_parser().parse_args(argv)
                with (
                    patch.object(cli, "read_package_metadata", return_value=metadata),
                    patch.object(cli, "run_benchmark", return_value=sentinel.report) as run,
                    patch.object(hooks, "prepare_tokens", return_value=sentinel.tokens) as tokens,
                ):
                    self.assertIs(cli.run(args), sentinel.report)
                    self.assertIs(run.call_args.kwargs["prepare_tokens"](), sentinel.tokens)
                self.assertIs(cli.model_hooks(model_type), hooks)
                directory = TEXT_DIR if model_type == "qwen38" else Path("checkpoint")
                tokens.assert_called_once_with(directory)
                self.assertEqual(args.model_dir, directory)
                self.assertEqual(args.native, default_build_dir(model_type) / "xinfer_benchmark")
                self.assertEqual(run.call_args.kwargs["model_type"], model_type)
