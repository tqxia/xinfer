"""Qwen3.8 frontend hooks, FP8 references and shared-ABI validation routing."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch, sentinel

import torch

from compiler.common.artifacts import MODEL_MANIFEST_NAME, default_build_dir
from compiler.common.tests.test_metadata import bundle_metadata
from compiler.qwen38.config import ARTIFACT_DIR, TEXT_DIR
from tools.validation import validate
from tools.validation.common.prompts import chat_prompts
from tools.validation.qwen35 import model as shared, validate_batch as shared_batch
from tools.validation.qwen35 import validate_runtime as shared_runtime
from tools.validation.qwen38 import model, validate_batch, validate_runtime
from tools.validation.tests import test_qwen35

MODEL_ID = "Qwen/Qwen3.8-27B-FP8"


class ValidationStages(test_qwen35.ValidationStages):
    model_id = MODEL_ID
    frontend = test_qwen35.fp8_export


class Frontend(unittest.TestCase):
    def test_chat_prompts_returns_independent_shared_fixtures(self):
        expected = chat_prompts()
        actual = chat_prompts()
        self.assertEqual(actual, expected)
        self.assertEqual(
            [prompt["name"] for prompt in actual], ["capital", "arithmetic", "chinese"]
        )
        actual[0]["text"] = "changed"
        self.assertEqual(chat_prompts(), expected)

    def test_suite_uses_fp8_loader_exporter_and_reference_protocol_with_shared_abi(self):
        meta = bundle_metadata("qwen38")
        manifest = dict(model=MODEL_ID, revision="fixture")
        meta["torch_version"] = torch.__version__
        meta["checkpoint"] = manifest
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / MODEL_MANIFEST_NAME).write_text(json.dumps(manifest))
            with (
                patch.object(shared, "configure"),
                patch.object(shared.AutoTokenizer, "from_pretrained"),
            ):
                suite = model.Suite(root, meta)
        self.assertEqual(suite.model_type, "qwen38")
        self.assertIs(suite.model_loader, model.load_model)
        self.assertIs(suite.step_exporter, model.export_step)
        self.assertEqual(suite.reference_protocol["weight_dtype"], "float8_e4m3fn")
        self.assertEqual(suite.reference_protocol["kernel_version"], 4)
        self.assertEqual(
            suite.reference_protocol["kernel_revision"],
            "03ae31933313f38fae2b050dbe2731024386c369",
        )
        self.assertEqual(model.default_model_dir(meta), TEXT_DIR)

    def test_numerical_entry_point_selects_qwen38_from_package_type(self):
        meta = bundle_metadata("qwen38")
        meta["checkpoint"]["model"] = MODEL_ID
        with TemporaryDirectory() as temporary:
            build = Path(temporary)
            binary = build / "xinfer_validate"
            binary.touch()
            with (
                patch.object(validate, "read_package_metadata", return_value=meta),
                patch.object(validate, "default_build_dir", return_value=build) as build_dir,
                patch("tools.packages.validate_shared_weights"),
                patch("tools.validation.common.operators.load_packaged_attention_ops"),
                patch("tools.validation.common.operators.load_packaged_deltanet_ops"),
                patch.object(model, "Suite", return_value=sentinel.suite) as suite,
                patch.object(shared, "Suite") as unused,
            ):
                actual, metadata, native = validate.prepare_validation(Path("bundle"), None, None)
            self.assertIs(actual, sentinel.suite)
            self.assertIs(metadata, meta)
            self.assertEqual(native, binary)
            build_dir.assert_called_once_with()
            suite.assert_called_once_with(TEXT_DIR, meta)
            unused.assert_not_called()

    def test_suite_rejects_a_different_package_type_before_model_loading(self):
        for package_type in ("qwen3", "qwen35"):
            with (
                self.subTest(package_type=package_type),
                patch.object(shared, "configure") as configure,
                self.assertRaisesRegex(ValueError, "Unsupported model type"),
            ):
                model.Suite(Path("unused"), bundle_metadata(package_type))
            configure.assert_not_called()
        with self.assertRaisesRegex(ValueError, "Unsupported model type"):
            shared.Suite(Path("unused"), bundle_metadata("qwen38"))

    def test_runtime_entry_point_reuses_pipeline_with_qwen38_defaults_and_overrides(self):
        for argv in (
            [],
            [
                "--model-dir",
                "custom/model",
                "--packages",
                "custom/bundle",
                "--binary",
                "custom/validate",
                "--cli",
                "custom/aoti",
                "--output",
                "custom/out",
            ],
        ):
            with self.subTest(argv=argv), patch.object(shared_runtime, "validate_runtime") as run:
                validate_runtime.main(argv)
                args = run.call_args.args[0]
                self.assertEqual(run.call_args.kwargs, dict(expected_model_type="qwen38"))
                self.assertEqual(args.model_dir, Path("custom/model") if argv else TEXT_DIR)
                self.assertEqual(
                    args.packages, Path("custom/bundle") if argv else ARTIFACT_DIR / "text"
                )
                self.assertEqual(
                    args.binary,
                    Path("custom/validate") if argv else default_build_dir() / "xinfer_validate",
                )
                self.assertEqual(
                    args.cli,
                    Path("custom/aoti") if argv else default_build_dir() / "xinfer_aoti",
                )
                self.assertEqual(
                    args.output, Path("custom/out") if argv else ARTIFACT_DIR / "runtime-validation"
                )

    def test_batch_entry_point_uses_shared_pipeline_and_package_build(self):
        meta = bundle_metadata("qwen38")
        meta["checkpoint"] = dict(model=MODEL_ID, revision="fixture")
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / MODEL_MANIFEST_NAME).write_text(json.dumps(meta["checkpoint"]))
            for explicit in (False, True):
                output = root / ("explicit" if explicit else "default")
                argv = ["--packages", "bundle", "--model-dir", str(root), "--output", str(output)]
                if explicit:
                    argv += ["--native", "custom/native"]
                with (
                    patch.object(shared_batch, "read_package_metadata", return_value=meta),
                    patch.object(shared_batch, "validate_model_metadata") as check_meta,
                    patch.object(shared_batch, "validate_shared_weights") as check_weights,
                    patch.object(shared_batch, "validate_python", return_value=["fixture"]),
                    patch.object(shared_batch, "validate_native", return_value={}) as native,
                    patch.object(shared_batch.torch.cuda, "empty_cache"),
                    self.assertRaises(SystemExit) as status,
                ):
                    validate_batch.main(argv)
                self.assertEqual(status.exception.code, False)
                check_meta.assert_called_once_with(meta)
                check_weights.assert_called_once_with(Path("bundle"), meta)
                args, groups = native.call_args.args
                self.assertEqual(groups, ["fixture"])
                self.assertEqual(
                    args.native,
                    Path("custom/native") if explicit else default_build_dir() / "xinfer_validate",
                )

    def test_qwen38_entry_points_reject_qwen35_packages_and_forward_model_prompts(self):
        qwen35_meta = bundle_metadata("qwen35")
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "batch-out"
            argv = ["--packages", "bundle", "--model-dir", str(root), "--output", str(output)]
            with (
                patch.object(shared_batch, "read_package_metadata", return_value=qwen35_meta),
                patch.object(shared_batch, "validate_python") as run_python,
                self.assertRaisesRegex(ValueError, "Unsupported model type"),
            ):
                validate_batch.main(argv)
            run_python.assert_not_called()
            self.assertFalse(output.exists())

            rt_args = type(
                "Args", (), dict(packages=Path("bundle"), model_dir=root, output=root / "rt")
            )()
            with (
                patch.object(shared_runtime, "read_package_metadata", return_value=qwen35_meta),
                self.assertRaisesRegex(ValueError, "Unsupported model type"),
            ):
                shared_runtime.preflight(rt_args, expected_model_type="qwen38")

        qwen38_meta = bundle_metadata("qwen38")
        tokenizer = type("Tok", (), {})()
        with (
            patch.object(
                shared_runtime,
                "fixtures",
                return_value=[dict(name="a", steps=[[1, 2]]), dict(name="b", steps=[[3, 4]])],
            ),
            patch.object(shared_runtime, "long_context_cases", return_value=[]),
            patch.object(shared_runtime, "encode_prompt", return_value=[1, 2]),
            patch.object(
                shared_runtime,
                "chat_prompts",
                return_value=[dict(name="capital", text="hi")],
            ) as prompts,
        ):
            shared_runtime.runtime_cases(tokenizer, qwen38_meta)
        prompts.assert_called_once_with()
