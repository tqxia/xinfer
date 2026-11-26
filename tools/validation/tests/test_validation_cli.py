"""The numerical CLI preserves completed checks and reports execution failures."""

from contextlib import redirect_stderr, redirect_stdout
import io
from importlib import import_module
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, sentinel

from compiler.common.tests.test_metadata import bundle_metadata
from tools.tests.test_packages import metadata
from tools.validation import validate
from tools.validation.common import pipeline


class FrontendDispatch(unittest.TestCase):
    def test_package_type_selects_frontend_and_build_without_checkpoint_fallback(self):
        for model_type in ("qwen3", "qwen35", "qwen38"):
            with self.subTest(model_type=model_type), tempfile.TemporaryDirectory() as temporary:
                bundle = bundle_metadata(model_type)
                bundle["checkpoint"]["model"] = "Qwen/Qwen3.8-27B-FP8"
                root = Path(temporary)
                binary = root / "xinfer_validate"
                binary.touch()
                frontend = import_module(f"tools.validation.{model_type}.model")
                with (
                    patch.object(validate, "read_package_metadata", return_value=bundle),
                    patch.object(validate, "default_build_dir", return_value=root) as build,
                    patch("tools.packages.validate_shared_weights"),
                    patch("tools.validation.common.operators.load_packaged_attention_ops"),
                    patch("tools.validation.common.operators.load_packaged_deltanet_ops"),
                    patch.object(frontend, "Suite", return_value=sentinel.suite) as suite,
                ):
                    actual, actual_metadata, native = validate.prepare_validation(
                        Path("bundle"), Path("checkpoint"), None
                    )
                self.assertIs(actual, sentinel.suite)
                self.assertIs(actual_metadata, bundle)
                self.assertEqual(native, binary)
                build.assert_called_once_with()
                suite.assert_called_once_with(Path("checkpoint"), bundle)


class ValidationReports(unittest.TestCase):
    def test_exit_status_and_partial_reports(self):
        for outcome in ("passed", "failed", "exception", "preflight"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "report"
                bundle = metadata()

                def execute(suite, packages, native, destination, metadata, report):
                    report["stages"].append(dict(name="hf"))
                    if outcome == "exception":
                        raise RuntimeError("native execution failed")
                    report["passed"] = outcome == "passed"

                with (
                    patch.object(
                        validate,
                        "prepare_validation",
                        return_value=(sentinel.suite, bundle, Path("native")),
                        side_effect=ValueError("invalid package")
                        if outcome == "preflight"
                        else None,
                    ),
                    patch.object(pipeline, "validate_chain", side_effect=execute) as chain,
                    patch.dict(os.environ),
                    redirect_stderr(io.StringIO()),
                    redirect_stdout(io.StringIO()),
                    self.assertRaises(SystemExit) as exited,
                ):
                    validate.main(["--packages", "packages", "--output", str(output)])
                self.assertEqual(exited.exception.code, 0 if outcome == "passed" else 1)
                report = json.loads((output / "summary.json").read_text())
                self.assertEqual(report["passed"], outcome == "passed")
                if outcome == "preflight":
                    chain.assert_not_called()
                    self.assertEqual(
                        report["error"], dict(type="ValueError", message="invalid package")
                    )
                else:
                    self.assertEqual(report["stages"], [dict(name="hf")])
                    if outcome == "exception":
                        self.assertEqual(
                            report["error"],
                            dict(type="RuntimeError", message="native execution failed"),
                        )

    def test_existing_output_is_preserved_before_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            previous = output / "summary.json"
            previous.write_text("keep existing report")
            with patch.object(validate, "prepare_validation") as prepare:
                with self.assertRaises(FileExistsError):
                    validate.validate_packages(packages=Path("packages"), output=output)
            prepare.assert_not_called()
            self.assertEqual(previous.read_text(), "keep existing report")


if __name__ == "__main__":
    unittest.main()
