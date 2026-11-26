"""Validate HF → eager adapters/passes → Python AOTI → C++ AOTI."""

import argparse
import importlib
import os
from pathlib import Path
import traceback

from compiler.common.artifacts import default_build_dir, write_json
from compiler.common.metadata import read_package_metadata
from tools.packages import validate_model_metadata
from tools.validation.common import require_assertions


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def prepare_validation(packages, model_dir, native):
    from tools.validation.common.attention import load_packaged_attention_ops
    from tools.packages import validate_shared_weights
    from compiler.common.artifacts import require

    metadata = read_package_metadata(packages)
    model_type = metadata["model_type"]
    native = native or default_build_dir() / "xinfer_validate"
    require(native.is_file(), "Build xinfer_validate with XINFER_BUILD_VALIDATION=ON first")
    validate_model_metadata(metadata)
    validate_shared_weights(packages, metadata)
    load_packaged_attention_ops(packages)
    module = importlib.import_module(f"tools.validation.{model_type}.model")
    suite = module.Suite(model_dir or module.default_model_dir(metadata), metadata)
    return suite, metadata, native


def validate_packages(*, packages, output, model_dir=None, native=None):
    """Run the numerical chain and persist its report, including partial failures."""
    output.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False)
    os.environ["AOTI_RUNTIME_CHECK_INPUTS"] = "1"
    try:
        from tools.validation.common.chain import new_report
        from tools.validation.common.pipeline import validate_chain

        suite, metadata, native = prepare_validation(packages, model_dir, native)
        report = new_report(metadata["model_type"], metadata)
        validate_chain(suite, packages, native, output, metadata, report)
    except Exception as error:
        report.update(passed=False, error=dict(type=type(error).__name__, message=str(error)))
        traceback.print_exc()
    write_json(output / "summary.json", report)
    return report


def main(argv=None):
    require_assertions()
    args = build_parser().parse_args(argv)
    report = validate_packages(
        packages=args.packages, output=args.output, model_dir=args.model_dir, native=args.native
    )
    print(f"Passed: {report['passed']}; report: {args.output / 'summary.json'}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
