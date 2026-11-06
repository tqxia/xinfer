"""Validate HF → eager adapters/passes → Python AOTI → C++ AOTI."""

import argparse
import importlib
import os
from pathlib import Path
import traceback

from compiler.common.artifacts import default_build_dir, write_json
from compiler.common.metadata import read_package_metadata
from tools.validation.common import require_assertions


def main():
    require_assertions()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False)
    os.environ["AOTI_RUNTIME_CHECK_INPUTS"] = "1"
    try:
        from compiler.common.attention import load_attention_ops
        from tools.common import native_attention_library
        from compiler.common.artifacts import require, sha256
        from tools.validation.common.chain import new_report
        from tools.validation.common.pipeline import validate_chain

        pair = read_package_metadata(args.packages)
        model_type = pair["decode"]["model_type"]
        native = args.native or default_build_dir(model_type) / "xinfer_validate"
        require(native.is_file(), "Build xinfer_validate with XINFER_BUILD_VALIDATION=ON first")
        library = native_attention_library(native)
        require(library is not None, "Native validation requires the shared attention library")
        os.environ["XINFER_ATTENTION_OPS_LIBRARY"] = str(library)
        load_attention_ops(library)
        module = importlib.import_module(f"tools.validation.{model_type}.model")
        suite = module.Suite(args.model_dir or module.default_model_dir(pair), pair)
        report = new_report(model_type, pair)
        report["attention_library_sha256"] = sha256(library)
        validate_chain(suite, args.packages, native, args.output, pair, report)
    except Exception as error:
        report.update(passed=False, error=dict(type=type(error).__name__, message=str(error)))
        traceback.print_exc()
    write_json(args.output / "summary.json", report)
    print(f"Passed: {report['passed']}; report: {args.output / 'summary.json'}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
