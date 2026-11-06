"""Validate MegaQwen with the shared HF → eager → AOTI validation chain.

This entry point adds hybrid handoff fixtures and a native state-contract probe.
The common pipeline owns model execution, live BF16 HF references, compact state
capture, AOTI parity and matched-history numerical comparisons. No timing here
is a decode performance measurement.
"""

import argparse
import os
from pathlib import Path
import subprocess
import traceback

from compiler.common.artifacts import default_build_dir, read_json, require, sha256, write_json
from compiler.common.metadata import read_package_metadata
from tools.validation.common import require_assertions


def extra_cases(cases, seed):
    """Exercise native handoffs, split-KV transitions and poisoned future state."""
    require(bool(seed), "Hybrid validation needs a nonempty prompt seed")
    names = {case["name"] for case in cases}

    def prompt(length):
        return (seed * (length // len(seed) + 1))[:length]

    additions = [
        dict(name="hybrid_prompt_25", steps=[prompt(25), [0], [151645], [151935]]),
        dict(
            name="hybrid_back_to_aoti",
            steps=[prompt(2), [13], prompt(13), [0], prompt(2), [151643]],
        ),
    ]
    clean = additions[1]
    additions.append(
        dict(
            name=clean["name"] + "_poison",
            steps=clean["steps"],
            same_as=clean["name"],
            poison_kv=3,
        )
    )
    for boundary in (128, 256):
        clean = dict(
            name=f"hybrid_split_{boundary}",
            steps=[prompt(boundary - 1), [13], [0], prompt(2), [151643]],
        )
        additions.extend(
            (
                clean,
                dict(
                    name=clean["name"] + "_poison",
                    steps=clean["steps"],
                    same_as=clean["name"],
                    poison_kv=3,
                ),
            )
        )
    for order, case in enumerate(additions, start=100):
        require(case["name"] not in names, "Duplicate hybrid fixture")
        case.update(order=order, dump_cache=True)
        names.add(case["name"])
    return [*cases, *additions]


def _suite(model_dir, metadata):
    # Import only in the selected Qwen3 environment; fixture-only CPU tests do
    # not need to load HF or its version-specific model implementation.
    from tools.validation.qwen3.model import Suite

    class HybridSuite(Suite):
        """Inherit live HF references and every metadata-driven context case."""

        def fixtures(self, runtime):
            cases, chats = super().fixtures(runtime)
            seed = next(case["steps"][0] for case in cases if case.get("fixture") == "capital")
            return extra_cases(cases, seed), chats

    return HybridSuite(model_dir, metadata)


def validate(args, report):
    from compiler.common.attention import load_attention_ops
    from tools.common import native_attention_library
    from tools.validation.common.chain import new_report
    from tools.validation.common.native import native_environment
    from tools.validation.common.pipeline import validate_chain
    from tools.validation.qwen3.model import default_model_dir

    metadata = read_package_metadata(args.packages, model_type="qwen3")
    native = (args.native or default_build_dir("qwen3") / "xinfer_validate").resolve()
    contract = (args.contract_native or native.parent / "qwen_hybrid_contract").resolve()
    require(native.is_file() and contract.is_file(), "Build native validation drivers first")
    library = native_attention_library(native)
    require(library is not None, "Native validation requires the shared attention library")
    os.environ["XINFER_ATTENTION_OPS_LIBRARY"] = str(library)
    load_attention_ops(library)
    suite = _suite(args.model_dir or default_model_dir(metadata), metadata)
    report.update(new_report("qwen3", metadata))
    report.update(
        benchmark=False,
        status="running",
        attention_library_sha256=sha256(library),
        contract_native_sha256=sha256(contract),
    )
    validate_chain(
        suite,
        args.packages,
        native,
        args.output,
        metadata,
        report,
        native_backends=("megaqwen",),
    )
    print("Control: native hybrid state and ownership contracts", flush=True)
    with (args.output / "contracts.log").open("w") as log:
        result = subprocess.run(
            [
                str(contract),
                str(args.packages.resolve()),
                str((args.output / "contracts.json").resolve()),
            ],
            env=native_environment(),
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
        )
    require(result.returncode == 0, "Native hybrid contracts failed; see contracts.log")
    contracts = read_json(args.output / "contracts.json")
    require(
        contracts.get("passed") and contracts.get("no_python_library"), "Invalid hybrid contracts"
    )
    report["hybrid_contracts"] = contracts
    report["passed"] = report["passed"] and contracts["passed"]
    report["status"] = "passed" if report["passed"] else "failed"


def main():
    require_assertions()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--contract-native", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--timeout", type=int, default=3600, help="Native contract subprocess timeout"
    )
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["AOTI_RUNTIME_CHECK_INPUTS"] = "1"
    report = dict(passed=False, status="running", benchmark=False)
    exit_code = 2
    try:
        validate(args, report)
        exit_code = 0 if report["passed"] else 1
    except Exception as error:
        report.update(
            passed=False,
            status="incomplete",
            error=dict(type=type(error).__name__, message=str(error)),
        )
        traceback.print_exc()
    write_json(args.output / "summary.json", report)
    print(f"Passed: {report['passed']}; report: {args.output / 'summary.json'}", flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
