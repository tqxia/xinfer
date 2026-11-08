"""Benchmark HF/native execution and compare compiled package bundles."""

import argparse
from importlib import import_module
from pathlib import Path

from compiler.common.artifacts import default_build_dir, require
from compiler.common.metadata import read_package_metadata
from tools.benchmark.common import (
    eager_worker,
    resolve_gpu,
    run_benchmark,
    validate_comparison,
    wait_idle,
)


def model_hooks(model_type):
    return import_module(f"tools.benchmark.{model_type}")


def run(args):
    paired = args.command == "compare"
    if paired:
        labels = args.baseline_label, args.candidate_label
        require(
            labels[0] != labels[1]
            and all(
                label.replace("_", "").isalnum() and label not in ("hf", "cpp_aoti")
                for label in labels
            ),
            "Provide distinct alphanumeric/underscore labels; hf and cpp_aoti are reserved",
        )
        packages = {
            args.baseline_label: args.baseline_packages,
            args.candidate_label: args.candidate_packages,
        }
    else:
        packages = {"cpp_aoti": args.packages}
    metadata = read_package_metadata(next(iter(packages.values())))
    model_type = metadata["decode"]["model_type"]
    hooks = model_hooks(model_type)
    for directory in packages.values():
        pair = read_package_metadata(directory, model_type=model_type)
        for entry in pair.values():
            hooks.validate_model_metadata(entry)
    args.model_dir = args.model_dir or hooks.default_model_dir(metadata)
    args.native = args.native or default_build_dir(model_type) / "xinfer_benchmark"

    def compare(profiles, weights):
        validate_comparison(
            profiles[args.baseline_label],
            profiles[args.candidate_label],
            weights[args.baseline_label],
            weights[args.candidate_label],
        )

    return run_benchmark(
        args,
        metadata=metadata,
        model_type=model_type,
        packages=packages,
        eager={} if paired else {"hf": ("tools.benchmark", "worker")},
        reference=args.baseline_label if paired else "hf",
        prepare_tokens=lambda: hooks.prepare_tokens(args.model_dir),
        validate_model=hooks.validate_model_metadata,
        compare=compare if paired else None,
        timeout=3600,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tools.benchmark", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gpu_help = (
        "nvidia-smi physical GPU index or UUID; defaults to CUDA device 0 "
        "under CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER"
    )
    idle = commands.add_parser("check-idle", help="Check the GPU without loading a model")
    idle.add_argument("--gpu", help=gpu_help)
    idle.add_argument("--output", type=Path, required=True)
    worker = commands.add_parser("worker", help="Internal HF worker; normally use run")
    worker.add_argument("--model-type", choices=("qwen3", "qwen35"), required=True)
    worker.add_argument("--model-dir", type=Path, required=True)
    worker.add_argument("--workload", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    for name in ("run", "compare"):
        runner = commands.add_parser(name)
        runner.add_argument("--model-dir", type=Path)
        runner.add_argument("--native", type=Path)
        if name == "run":
            runner.add_argument("--packages", type=Path, required=True)
        else:
            runner.add_argument("--baseline-packages", type=Path, required=True)
            runner.add_argument("--candidate-packages", type=Path, required=True)
            runner.add_argument("--baseline-label", default="baseline")
            runner.add_argument("--candidate-label", default="candidate")
        runner.add_argument("--output", type=Path, required=True)
        runner.add_argument("--gpu", help=gpu_help)
        runner.add_argument("--lengths", type=int, nargs="+", default=[25, 128, 512])
        runner.add_argument("--decode-steps", type=int, default=64)
        runner.add_argument("--warmup", type=int, default=2)
        runner.add_argument("--repeats", type=int, default=5)
        runner.add_argument("--rounds", type=int, default=3 if name == "run" else 4)
        runner.add_argument(
            "--allow-busy",
            action="store_true",
            help="Functional harness checks only; timings are provisional",
        )
    args = parser.parse_args(argv)
    if args.command == "check-idle":
        require(not args.output.exists(), "Output already exists")
        gpu = resolve_gpu(args.gpu)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        wait_idle(gpu, output=args.output)
    elif args.command == "worker":
        eager_worker(args, model_hooks(args.model_type).eager_loader(), "hf")
    else:
        run(args)
