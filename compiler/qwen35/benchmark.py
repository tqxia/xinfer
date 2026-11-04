"""Fixed-token HF/native benchmark. Busy-GPU runs validate the harness only."""

import argparse
from pathlib import Path
import shutil

from compiler.common.benchmark import (
    eager_worker as measure_eager,
    run_benchmark,
    resolve_gpu,
    wait_idle,
)
from compiler.common.artifacts import default_build_dir, read_json, require, sha256
from compiler.qwen35.config import ARTIFACT_DIR, TEXT_DIR
from compiler.qwen35.packages import validate_model_metadata


def eager_loader():
    from compiler.qwen35.reference import HFRuntime
    from compiler.qwen35.model import configure_text, load_model

    configure_text()

    def load(model_dir):
        runtime = HFRuntime(load_model(model_dir))
        runtime.reset()

        def predict(tokens, position):
            result = runtime.step(tokens).argmax(-1).item()
            require(runtime.position == position + len(tokens), "Consumed position differs")
            return result

        return runtime.reset, predict

    return load


def worker(args):
    measure_eager(args, eager_loader(), "hf")


def benchmark(args):
    def tokens():
        from transformers import AutoTokenizer
        from compiler.qwen35.prepare import encode_prompt

        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        return (
            encode_prompt(tokenizer, "What is the capital of France? Answer briefly."),
            tokenizer.encode("Paris is the capital of France. ", add_special_tokens=False),
        )

    def correctness(environment, out):
        if not args.validation.exists():
            return dict(passed=None, reason="No matching end-to-end report supplied")
        validation = read_json(args.validation)
        metadata = {
            kind: entry["metadata"] for kind, entry in environment["packages"]["cpp_aoti"].items()
        }
        require(
            validation["package_sha256"]
            == {kind: meta["package_sha256"] for kind, meta in metadata.items()},
            "Validation is for different packages",
        )
        require(
            validation.get("shared_weights") == metadata["decode"]["shared_weights"],
            "Validation is for a different weight archive",
        )
        shutil.copy2(args.validation, out / "correctness.json")
        return dict(
            passed=validation["passed"],
            native_bitwise=validation["native_bitwise"],
            report_sha256=sha256(args.validation),
            numerical_failures=len(validation["failures"]),
            greedy_failures=len(validation["greedy_failures"]),
        )

    return run_benchmark(
        args,
        model_type="qwen35",
        packages={"cpp_aoti": args.packages},
        eager={"hf": ("compiler.qwen35.benchmark", "worker")},
        reference="hf",
        prepare_tokens=tokens,
        manifest_name="manifest.json",
        validate_model=validate_model_metadata,
        correctness=correctness,
        timeout=3600,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker_parser = commands.add_parser("worker", help="Internal eager worker; normally use run")
    worker_parser.add_argument("--model-dir", type=Path, default=TEXT_DIR)
    worker_parser.add_argument("--workload", type=Path, required=True)
    worker_parser.add_argument("--output", type=Path, required=True)
    idle = commands.add_parser("check-idle", help="Test the idle guard without launching a model")
    idle.add_argument("--output", type=Path, required=True)
    idle.add_argument(
        "--gpu",
        help="nvidia-smi physical GPU index or UUID; defaults to CUDA device 0 "
        "under CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER",
    )
    runner = commands.add_parser("run")
    runner.add_argument("--model-dir", type=Path, default=TEXT_DIR)
    runner.add_argument("--packages", type=Path, default=ARTIFACT_DIR / "text")
    runner.add_argument(
        "--native", type=Path, default=default_build_dir("qwen35") / "xinfer_benchmark"
    )
    runner.add_argument("--validation", type=Path, default=ARTIFACT_DIR / "end-to-end/summary.json")
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument(
        "--gpu",
        help="nvidia-smi physical GPU index or UUID; defaults to CUDA device 0 "
        "under CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER",
    )
    runner.add_argument("--lengths", type=int, nargs="+", default=[22, 64, 65, 128, 511, 512])
    runner.add_argument("--decode-steps", type=int, default=64)
    runner.add_argument("--warmup", type=int, default=2)
    runner.add_argument("--repeats", type=int, default=5)
    runner.add_argument("--rounds", type=int, default=2)
    runner.add_argument(
        "--allow-busy",
        action="store_true",
        help="Functional harness checks only; timings are provisional",
    )
    args = parser.parse_args()
    if args.command == "check-idle":
        if args.output.exists():
            parser.error("Output already exists")
        gpu = resolve_gpu(args.gpu)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        wait_idle(gpu, output=args.output)
    else:
        (worker if args.command == "worker" else benchmark)(args)


if __name__ == "__main__":
    main()
