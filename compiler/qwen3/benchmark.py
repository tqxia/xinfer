"""Benchmark identical fixed-token requests in eager Python and native AOTI."""

import argparse
from pathlib import Path

from compiler.common.benchmark import eager_worker as measure_eager, run_benchmark, validate_comparison
from compiler.common.artifacts import require
from compiler.qwen3.packages import validate_model_metadata
from compiler.qwen3.prepare import DEFAULT_MODEL


def eager_loader():
    import torch
    from compiler.qwen3.model import QwenStep, allocate_cache, load_model

    def load(model_dir):
        model = load_model(model_dir)
        step, cache = QwenStep(model), allocate_cache(model.config)

        def predict(tokens, position):
            ids = (torch.full((1, 1), tokens[0], device="cuda", dtype=torch.int64) if len(tokens) == 1
                   else torch.tensor(tokens, dtype=torch.int64).unsqueeze(0).to("cuda"))
            pos = (torch.full((1,), position, device="cuda", dtype=torch.int64) if len(tokens) == 1
                   else torch.arange(position, position + len(tokens), device="cuda"))
            return step(ids, pos, cache).argmax(-1).item()

        def reset():
            for tensor in cache:
                tensor.zero_()

        return reset, predict

    return load


def eager_worker(args):
    measure_eager(args, eager_loader(), "eager")


def run(args):
    paired = args.command == "compare"
    if paired:
        require(args.baseline_label != args.candidate_label and all(
            label.replace("_", "").isalnum() and label != "eager"
            for label in (args.baseline_label, args.candidate_label)),
            "Provide distinct alphanumeric/underscore labels; eager is reserved")
    packages = ({args.baseline_label: args.baseline_packages, args.candidate_label: args.candidate_packages}
                if paired else dict(aoti=args.packages))

    def tokens():
        directory = args.reference / "capital"
        return ([int(word) for word in (directory / "prompt.ids").read_text().split()],
                [int(word) for word in (directory / "generated.ids").read_text().split()])

    def compare(profiles, weights):
        validate_comparison(profiles[args.baseline_label], profiles[args.candidate_label],
                            weights[args.baseline_label], weights[args.candidate_label])

    return run_benchmark(args, model_type="qwen3", packages=packages,
                         eager={} if paired else {"eager": ("compiler.qwen3.benchmark", "eager")},
                         reference=args.baseline_label if paired else "eager", prepare_tokens=tokens,
                         manifest_name="model_manifest.json", validate_model=validate_model_metadata,
                         compare=compare if paired else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    worker = commands.add_parser('eager', help='Internal eager worker; normally use run')
    worker.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL)
    worker.add_argument('--workload', type=Path, required=True)
    worker.add_argument('--output', type=Path, required=True)
    for name in ('run', 'compare'):
        runner = commands.add_parser(name)
        runner.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL)
        runner.add_argument('--reference', type=Path, default=Path('artifacts/reference'))
        runner.add_argument('--native', type=Path, default=Path('build/xinfer_benchmark'))
        if name == 'compare':
            runner.add_argument('--baseline-packages', type=Path, default=Path('artifacts/aoti-baseline'))
            runner.add_argument('--candidate-packages', type=Path, default=Path('artifacts/aoti-candidate'))
            runner.add_argument('--baseline-label', default='baseline')
            runner.add_argument('--candidate-label', default='candidate')
        else:
            runner.add_argument('--packages', type=Path, default=Path('artifacts/aoti'))
        runner.add_argument('--output', type=Path, required=True)
        runner.add_argument('--gpu', help='nvidia-smi physical GPU index or UUID; defaults to CUDA device 0 '
                            'under CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER')
        runner.add_argument('--lengths', type=int, nargs='+', default=[25, 128, 512])
        runner.add_argument('--decode-steps', type=int, default=64)
        runner.add_argument('--warmup', type=int, default=2)
        runner.add_argument('--repeats', type=int, default=5)
        runner.add_argument('--rounds', type=int, default=4 if name == 'compare' else 3)
        runner.add_argument('--allow-busy', action='store_true', help='Provisional measurements or harness smoke tests only')
    args = parser.parse_args()
    (eager_worker if args.command == 'eager' else run)(args)


if __name__ == '__main__':
    main()
