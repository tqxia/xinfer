"""Fixed-token HF/native benchmark. Busy-GPU runs validate the harness only."""

import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from compiler.common.benchmark import (
    memory_stats as memory, measure_request, measure_cases, distribution,
    wait_idle as check_idle, run_worker,
)
from compiler.common.artifacts import sha256, read_json, write_json
from compiler.qwen35.config import ARTIFACT_DIR, TEXT_DIR


def wait_idle(path, device="0", allow_busy=False):
    return check_idle(device=device, allow_busy=allow_busy, output=path)


def worker(args):
    import torch
    from compiler.qwen35.reference import HFRuntime
    from compiler.qwen35.model import configure_text, load_model

    if args.output.exists():
        raise ValueError("Output already exists")
    configure_text()
    workload = read_json(args.workload)

    with torch.inference_mode():
        start = time.perf_counter()
        torch.empty(0, device="cuda")
        torch.cuda.synchronize()
        init_ms = (time.perf_counter() - start) * 1000
        initial_memory = memory()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        model = load_model(args.model_dir)
        runtime = HFRuntime(model, chunked=args.profile == "hf_chunked")
        runtime.reset()
        torch.cuda.synchronize()
        report = dict(profile=args.profile, torch=torch.__version__, cpu_threads=torch.get_num_threads(),
                      cuda_init_ms=init_ms, load_ms=(time.perf_counter() - start) * 1000,
                      initial_memory=initial_memory, load_memory=memory(), cases=[])

        def predict(tokens, position):
            return runtime.step(tokens).argmax(-1).item()

        def request(case):
            result = measure_request(runtime.reset, predict, case)
            assert runtime.position == result["consumed_positions"]
            return result

        report["cases"] = measure_cases(request, workload)
        write_json(args.output, report)


def summarize(records, workload):
    profiles = {}
    for label in ("hf", "hf_chunked", "cpp_aoti"):
        reports = [read_json(r["result"]) for r in records if r["profile"] == label]
        assert reports and all(r["cpu_threads"] == 1 for r in reports)
        profile = dict(load_ms=distribution([r["load_ms"] for r in reports]), cases={})
        for case in workload["cases"]:
            rows = [next(c for c in r["cases"] if c["name"] == case["name"]) for r in reports]
            samples = [s for row in rows for s in row["samples"]]
            expected = rows[0]["first_request"]["prediction_ids"]
            assert len(expected) == 1 + len(case["decode_tokens"])
            assert all(s["prediction_ids"] == expected and s["consumed_positions"] ==
                       len(case["prompt"]) + len(case["decode_tokens"]) for s in samples)
            metrics = {key: distribution([s[key] for s in samples]) for key in
                       ("reset_ms", "prefill_ms", "first_prediction_ms", "decode_ms_per_token", "request_ms")}
            metrics["decode_tokens_per_second"] = 1000 / metrics["decode_ms_per_token"]["median"]
            metrics["prediction_ids"] = expected
            metrics["memory"] = {k: max(row["memory"][k] for row in rows) for k in
                                 ("peak_allocated_bytes", "peak_reserved_bytes", "sampled_peak_device_used_bytes")}
            profile["cases"][case["name"]] = metrics
        profiles[label] = profile
    for case in workload["cases"]:
        for profile in profiles.values():
            row = profile["cases"][case["name"]]
            row["prediction_agreement"] = {label: sum(a == b for a, b in zip(row["prediction_ids"],
                profiles[label]["cases"][case["name"]]["prediction_ids"])) for label in ("hf", "hf_chunked")}
    return profiles


def benchmark(args):
    from transformers import AutoTokenizer
    from compiler.qwen35.prepare import encode_prompt

    if min(args.decode_steps, args.warmup, args.repeats, args.rounds) < 1:
        raise ValueError("Counts must be positive")
    if not args.lengths or len(set(args.lengths)) != len(args.lengths) or any(n < 1 or n > 512 or n + args.decode_steps > 1024 for n in args.lengths):
        raise ValueError("Provide distinct prompt lengths within the prompt/context limits")
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output.resolve()
    # Use the same physical device for CUDA logical device 0 and telemetry.
    gpu = subprocess.check_output(["nvidia-smi", "-i", args.gpu, "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if not gpu.startswith("GPU-") or "\n" in gpu:
        raise ValueError("Select one physical GPU by nvidia-smi index or UUID")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    prompt = encode_prompt(tokenizer, "What is the capital of France? Answer briefly.")
    continuation = tokenizer.encode("Paris is the capital of France. ", add_special_tokens=False)
    workload = dict(warmup=args.warmup, repeats=args.repeats, cases=[
        dict(name=f"length_{n}", prompt=(prompt * (n // len(prompt) + 1))[:n],
             decode_tokens=(continuation * (args.decode_steps // len(continuation) + 1))[:args.decode_steps]) for n in args.lengths])
    write_json(out / "workload.json", workload)
    root = Path(__file__).resolve().parents[2]
    sources = ["compiler/qwen35/" + n for n in ("benchmark.py", "reference.py", "model.py", "config.py", "prepare.py", "export.py", "requirements.txt")]
    sources += ["runtime/qwen35/benchmark.cpp", "runtime/qwen35/runtime.h", "runtime/qwen35/CMakeLists.txt", "runtime/CMakeLists.txt", "CMakeLists.txt"]
    sources += ["compiler/common/artifacts.py", "compiler/common/validation.py", "compiler/common/benchmark.py", "runtime/common/io.h",
                "runtime/common/generation.h", "runtime/common/benchmark.h"]
    for name in sources:
        target = out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    environment = dict(platform=platform.platform(), cpu_count=os.cpu_count(), affinity=sorted(os.sched_getaffinity(0)),
                       git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                       source_sha256={n: sha256(root / n) for n in sources}, binary_sha256=sha256(args.native),
                       workload_sha256=sha256(out / "workload.json"), packages={},
                       model_manifest=read_json(args.model_dir / "manifest.json"), allow_busy=args.allow_busy, gpu_uuid=gpu)
    environment["gpu"] = subprocess.check_output(["nvidia-smi", "-i", gpu,
        "--query-gpu=name,uuid,driver_version,memory.total,clocks.max.sm,clocks.max.memory", "--format=csv"], text=True)
    linked = subprocess.check_output(["ldd", str(args.native.resolve())], text=True)
    assert "libpython" not in linked and "libtorch_python" not in linked
    (out / "native.ldd").write_text(linked)
    for kind in ("prefill", "decode"):
        meta = read_json(args.packages / f"{kind}.json")
        assert meta["package_sha256"] == sha256(args.packages / f"{kind}.pt2")
        assert meta["checkpoint"] == environment["model_manifest"]
        environment["packages"][kind] = meta
    correctness = dict(passed=None, reason="No matching end-to-end report supplied")
    if args.validation.exists():
        validation = read_json(args.validation)
        assert validation["package_sha256"] == {k: m["package_sha256"] for k, m in environment["packages"].items()}, "Validation is for different packages"
        correctness = dict(passed=validation["passed"], native_bitwise=validation["native_bitwise"],
                           report_sha256=sha256(args.validation), numerical_failures=len(validation["failures"]),
                           greedy_failures=len(validation["greedy_failures"]))
        shutil.copy2(args.validation, out / "correctness.json")
    environment["correctness"] = correctness
    write_json(out / "environment.json", environment)
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH")}
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES=gpu)
    records = []
    try:
        for round_index in range(args.rounds):
            labels = ["hf", "hf_chunked", "cpp_aoti"]
            labels = labels[round_index % 3:] + labels[:round_index % 3]
            for label in labels:
                stem = f"round-{round_index + 1}-{label}"
                wait_idle(out / f"{stem}.before-idle.json", gpu, args.allow_busy)
                result = out / f"{stem}.json"
                if label == "cpp_aoti":
                    command = [str(args.native.resolve()), str(args.packages.resolve()), str(out / "workload.json"), str(result)]
                    child_env = env | dict(PATH="/usr/bin:/bin", PYTHONHOME="/nonexistent-python-home")
                else:
                    command = [sys.executable, "-m", "compiler.qwen35.benchmark", "worker", "--profile", label,
                               "--model-dir", str(args.model_dir.resolve()), "--workload", str(out / "workload.json"), "--output", str(result)]
                    child_env = env
                print(stem, "starting", flush=True)
                elapsed = run_worker(command, child_env, gpu, out / f"{stem}.gpu.csv", out / f"{stem}.log", 3600)
                records.append(dict(round=round_index + 1, profile=label, result=str(result), process_ms=elapsed))
                write_json(out / "runs.json", records)
                wait_idle(out / f"{stem}.after-idle.json", gpu, args.allow_busy)
                print(stem, "finished", flush=True)
    except Exception as error:
        write_json(out / "failure.json", dict(error=str(error), completed_runs=len(records)))
        raise
    summary = dict(status="provisional_shared_gpu" if args.allow_busy else "idle_boundaries_checked",
                   numerical_acceptance=correctness, rounds=args.rounds, warmup=args.warmup, repeats=args.repeats,
                   decode_steps=args.decode_steps, functional_checks_passed=True, profiles=summarize(records, workload))
    write_json(out / "summary.json", summary)
    print(summary["status"], "functional checks passed; numerical acceptance:", correctness["passed"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker_parser = commands.add_parser("worker", help="Internal eager worker; normally use run")
    worker_parser.add_argument("--profile", choices=("hf", "hf_chunked"), required=True)
    worker_parser.add_argument("--model-dir", type=Path, default=TEXT_DIR)
    worker_parser.add_argument("--workload", type=Path, required=True)
    worker_parser.add_argument("--output", type=Path, required=True)
    idle = commands.add_parser("check-idle", help="Test the idle guard without launching a model")
    idle.add_argument("--output", type=Path, required=True)
    idle.add_argument("--gpu", default="0", help="nvidia-smi physical GPU index or UUID")
    runner = commands.add_parser("run")
    runner.add_argument("--model-dir", type=Path, default=TEXT_DIR)
    runner.add_argument("--packages", type=Path, default=ARTIFACT_DIR / "text")
    runner.add_argument("--native", type=Path, default=Path("work/qwen35/build/qwen35_benchmark"))
    runner.add_argument("--validation", type=Path, default=ARTIFACT_DIR / "end-to-end/summary.json")
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--gpu", default="0", help="nvidia-smi physical GPU index or UUID; workers use it as CUDA device 0")
    runner.add_argument("--lengths", type=int, nargs="+", default=[22, 64, 65, 128, 511, 512])
    runner.add_argument("--decode-steps", type=int, default=64)
    runner.add_argument("--warmup", type=int, default=2)
    runner.add_argument("--repeats", type=int, default=5)
    runner.add_argument("--rounds", type=int, default=3)
    runner.add_argument("--allow-busy", action="store_true", help="Functional harness checks only; timings are provisional")
    args = parser.parse_args()
    if args.command == "check-idle":
        if args.output.exists():
            parser.error("Output already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        wait_idle(args.output, args.gpu)
    else:
        (worker if args.command == "worker" else benchmark)(args)


if __name__ == "__main__":
    main()
