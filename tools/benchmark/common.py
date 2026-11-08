"""Fixed-token benchmark measurement, environment capture, orchestration and reports."""

import os
import platform
import statistics
import subprocess
import sys
import time

from compiler.common.artifacts import (
    MODEL_MANIFEST_NAME,
    read_json,
    require,
    sha256,
    write_json,
)
from compiler.common.metadata import (
    COMPARISON_EXCLUDED_FIELDS,
    FIELDS as METADATA_FIELDS,
    read_package_metadata,
)
from tools.common import prompt_chunks


def resolve_gpu(device=None):
    """Resolve one physical GPU without creating a CUDA context in this process."""
    cuda_uuid = None
    try:
        if device is None:
            # CUDA ordinals can differ from nvidia-smi indices. Let CUDA resolve
            # its logical device 0 under the caller's visibility/order settings.
            cuda_uuid = subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    'import torch; print("GPU-" + str(torch.cuda.get_device_properties(0).uuid))',
                ],
                text=True,
            ).strip()
            if not cuda_uuid.startswith("GPU-") or "\n" in cuda_uuid:
                raise ValueError("CUDA device 0 did not resolve to one physical GPU UUID")
            device = cuda_uuid
        else:
            device = device.strip()
            if not (device.isdecimal() or device.startswith("GPU-")):
                raise ValueError("Select one physical GPU by nvidia-smi index or UUID")
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                device,
                "--query-gpu=uuid",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, OSError) as error:
        raise ValueError(
            "Cannot resolve the benchmark GPU; check CUDA_VISIBLE_DEVICES or pass "
            "--gpu with one nvidia-smi physical GPU index or UUID"
        ) from error
    if not gpu.startswith("GPU-") or "\n" in gpu:
        raise ValueError("Select one physical GPU by nvidia-smi index or UUID")
    if cuda_uuid is not None and gpu != cuda_uuid:
        raise ValueError("CUDA device 0 UUID does not match the selected physical GPU")
    return gpu


EAGER_BASELINE = "hf_dynamic_cache"
PROMPT_TEXT = "What is the capital of France? Answer briefly."
CONTINUATION_TEXT = "Paris is the capital of France. "

TIMING_FIELDS = (
    "reset_ms",
    "prefill_ms",
    "first_prediction_ms",
    "decode_ms",
    "decode_ms_per_token",
    "request_ms",
)
COMPARISON_FIELDS = tuple(
    field for field in METADATA_FIELDS if field not in COMPARISON_EXCLUDED_FIELDS
)


def memory_stats():
    import torch

    free, total = torch.cuda.mem_get_info()
    return dict(
        allocated_bytes=torch.cuda.memory_allocated(),
        reserved_bytes=torch.cuda.memory_reserved(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        device_used_bytes=total - free,
        device_total_bytes=total,
    )


def measure_request(reset, predict, case):
    """predict(tokens, position) returns an argmax token after its host read."""
    import torch

    prompt, tokens = case["prompt"], case["decode_tokens"]
    predictions = []
    torch.cuda.synchronize()
    begin = time.perf_counter()
    reset()
    torch.cuda.synchronize()
    prefill = time.perf_counter()
    predictions.append(predict(prompt, 0))
    torch.cuda.synchronize()
    decode = time.perf_counter()
    for position, token in enumerate(tokens, len(prompt)):
        predictions.append(predict([token], position))
    torch.cuda.synchronize()
    end = time.perf_counter()
    return dict(
        reset_ms=(prefill - begin) * 1000,
        prefill_ms=(decode - prefill) * 1000,
        first_prediction_ms=(decode - begin) * 1000,
        decode_ms=(end - decode) * 1000,
        decode_ms_per_token=(end - decode) * 1000 / len(tokens),
        request_ms=(end - begin) * 1000,
        prediction_ids=predictions,
        consumed_positions=len(prompt) + len(tokens),
    )


def measure_cases(request, workload):
    import torch

    rows = []
    for case in workload["cases"]:
        first = request(case)
        check_predictions(first, case)
        peak_device = memory_stats()["device_used_bytes"]
        samples = []
        for i in range(workload["warmup"] + workload["repeats"]):
            if i == workload["warmup"]:
                torch.cuda.reset_peak_memory_stats()
            sample = request(case)
            check_predictions(sample, case)
            require(
                sample["prediction_ids"] == first["prediction_ids"], "Repeated predictions differ"
            )
            peak_device = max(peak_device, memory_stats()["device_used_bytes"])
            if i >= workload["warmup"]:
                samples.append(sample)
        rows.append(
            dict(
                name=case["name"],
                prompt_tokens=len(case["prompt"]),
                decode_steps=len(case["decode_tokens"]),
                first_request=first,
                samples=samples,
                memory=memory_stats() | dict(sampled_peak_device_used_bytes=peak_device),
            )
        )
        print(case["name"], "completed", flush=True)
    return rows


def distribution(values):
    values = sorted(values)

    def percentile(p):
        index = (len(values) - 1) * p
        low = int(index)
        return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)

    return dict(
        n=len(values),
        median=statistics.median(values),
        mean=statistics.mean(values),
        min=min(values),
        max=max(values),
        p10=percentile(0.1),
        p90=percentile(0.9),
    )


def wait_idle(device="0", allow_busy=False, output=None):
    samples, quiet = [], 0
    for _ in range(30):
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                device,
                "--query-gpu=utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        gpu, memory = (int(v.strip()) for v in result.strip().split(","))
        samples.append(dict(time=time.time(), gpu_percent=gpu, memory_percent=memory))
        quiet = quiet + 1 if max(gpu, memory) <= 10 else 0
        if output is not None:
            write_json(output, dict(samples=samples, allow_busy=allow_busy, idle=quiet >= 5))
        if allow_busy or quiet >= 5:
            return samples
        time.sleep(1)
    raise RuntimeError(f"GPU is not idle; last readings: {samples[-5:]}; samples file: {output}")


def run_worker(command, env, device, telemetry_path, log_path, timeout):
    with telemetry_path.open("w") as telemetry, log_path.open("w") as log:
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "-i",
                device,
                "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem",
                "--format=csv",
                "-l",
                "1",
            ],
            stdout=telemetry,
            stderr=log,
        )
        start = time.perf_counter()
        try:
            subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout
            )
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
    return (time.perf_counter() - start) * 1000


def check_predictions(sample, case):
    require(
        len(sample["prediction_ids"]) == 1 + len(case["decode_tokens"]), "Prediction count differs"
    )
    require(
        sample["consumed_positions"] == len(case["prompt"]) + len(case["decode_tokens"]),
        "Consumed position differs",
    )


def hf_eager_loader(load_model):
    """Use ordinary HF cached forwards; HF owns cache creation, positions and masks."""

    def load(model_dir):
        import torch

        model = load_model(model_dir)
        cache, consumed = None, 0

        def reset():
            nonlocal cache, consumed
            cache, consumed = None, 0

        def predict(tokens, position):
            nonlocal cache, consumed
            require(position == consumed, "Consumed position differs")
            ids = torch.tensor([tokens], dtype=torch.int64, device="cuda")
            result = model(
                input_ids=ids,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = result.past_key_values
            require(cache is not None, "HF model did not return a cache")
            consumed += len(tokens)
            return result.logits[:, -1, :].argmax(-1).item()

        return reset, predict

    return load


def eager_worker(args, load_runtime, profile):
    """load_runtime(model_dir) returns the model's reset and predict callbacks."""
    import torch

    require(not args.output.exists(), "Output already exists")
    workload = read_json(args.workload)
    require(
        "max_prompt_length" not in workload,
        "Benchmark workload no longer accepts max_prompt_length",
    )
    if "cache_length" in workload:
        capacity = workload["cache_length"]
        for case in workload["cases"]:
            require(1 <= len(case["prompt"]) <= capacity, "Context capacity exceeded")
            require(
                len(case["prompt"]) + len(case["decode_tokens"]) <= workload["cache_length"],
                "Context capacity exceeded",
            )
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.inference_mode():
        start = time.perf_counter()
        torch.empty(0, device="cuda")
        torch.cuda.synchronize()
        init_ms = (time.perf_counter() - start) * 1000
        initial_memory = memory_stats()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        reset, predict = load_runtime(args.model_dir)
        torch.cuda.synchronize()
        report = dict(
            profile=profile,
            eager_baseline=EAGER_BASELINE,
            torch_version=torch.__version__,
            cpu_threads=torch.get_num_threads(),
            cuda_init_ms=init_ms,
            load_ms=(time.perf_counter() - start) * 1000,
            initial_memory=initial_memory,
            load_memory=memory_stats(),
        )
        report["cases"] = measure_cases(
            lambda case: measure_request(reset, predict, case), workload
        )
        write_json(args.output, report)


def make_workload(args, prompt, continuation):
    require(bool(prompt) and bool(continuation), "Prompt and continuation must contain tokens")
    return dict(
        warmup=args.warmup,
        repeats=args.repeats,
        cases=[
            dict(
                name=f"length_{n}",
                prompt=(prompt * (n // len(prompt) + 1))[:n],
                decode_tokens=(continuation * (args.decode_steps // len(continuation) + 1))[
                    : args.decode_steps
                ],
            )
            for n in args.lengths
        ],
    )


def summarize(records, workload, reference):
    """Aggregate one report format for eager/native and arbitrary paired profiles."""
    result = {}
    for label in dict.fromkeys(record["profile"] for record in records):
        runs = [read_json(record["result"]) for record in records if record["profile"] == label]
        require(all(run["cpu_threads"] == 1 for run in runs), "Benchmark requires one CPU thread")
        profile = dict(load_ms=distribution([run["load_ms"] for run in runs]), cases={})
        for case in workload["cases"]:
            rows = []
            for run in runs:
                require(
                    [row["name"] for row in run["cases"]] == [c["name"] for c in workload["cases"]],
                    "Benchmark case names/order differ",
                )
                row = next(row for row in run["cases"] if row["name"] == case["name"])
                require(
                    row["prompt_tokens"] == len(case["prompt"])
                    and row["decode_steps"] == len(case["decode_tokens"]),
                    "Benchmark case lengths differ",
                )
                require(
                    len(row["samples"]) == workload["repeats"], "Benchmark sample count differs"
                )
                rows.append(row)
            expected = rows[0]["first_request"]["prediction_ids"]
            samples = [sample for row in rows for sample in row["samples"]]
            for sample in [row["first_request"] for row in rows] + samples:
                check_predictions(sample, case)
                require(sample["prediction_ids"] == expected, "Repeated predictions differ")
            metrics = {
                key: distribution([sample[key] for sample in samples]) for key in TIMING_FIELDS
            }
            median = metrics["decode_ms_per_token"]["median"]
            require(median > 0, "Decode timing must be positive")
            metrics["decode_tokens_per_second"] = 1000 / median
            metrics["memory"] = {
                key: max(row["memory"][key] for row in rows)
                for key in (
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                    "sampled_peak_device_used_bytes",
                )
            }
            metrics["memory"]["sampled_peak_device_delta_bytes"] = max(
                row["memory"]["sampled_peak_device_used_bytes"]
                - run["initial_memory"]["device_used_bytes"]
                for row, run in zip(rows, runs)
            )
            metrics["prediction_ids"] = expected
            profile["cases"][case["name"]] = metrics
        result[label] = profile
    require(reference in result, "Reference profile is missing")
    for profile in result.values():
        for name, case in profile["cases"].items():
            ref = result[reference]["cases"][name]["prediction_ids"]
            case["prediction_agreement"] = {
                reference: sum(a == b for a, b in zip(case["prediction_ids"], ref))
            }
    return result


def validate_comparison(baseline, candidate, baseline_weights, candidate_weights):
    """Compare model inputs/numerical settings while allowing different compiled artifacts."""
    decode_profiles = (
        baseline["decode"]["metadata"]["model_specific"],
        candidate["decode"]["metadata"]["model_specific"],
    )
    qwen3_backend_change = baseline["decode"]["metadata"]["model_type"] == candidate["decode"][
        "metadata"
    ]["model_type"] == "qwen3" and decode_profiles in (
        ({}, {"decode_backend": "megaqwen"}),
        ({"decode_backend": "megaqwen"}, {}),
    )
    for kind in ("prefill", "decode"):
        before, after = (profile[kind]["metadata"] for profile in (baseline, candidate))
        for key in COMPARISON_FIELDS:
            if key == "model_specific" and qwen3_backend_change:
                require(
                    (before[key], after[key]) == decode_profiles,
                    "Incomparable packages: prefill/decode profiles differ",
                )
                continue
            require(before[key] == after[key], f"Incomparable packages: {kind}/{key}")
    require(
        baseline_weights["identity_sha256"] == candidate_weights["identity_sha256"],
        "Incomparable shared weights: identity_sha256",
    )
    # MegaQwen materializes immutable RoPE tables from the same checkpoint.
    # Compare every canonical model tensor while permitting these derived buffers.
    derived = {"megaqwen_cos", "megaqwen_sin"} if qwen3_backend_change else set()
    before, after = (
        {name: tensor for name, tensor in weights["sources"].items() if name not in derived}
        for weights in (baseline_weights, candidate_weights)
    )
    require(before == after, "Incomparable shared weights: sources")


def worker_environment(gpu, native=False):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH")
    }
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES=gpu,
    )
    if native:
        env.update(PATH="/usr/bin:/bin", PYTHONHOME="/nonexistent-python-home")
    return env


def capture_environment(args, out, gpu, model_type, packages, validate_model):
    from compiler.common.weights import validate_shared_weights

    binary = args.native.resolve()
    require(binary.is_file(), "Build xinfer_benchmark with -DXINFER_BUILD_BENCHMARK=ON first")
    linked = subprocess.check_output(
        ["ldd", str(binary)], env=worker_environment(gpu, native=True), text=True
    )
    require(
        "libpython" not in linked and "libtorch_python" not in linked,
        "Native benchmark links Python",
    )
    (out / "native.ldd").write_text(linked)
    manifest = read_json(args.model_dir / MODEL_MANIFEST_NAME)
    environment = dict(
        platform=platform.platform(),
        cpu_count=os.cpu_count(),
        affinity=sorted(os.sched_getaffinity(0)),
        binary_sha256=sha256(binary),
        workload_sha256=sha256(out / "workload.json"),
        model_manifest=manifest,
        allow_busy=args.allow_busy,
        cpu_threads=1,
        gpu_uuid=gpu,
        packages={},
        shared_weights={},
    )
    environment["gpu"] = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            gpu,
            "--query-gpu=name,uuid,driver_version,memory.total,clocks.max.sm,clocks.max.memory",
            "--format=csv",
        ],
        text=True,
    )
    bundles = {}
    for label, directory in packages.items():
        if directory in bundles:
            entries, weights = bundles[directory]
            environment["packages"][label] = entries
            environment["shared_weights"][label] = weights
            continue
        metadata = read_package_metadata(directory, model_type=model_type)
        entries = {}
        for kind, meta in metadata.items():
            validate_model(meta)
            package = directory / f"{kind}.pt2"
            size, digest = package.stat().st_size, sha256(package)
            require(
                meta["package_bytes"] == size and meta["package_sha256"] == digest,
                f"Package file differs from metadata: {label}/{kind}",
            )
            require(
                meta["checkpoint"]["manifest"] == manifest, "Package checkpoint manifest differs"
            )
            entries[kind] = dict(
                metadata=meta,
                bytes=size,
                sha256=digest,
            )
        environment["packages"][label] = entries
        environment["shared_weights"][label] = validate_shared_weights(
            directory, metadata["prefill"], metadata["decode"]
        )
        bundles[directory] = entries, environment["shared_weights"][label]
    return environment


def run_rounds(args, out, gpu, model_type, packages, eager, records, timeout):
    """eager maps one optional profile label to its Python module and worker subcommand."""
    labels = [*eager, *packages]
    require(
        bool(labels) and len(set(labels)) == len(labels),
        "Benchmark profile labels must be distinct",
    )
    for round_index in range(args.rounds):
        order = labels[round_index % len(labels) :] + labels[: round_index % len(labels)]
        for label in order:
            stem = f"round-{round_index + 1}-{label}"
            wait_idle(gpu, args.allow_busy, out / f"{stem}.before-idle.json")
            result = out / f"{stem}.json"
            if label in eager:
                module, command = eager[label]
                argv = [
                    sys.executable,
                    "-m",
                    module,
                    command,
                    "--model-type",
                    model_type,
                    "--model-dir",
                    str(args.model_dir.resolve()),
                    "--workload",
                    str(out / "workload.json"),
                    "--output",
                    str(result),
                ]
            else:
                argv = [
                    str(args.native.resolve()),
                    str(packages[label]),
                    str(out / "workload.json"),
                    str(result),
                ]
            print(stem, "starting", flush=True)
            elapsed = run_worker(
                argv,
                worker_environment(gpu, native=label not in eager),
                gpu,
                out / f"{stem}.gpu.csv",
                out / f"{stem}.log",
                timeout,
            )
            records.append(
                dict(
                    round=round_index + 1,
                    profile=label,
                    result=str(result),
                    process_ms=elapsed,
                    host_loadavg=os.getloadavg(),
                )
            )
            write_json(out / "runs.json", records)
            wait_idle(gpu, args.allow_busy, out / f"{stem}.after-idle.json")
            print(stem, "finished", flush=True)


def run_benchmark(
    args,
    *,
    model_type,
    metadata,
    packages,
    eager,
    reference,
    prepare_tokens,
    validate_model,
    compare=None,
    timeout=900,
):
    require(
        min(args.warmup, args.repeats, args.rounds, args.decode_steps) >= 1,
        "Counts must be positive",
    )

    def supported(length):
        prefill, decode = metadata["prefill"], metadata["decode"]
        if not (1 <= length and length + args.decode_steps <= decode["cache_length"]):
            return False
        lower, upper = prefill["token_range"]
        return all(
            len(chunk) == 1 or lower <= len(chunk) <= upper
            for chunk in prompt_chunks(range(length), upper)
        )

    require(
        bool(args.lengths)
        and len(set(args.lengths)) == len(args.lengths)
        and all(supported(length) for length in args.lengths),
        "Provide distinct prompt lengths within the prompt/context limits",
    )
    require(not args.output.exists(), "Choose an absent output directory")
    gpu = resolve_gpu(args.gpu)
    out = args.output.resolve()
    out.mkdir(parents=True)
    records = []
    try:
        workload = make_workload(args, *prepare_tokens())
        workload["cache_length"] = metadata["decode"]["cache_length"]
        write_json(out / "workload.json", workload)
        packages = {label: directory.resolve() for label, directory in packages.items()}
        environment = capture_environment(args, out, gpu, model_type, packages, validate_model)
        if compare is not None:
            compare(environment["packages"], environment["shared_weights"])
        write_json(out / "environment.json", environment)
        run_rounds(args, out, gpu, model_type, packages, eager, records, timeout)
        summary = dict(
            schema_version=1,
            model_type=model_type,
            status="provisional_shared_gpu" if args.allow_busy else "idle_boundaries_checked",
            rounds=args.rounds,
            warmup=args.warmup,
            repeats=args.repeats,
            decode_steps=args.decode_steps,
            functional_checks_passed=True,
            prefill_execution=dict(
                native_chunk_tokens=metadata["prefill"]["token_range"][1],
                hf="whole_prompt" if eager else None,
            ),
            profiles=summarize(records, workload, reference),
        )
        if eager:
            summary["eager_baseline"] = EAGER_BASELINE
        write_json(out / "summary.json", summary)
    except Exception as error:
        write_json(out / "failure.json", dict(error=str(error), completed_runs=len(records)))
        raise
    for label, profile in summary["profiles"].items():
        for name, metrics in profile["cases"].items():
            print(
                label,
                name,
                "prefill_ms=",
                round(metrics["prefill_ms"]["median"], 3),
                "decode_ms/token=",
                round(metrics["decode_ms_per_token"]["median"], 3),
                flush=True,
            )
    return summary
