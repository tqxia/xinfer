"""Shared timing and worker mechanics; model scripts own profiles and comparisons."""

import statistics
import subprocess
import time

from compiler.common.artifacts import write_json


def memory_stats():
    import torch

    free, total = torch.cuda.mem_get_info()
    return dict(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                device_used_bytes=total - free, device_total_bytes=total)


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
    return dict(reset_ms=(prefill - begin) * 1000, prefill_ms=(decode - prefill) * 1000,
                first_prediction_ms=(decode - begin) * 1000, decode_ms=(end - decode) * 1000,
                decode_ms_per_token=(end - decode) * 1000 / len(tokens), request_ms=(end - begin) * 1000,
                prediction_ids=predictions, consumed_positions=len(prompt) + len(tokens))


def measure_cases(request, workload):
    import torch

    rows = []
    for case in workload["cases"]:
        first = request(case)
        peak_device = memory_stats()["device_used_bytes"]
        samples = []
        for i in range(workload["warmup"] + workload["repeats"]):
            if i == workload["warmup"]:
                torch.cuda.reset_peak_memory_stats()
            sample = request(case)
            assert sample["prediction_ids"] == first["prediction_ids"], "Repeated predictions differ"
            peak_device = max(peak_device, memory_stats()["device_used_bytes"])
            if i >= workload["warmup"]:
                samples.append(sample)
        rows.append(dict(name=case["name"], prompt_tokens=len(case["prompt"]), decode_steps=len(case["decode_tokens"]),
                         first_request=first, samples=samples,
                         memory=memory_stats() | dict(sampled_peak_device_used_bytes=peak_device)))
        print(case["name"], "completed", flush=True)
    return rows


def distribution(values):
    values = sorted(values)

    def percentile(p):
        index = (len(values) - 1) * p
        low = int(index)
        return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)

    return dict(n=len(values), median=statistics.median(values), mean=statistics.mean(values),
                min=min(values), max=max(values), p10=percentile(.1), p90=percentile(.9))


def wait_idle(device="0", allow_busy=False, output=None):
    samples, quiet = [], 0
    for _ in range(30):
        result = subprocess.check_output(["nvidia-smi", "-i", device, "--query-gpu=utilization.gpu,utilization.memory",
                                          "--format=csv,noheader,nounits"], text=True)
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
        monitor = subprocess.Popen(["nvidia-smi", "-i", device,
            "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem",
            "--format=csv", "-l", "1"], stdout=telemetry, stderr=log)
        start = time.perf_counter()
        try:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
    return (time.perf_counter() - start) * 1000
