import csv
import datetime
import io
import json
import pathlib
import shutil
import statistics
import subprocess
import threading
import time

import torch


root = pathlib.Path('artifacts') / ('clock-diagnostic-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
root.mkdir(parents=True, exist_ok=False)
shutil.copyfile(__file__, root / 'diagnostic.py')
samples = []
phases = []
stop = threading.Event()
fields = 'timestamp,pstate,clocks.sm,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks_event_reasons.active'


def snapshot(name):
    proc = subprocess.run(['nvidia-smi', '-q', '-d', 'PERFORMANCE,POWER,CLOCK,TEMPERATURE'], text=True, capture_output=True, check=True)
    (root / (name + '.txt')).write_text(proc.stdout)


def monitor():
    with (root / 'telemetry.csv').open('w') as log:
        proc = subprocess.Popen(['nvidia-smi', '--query-gpu=' + fields, '--format=csv,noheader,nounits', '--loop-ms=200'], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            for line in proc.stdout:
                when = time.time()
                log.write(line)
                log.flush()
                values = [s.strip() for s in next(csv.reader(io.StringIO(line)))]
                if len(values) == 8:
                    try:
                        samples.append(dict(epoch=when, timestamp=values[0], pstate=values[1], sm_mhz=float(values[2]), gpu_util_pct=float(values[3]), memory_util_pct=float(values[4]), temperature_c=float(values[5]), gpu_power_w=float(values[6]), clock_event_reasons=values[7]))
                    except ValueError:
                        pass
                if stop.is_set():
                    break
        finally:
            proc.terminate()
            proc.wait(timeout=5)


def stats(records):
    result = {'samples': len(records)}
    for field in ('sm_mhz', 'gpu_util_pct', 'memory_util_pct', 'temperature_c', 'gpu_power_w'):
        values = [r[field] for r in records]
        if values:
            result[field] = dict(min=min(values), median=statistics.median(values), max=max(values))
    result['clock_event_reasons'] = sorted({r['clock_event_reasons'] for r in records})
    result['pstates'] = sorted({r['pstate'] for r in records})
    return result


print('Output: ' + str(root), flush=True)
torch.set_num_threads(1)
snapshot('before')
thread = threading.Thread(target=monitor, daemon=True)
thread.start()
try:
    start = time.time()
    time.sleep(3)
    phases.append(dict(name='idle_before', start=start, end=time.time()))
    torch.cuda.init()
    torch.manual_seed(1234)
    for dtype in (torch.bfloat16, torch.float16):
        label = str(dtype).split('.')[-1]
        print('Starting 15-second ' + label + ' 8192x8192 GEMM', flush=True)
        n = 8192
        a = torch.randn((n, n), device='cuda', dtype=dtype)
        b = torch.randn_like(a)
        c = torch.empty_like(a)
        for _ in range(3):
            torch.mm(a, b, out=c)
        torch.cuda.synchronize()
        start = time.time()
        deadline = time.perf_counter() + 15
        total_ms = 0.0
        count = 0
        snap_done = False
        while time.perf_counter() < deadline:
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(8):
                torch.mm(a, b, out=c)
            end.record()
            end.synchronize()
            total_ms += begin.elapsed_time(end)
            count += 8
            if not snap_done and time.time() - start >= 7:
                snapshot('during_' + label)
                snap_done = True
        finish = time.time()
        finite = bool(torch.isfinite(c).all().item())
        phase = dict(name=label, start=start, end=finish, dimension=n, iterations=count, cuda_elapsed_ms=total_ms, cuda_ms_per_gemm=total_ms / count, dense_tflops=2 * n**3 * count / (total_ms / 1000) / 1e12, output_finite=finite)
        phases.append(phase)
        print(json.dumps(phase), flush=True)
        del a, b, c
    torch.cuda.synchronize()
    start = time.time()
    time.sleep(3)
    phases.append(dict(name='idle_after', start=start, end=time.time()))
    snapshot('after')
finally:
    stop.set()
    thread.join(timeout=10)
    for phase in phases:
        # Discard the first second of each phase to avoid previous utilization samples.
        records = [s for s in samples if phase['start'] + 1 <= s['epoch'] <= phase['end']]
        phase['telemetry'] = stats(records)
    result = dict(torch_version=torch.__version__, device=torch.cuda.get_device_name(), phases=phases, samples=samples)
    (root / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'samples'}, indent=2), flush=True)
    print('Saved to ' + str(root), flush=True)
