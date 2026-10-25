"""Compare native decode operators and capture their CUDA kernels (no CUDA Graphs).

Loop times include Python dispatch, native scheduling, allocation, and GPU work.
Use trace kernel durations for device work and the C++ model benchmark for the
deployment decision; CUDA event intervals here can include host launch gaps.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"compiler"))
import torch
from custom_ops.flashinfer_decode import SOURCE, load_flashinfer_ops
from model import cache_extent


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[25, 128, 256, 257, 512, 576, 1024])
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(43)
    load_flashinfer_ops()
    q = torch.randn(1,16,1,128,device="cuda",dtype=torch.bfloat16)
    k = torch.randn(1,8,1024,128,device="cuda",dtype=torch.bfloat16)
    v = torch.randn_like(k)
    ops = {"flash": torch.ops.xinfer.flash_attention.default,
           "flashinfer": torch.ops.xinfer.flashinfer_decode.default}
    report = dict(source=SOURCE, torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                  repeats=args.repeats, rounds=args.rounds, cuda_graphs=False, cases=[])
    for length in args.lengths:
        extent = cache_extent(length)
        for op in ops.values():
            for _ in range(50):
                op(q,k,v,extent,128**-0.5)
        torch.cuda.synchronize()
        times = {label: [] for label in ops}
        for round_index in range(args.rounds):
            for label in (list(ops) if round_index % 2 == 0 else list(reversed(ops))):
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                begin = time.perf_counter()
                start.record()
                for _ in range(args.repeats):
                    ops[label](q,k,v,extent,128**-0.5)
                end.record()
                end.synchronize()
                times[label].append(dict(wall_us=(time.perf_counter()-begin)*1e6/args.repeats,
                                         event_us=start.elapsed_time(end)*1000/args.repeats))
        for label, op in ops.items():
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as profile:
                for _ in range(10):
                    op(q,k,v,extent,128**-0.5)
                torch.cuda.synchronize()
            trace_path = args.output/f"{label}-{length}.trace.json"
            profile.export_chrome_trace(str(trace_path))
            kernels = {}
            for event in json.loads(trace_path.read_text())["traceEvents"]:
                if event.get("cat") == "kernel":
                    item = kernels.setdefault(event["name"], dict(count=0, total_us=0))
                    item["count"] += 1
                    item["total_us"] += event["dur"]
            assert kernels, "Profiler did not capture CUDA kernels"
            report["cases"].append(dict(length=length, operator=label, rounds=times[label], kernels=kernels,
                wall_us_median=statistics.median(t["wall_us"] for t in times[label]),
                event_us_median=statistics.median(t["event_us"] for t in times[label])))
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    for case in report["cases"]:
        print(case["length"],case["operator"],round(case["wall_us_median"],3),"us/call",flush=True)


if __name__ == "__main__":
    main()
