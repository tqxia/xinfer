"""Benchmark identical fixed-token requests in eager Python and native AOTI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

from prepare import DEFAULT_MODEL

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def digest(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def eager_worker(args):
    import torch
    from model import QwenStep, allocate_cache, load_model

    workload = read_json(args.workload)
    if args.output.exists():
        raise ValueError('Output already exists')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def memory():
        free, total = torch.cuda.mem_get_info()
        return dict(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(), device_used_bytes=total-free,
                    device_total_bytes=total)

    with torch.inference_mode():
        start = time.perf_counter()
        torch.empty(0, device='cuda')
        torch.cuda.synchronize()
        init_ms = (time.perf_counter()-start)*1000
        initial_memory = memory()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        model = load_model(args.model_dir)
        fn, cache = QwenStep(model), allocate_cache(model.config)
        torch.cuda.synchronize()
        load_ms = (time.perf_counter()-start)*1000
        report = dict(kind='eager', torch_version=torch.__version__, cpu_threads=torch.get_num_threads(),
                      cuda_init_ms=init_ms, load_ms=load_ms, initial_memory=initial_memory,
                      load_memory=memory(), cases=[])

        def step(tokens, position):
            ids = (torch.full((1, 1), tokens[0], device='cuda', dtype=torch.int64) if len(tokens) == 1
                   else torch.tensor(tokens, dtype=torch.int64).unsqueeze(0).to('cuda'))
            pos = (torch.full((1,), position, device='cuda', dtype=torch.int64) if len(tokens) == 1
                   else torch.arange(position, position+len(tokens), device='cuda'))
            return fn(ids, pos, cache).argmax(-1).item()

        def request(case):
            predictions = []
            prompt, tokens = case['prompt'], case['decode_tokens']
            torch.cuda.synchronize()
            begin = time.perf_counter()
            for tensor in cache:
                tensor.zero_()
            torch.cuda.synchronize()
            prefill_start = time.perf_counter()
            predictions.append(step(prompt, 0))
            torch.cuda.synchronize()
            decode_start = time.perf_counter()
            for position, token in enumerate(tokens, len(prompt)):
                predictions.append(step([token], position))
            torch.cuda.synchronize()
            end = time.perf_counter()
            return dict(reset_ms=(prefill_start-begin)*1000, prefill_ms=(decode_start-prefill_start)*1000,
                        decode_ms=(end-decode_start)*1000, decode_ms_per_token=(end-decode_start)*1000/len(tokens),
                        request_ms=(end-begin)*1000, prediction_ids=predictions)

        for case in workload['cases']:
            first = request(case)
            peak_device = memory()['device_used_bytes']
            for _ in range(workload['warmup']):
                request(case)
                peak_device = max(peak_device, memory()['device_used_bytes'])
            torch.cuda.reset_peak_memory_stats()
            samples = []
            for _ in range(workload['repeats']):
                sample = request(case)
                assert sample['prediction_ids'] == first['prediction_ids'], 'Repeated predictions differ'
                samples.append(sample)
                peak_device = max(peak_device, memory()['device_used_bytes'])
            report['cases'].append(dict(name=case['name'], prompt_tokens=len(case['prompt']),
                                        decode_steps=len(case['decode_tokens']), first_request=first,
                                        samples=samples,
                                        memory=memory() | {'sampled_peak_device_used_bytes': peak_device}))
            print(case['name'], ':', len(samples), 'measured requests', flush=True)
        write_json(args.output, report)


def wait_idle(allow_busy):
    samples, quiet = [], 0
    for _ in range(30):
        result = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=utilization.gpu,utilization.memory',
                                           '--format=csv,noheader,nounits'], text=True)
        gpu, memory = (int(x.strip()) for x in result.strip().split(','))
        samples.append(dict(gpu_percent=gpu, memory_percent=memory))
        quiet = quiet+1 if gpu <= 10 and memory <= 10 else 0
        if allow_busy or quiet >= 5:
            return samples
        time.sleep(1)
    raise RuntimeError(f'GPU is not idle. Last readings: {samples[-5:]}')


def distribution(values):
    ordered = sorted(values)
    def percentile(p):
        index = (len(ordered)-1)*p
        low = int(index)
        high = min(low+1, len(ordered)-1)
        return ordered[low] + (ordered[high]-ordered[low])*(index-low)
    return dict(n=len(values), median=statistics.median(values), mean=statistics.mean(values),
                min=min(values), max=max(values), p10=percentile(.1), p90=percentile(.9))


def summarize(records, workload, reference='eager'):
    result = {}
    for label in dict.fromkeys(x['profile'] for x in records):
        runs = [read_json(x['result']) for x in records if x['profile'] == label]
        profile = dict(load_ms=distribution([r['load_ms'] for r in runs]), cases={})
        for case in workload['cases']:
            rows = [next(c for c in r['cases'] if c['name'] == case['name']) for r in runs]
            samples = [s for row in rows for s in row['samples']]
            metrics = {key: distribution([s[key] for s in samples])
                       for key in ('reset_ms', 'prefill_ms', 'decode_ms_per_token', 'request_ms')}
            metrics['decode_tokens_per_second'] = 1000/metrics['decode_ms_per_token']['median']
            metrics['memory'] = {key: max(row['memory'][key] for row in rows)
                                 for key in ('peak_allocated_bytes', 'peak_reserved_bytes',
                                             'sampled_peak_device_used_bytes')}
            metrics['memory']['sampled_peak_device_delta_bytes'] = max(
                row['memory']['sampled_peak_device_used_bytes']-run['initial_memory']['device_used_bytes']
                for row, run in zip(rows, runs))
            predictions = rows[0]['samples'][0]['prediction_ids']
            assert all(s['prediction_ids'] == predictions for s in samples), (label, case['name'])
            metrics['prediction_ids'] = predictions
            profile['cases'][case['name']] = metrics
        result[label] = profile
    for label, profile in result.items():
        for name, case in profile['cases'].items():
            ref = result[reference]['cases'][name]['prediction_ids']
            case[f'prediction_agreement_with_{reference}'] = sum(a == b for a, b in zip(case['prediction_ids'], ref))
            if label == 'aoti_eager':
                assert case['prediction_ids'] == ref, (label, name, 'Eager-mode predictions differ')
    return result


def run(args):
    from custom_ops.attention import load_package_attention_ops

    if min(args.warmup, args.repeats, args.rounds, args.decode_steps) < 1:
        raise ValueError('Counts must be positive')
    if not args.lengths or len(set(args.lengths)) != len(args.lengths):
        raise ValueError('Provide distinct prompt lengths')
    if any(n < 1 or n > 512 or n+args.decode_steps > 1024 for n in args.lengths):
        raise ValueError('Prompt/context length outside package contract')
    if args.output.exists():
        raise ValueError('Choose an absent output directory')
    paired = args.command == 'compare'
    if paired and (args.baseline_label == args.candidate_label or any(
            not label.replace('_', '').isalnum() or label in ('eager', 'aoti_eager')
            for label in (args.baseline_label, args.candidate_label))):
        raise ValueError('Provide distinct alphanumeric/underscore labels; eager and aoti_eager are reserved')
    packages = ({args.baseline_label: args.baseline_packages.resolve(),
                 args.candidate_label: args.candidate_packages.resolve()}
                if paired else dict(aoti_eager=args.eager_packages.resolve(), aoti_fused=args.fused_packages.resolve()))
    binary = args.native.resolve()
    if not binary.is_file():
        raise ValueError('Build qwen_benchmark with -DQWEN_BUILD_BENCHMARK=ON first')
    prompt = [int(x) for x in Path('artifacts/reference/capital/prompt.ids').read_text().split()]
    continuation = [int(x) for x in Path('artifacts/reference/capital/generated.ids').read_text().split()]
    workload = dict(warmup=args.warmup, repeats=args.repeats, cases=[
        dict(name=f'length_{n}', prompt=(prompt*(n//len(prompt)+1))[:n],
             decode_tokens=(continuation*(args.decode_steps//len(continuation)+1))[:args.decode_steps])
        for n in args.lengths])
    out = args.output.resolve()
    out.mkdir(parents=True)
    write_json(out/'workload.json', workload)
    source = out/'source'
    source.mkdir()
    for name in ('compiler/benchmark.py', 'runtime/benchmark.cpp', 'runtime/runtime.h', 'compiler/model.py',
                 'compiler/export.py', 'compiler/prepare.py', 'CMakeLists.txt', 'runtime/CMakeLists.txt',
                 'runtime/custom_ops/attention.cpp', 'runtime/custom_ops/attention.h',
                 'runtime/custom_ops/flashinfer_decode.cu', 'compiler/custom_ops/flashinfer_decode.py',
                 'compiler/custom_ops/__init__.py', 'compiler/custom_ops/attention.py',
                 'compiler/passes/__init__.py', 'compiler/passes/attention.py',
                 'compiler/passes/projections.py'):
        (source/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, source/name)
    environment = dict(platform=platform.platform(), cpu_count=os.cpu_count(),
                       affinity=sorted(os.sched_getaffinity(0)),
                       git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                       source_sha256={p.relative_to(source).as_posix(): digest(p)
                                      for p in sorted(source.rglob('*')) if p.is_file()},
                       binary_sha256=digest(binary), workload_sha256=digest(out/'workload.json'),
                       model_manifest=read_json(args.model_dir/'model_manifest.json'),
                       allow_busy=args.allow_busy, cpu_threads=1, packages={})
    environment['attention_library_sha256'] = digest(ROOT/'build/libxinfer_attention_ops.so')
    environment['gpu'] = subprocess.check_output([
        'nvidia-smi', '-i', '0', '--query-gpu=name,uuid,driver_version,memory.total,clocks.max.sm,clocks.max.memory',
        '--format=csv'], text=True)
    for label, directory in packages.items():
        environment['packages'][label] = {kind: dict(metadata=read_json(directory/(kind+'.json')),
                                                    bytes=(directory/(kind+'.pt2')).stat().st_size,
                                                    sha256=digest(directory/(kind+'.pt2')))
                                           for kind in ('prefill', 'decode')}
        for entry in environment['packages'][label].values():
            load_package_attention_ops(entry['metadata'])
            assert entry['metadata'].get('eager_numerics', False) == (label == 'aoti_eager'), 'Wrong compilation profile'
            if paired:
                assert entry['metadata'].get('attention_backend', 'efficient') in (
                    'efficient', 'flash', 'flash-custom', 'flashinfer-decode'), 'Unsupported attention backend'
    if paired:
        for kind in ('prefill', 'decode'):
            baseline = environment['packages'][args.baseline_label][kind]['metadata']
            candidate = environment['packages'][args.candidate_label][kind]['metadata']
            for key in ('model_revision', 'torch_version', 'torch_cuda_version', 'batch_size', 'dtype',
                        'layers', 'kv_heads', 'head_dim', 'cache_length', 'token_length',
                        'vocab_size', 'triton_enable_fp_fusion', 'inductor_configs'):
                assert baseline[key] == candidate[key], f'Incomparable packages: {kind}/{key}'
    write_json(out/'environment.json', environment)
    env = {k: v for k, v in os.environ.items()
           if k not in ('VIRTUAL_ENV', 'PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH')}
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONDONTWRITEBYTECODE='1')
    records = []
    labels = list(packages) if paired else ['eager', 'aoti_eager', 'aoti_fused']
    for round_index in range(args.rounds):
        order = labels[round_index % len(labels):]+labels[:round_index % len(labels)]
        for label in order:
            stem = f'round-{round_index+1}-{label}'
            idle = wait_idle(args.allow_busy)
            write_json(out/(stem+'.idle.json'), idle)
            result = out/(stem+'.json')
            if label == 'eager':
                command = [sys.executable, str(Path(__file__).resolve()), 'eager', '--model-dir', str(args.model_dir),
                           '--workload', str(out/'workload.json'), '--output', str(result)]
                child_env = env
            else:
                command = [str(binary), str(packages[label]), str(out/'workload.json'), str(result)]
                child_env = env | {'PATH': '/usr/bin:/bin', 'PYTHONHOME': '/nonexistent-python-home'}
            print(stem, 'starting', flush=True)
            with (out/(stem+'.gpu.csv')).open('w') as telemetry, (out/(stem+'.log')).open('w') as log:
                monitor = subprocess.Popen(['nvidia-smi', '-i', '0',
                    '--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem',
                    '--format=csv', '-l', '1'], stdout=telemetry, stderr=log)
                start = time.perf_counter()
                try:
                    subprocess.run(command, env=child_env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
                finally:
                    monitor.terminate()
                    monitor.wait(timeout=10)
            records.append(dict(round=round_index+1, profile=label, result=str(result),
                                process_ms=(time.perf_counter()-start)*1000, host_loadavg=os.getloadavg()))
            write_json(out/'runs.json', records)
            write_json(out/(stem+'.after-idle.json'), wait_idle(args.allow_busy))
            print(stem, 'finished', flush=True)
    summary = dict(status='provisional_shared_gpu' if args.allow_busy else 'idle_boundaries_checked',
                   rounds=args.rounds, warmup=args.warmup, repeats=args.repeats, decode_steps=args.decode_steps,
                   profiles=summarize(records, workload, args.baseline_label if paired else 'eager'))
    write_json(out/'summary.json', summary)
    for label, profile in summary['profiles'].items():
        for name, metrics in profile['cases'].items():
            print(label, name, 'prefill_ms=', round(metrics['prefill_ms']['median'], 3),
                  'decode_ms/token=', round(metrics['decode_ms_per_token']['median'], 3), flush=True)


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
        runner.add_argument('--native', type=Path, default=Path('build/qwen_benchmark'))
        if name == 'compare':
            runner.add_argument('--baseline-packages', type=Path, default=Path('artifacts/aoti'))
            runner.add_argument('--candidate-packages', type=Path, default=Path('artifacts/aoti-flash'))
            runner.add_argument('--baseline-label', default='aoti_fused')
            runner.add_argument('--candidate-label', default='aoti_flash')
        else:
            runner.add_argument('--eager-packages', type=Path, default=Path('artifacts/aoti-eager'))
            runner.add_argument('--fused-packages', type=Path, default=Path('artifacts/aoti'))
        runner.add_argument('--output', type=Path, required=True)
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
