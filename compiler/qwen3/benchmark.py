"""Benchmark identical fixed-token requests in eager Python and native AOTI."""
import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from compiler.qwen3.prepare import DEFAULT_MODEL
from compiler.common.benchmark import (
    memory_stats as memory, measure_request, measure_cases, distribution,
    wait_idle as check_idle, run_worker,
)
from compiler.common.artifacts import read_json, write_json, sha256 as digest

ROOT = Path(__file__).resolve().parents[2]


def wait_idle(allow_busy):
    return check_idle(allow_busy=allow_busy)


def eager_worker(args):
    import torch
    from compiler.qwen3.model import QwenStep, allocate_cache, load_model

    workload = read_json(args.workload)
    if args.output.exists():
        raise ValueError('Output already exists')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

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

        def reset():
            for tensor in cache:
                tensor.zero_()

        report["cases"] = measure_cases(lambda case: measure_request(reset, step, case), workload)
        write_json(args.output, report)


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
    from compiler.qwen3.custom_ops.attention import native_attention_library

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
    prompt = [int(x) for x in (args.reference/'capital/prompt.ids').read_text().split()]
    continuation = [int(x) for x in (args.reference/'capital/generated.ids').read_text().split()]
    workload = dict(warmup=args.warmup, repeats=args.repeats, cases=[
        dict(name=f'length_{n}', prompt=(prompt*(n//len(prompt)+1))[:n],
             decode_tokens=(continuation*(args.decode_steps//len(continuation)+1))[:args.decode_steps])
        for n in args.lengths])
    out = args.output.resolve()
    out.mkdir(parents=True)
    write_json(out/'workload.json', workload)
    source = out/'source'
    source.mkdir()
    for name in ('compiler/qwen3/benchmark.py', 'runtime/qwen3/benchmark.cpp', 'runtime/qwen3/runtime.h', 'compiler/qwen3/model.py',
                 'compiler/qwen3/export.py', 'compiler/qwen3/prepare.py', 'CMakeLists.txt', 'runtime/CMakeLists.txt',
                 'runtime/qwen3/custom_ops/attention.cpp', 'runtime/qwen3/custom_ops/attention.h',
                 'compiler/qwen3/custom_ops/attention.py', 'compiler/qwen3/custom_ops/__init__.py',
                 'compiler/qwen3/adapters/qwen3_flash.py', 'compiler/qwen3/adapters/__init__.py', 'compiler/common/artifacts.py',
                 'runtime/common/io.h', 'runtime/common/generation.h', 'runtime/common/benchmark.h',
                 'compiler/common/benchmark.py', 'runtime/qwen3/CMakeLists.txt'):
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
    library = native_attention_library(binary)
    if library is not None:
        environment['attention_library'] = dict(path=str(library), sha256=digest(library))
    environment['gpu'] = subprocess.check_output([
        'nvidia-smi', '-i', '0', '--query-gpu=name,uuid,driver_version,memory.total,clocks.max.sm,clocks.max.memory',
        '--format=csv'], text=True)
    for label, directory in packages.items():
        environment['packages'][label] = {kind: dict(metadata=read_json(directory/(kind+'.json')),
                                                    bytes=(directory/(kind+'.pt2')).stat().st_size,
                                                    sha256=digest(directory/(kind+'.pt2')))
                                           for kind in ('prefill', 'decode')}
        for entry in environment['packages'][label].values():
            backend = entry['metadata'].get('attention_backend', 'efficient')
            if backend not in ('efficient', 'flash'):
                raise ValueError(f'Unsupported attention backend: {backend}')
            if backend == 'flash' and library is None:
                raise ValueError('Build the native benchmark with -DQWEN_ENABLE_FLASH_ATTENTION=ON')
            if not paired:
                assert entry['metadata'].get('eager_numerics', False) == (label == 'aoti_eager'), 'Wrong compilation profile'
    if paired:
        # Original fused packages predate these fields; match runtime defaults.
        defaults = dict(eager_numerics=False, triton_enable_fp_fusion=True)
        for kind in ('prefill', 'decode'):
            baseline = defaults | environment['packages'][args.baseline_label][kind]['metadata']
            candidate = defaults | environment['packages'][args.candidate_label][kind]['metadata']
            for key in ('model_revision', 'torch_version', 'torch_cuda_version', 'batch_size', 'dtype',
                        'layers', 'kv_heads', 'head_dim', 'cache_length', 'token_length',
                        'vocab_size', 'cxx11_abi', 'cuda_compute_capability', 'eager_numerics',
                        'triton_enable_fp_fusion', 'inductor_configs'):
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
                command = [sys.executable, '-m', 'compiler.qwen3.benchmark', 'eager', '--model-dir', str(args.model_dir),
                           '--workload', str(out/'workload.json'), '--output', str(result)]
                child_env = env
            else:
                command = [str(binary), str(packages[label]), str(out/'workload.json'), str(result)]
                child_env = env | {'PATH': '/usr/bin:/bin', 'PYTHONHOME': '/nonexistent-python-home'}
            print(stem, 'starting', flush=True)
            elapsed = run_worker(command, child_env, "0", out/(stem+".gpu.csv"), out/(stem+".log"), 900)
            records.append(dict(round=round_index+1, profile=label, result=str(result),
                                process_ms=elapsed, host_loadavg=os.getloadavg()))
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
        runner.add_argument('--reference', type=Path, default=Path('artifacts/reference'))
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
