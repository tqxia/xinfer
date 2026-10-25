"""Analyze Nsight SQLite using CUDA launch correlation inside NVTX stages."""
import argparse
import collections
import json
import sqlite3
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('sqlite', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
db = sqlite3.connect(args.sqlite)
db.row_factory = sqlite3.Row
strings = dict(db.execute('select id,value from StringIds'))
ranges = [dict(r) for r in db.execute('select * from NVTX_EVENTS where end is not null')]
apis = [dict(r) for r in db.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')]
for r in apis:
    r['name'] = strings[r['nameId']]
api_by_corr = {a['correlationId']: a for a in apis}
ops = []
for kind, table in [('kernel', 'KERNEL'), ('memcpy', 'MEMCPY'), ('memset', 'MEMSET')]:
    for row in db.execute('select * from CUPTI_ACTIVITY_KIND_' + table):
        row = dict(row)
        row['kind'] = kind
        row['name'] = strings[row['demangledName']] if kind == 'kernel' else kind
        ops.append(row)

def merged_duration(rows):
    total = 0
    begin = end = None
    for row in sorted(rows, key=lambda r: r['start']):
        a, b = row['start'], row['end']
        if end is None or a > end:
            if end is not None:
                total += end-begin
            begin, end = a, b
        else:
            end = max(end, b)
    return total + (end-begin if end is not None else 0)

def group(name):
    n = name.lower()
    if ('fmha' in n or 'pytorch_flash::' in n or
            ('flashinfer::' in n and ('decodewithkvcache' in n or 'mergestates' in n))):
        return 'attention'
    if 'gemv' in n:
        return 'linear_gemv'
    if 'gemm' in n or 'nvjet_' in n:
        return 'linear_gemm'
    if 'argmax' in n:
        return 'argmax'
    if 'triton_' in n and 'clone_expand' in n:
        return 'kv_head_expansion'
    if 'index_copy' in n:
        return 'cache_update_with_fused_ops'
    if 'silu' in n:
        return 'silu_multiply'
    if 'triton_' in n:
        return 'other_fused_pointwise_reduction'
    return 'other'

records = []
for stage in [r for r in ranges if r['text'] in ('prefill', 'decode', 'reset')]:
    parent = next(r for r in ranges if r['text'].startswith('case:') and r['start'] <= stage['start'] and r['end'] >= stage['end'])
    stage_apis = [a for a in apis if a['globalTid'] == stage['globalTid'] and stage['start'] <= a['start'] and a['end'] <= stage['end']]
    correlations = {a['correlationId'] for a in stage_apis}
    work = [r for r in ops if r['correlationId'] in correlations]
    kernels = sorted([r for r in work if r['kind'] == 'kernel'], key=lambda r: r['start'])
    linear = [r for r in kernels if group(r['name']).startswith('linear_')]
    # Generated wrapper ends with the single last-token vocabulary projection.
    head = linear[-1] if linear else None
    groups = collections.defaultdict(lambda: {'count': 0, 'gpu_ms': 0})
    kernel_names = collections.defaultdict(lambda: {'count': 0, 'gpu_ms': 0})
    for k in kernels:
        category = 'lm_head' if k is head else group(k['name'])
        for target in (groups[category], kernel_names[k['name']]):
            target['count'] += 1
            target['gpu_ms'] += (k['end']-k['start'])/1e6
    api_groups = collections.defaultdict(lambda: {'count': 0, 'cpu_ms': 0})
    for a in stage_apis:
        api_groups[a['name']]['count'] += 1
        api_groups[a['name']]['cpu_ms'] += (a['end']-a['start'])/1e6
    children = {r['text']: (r['end']-r['start'])/1e6 for r in ranges if r['text'] in ('forward', 'argmax_host') and r['start'] >= stage['start'] and r['end'] <= stage['end']}
    wall = stage['end']-stage['start']
    busy = merged_duration(work)
    assert all(stage['start'] <= k['start'] and k['end'] <= stage['end'] for k in work)
    records.append(dict(case=parent['text'][5:], stage=stage['text'], wall_ms=wall/1e6,
                        gpu_busy_ms=busy/1e6, gpu_idle_ms=(wall-busy)/1e6,
                        kernel_sum_ms=sum(k['end']-k['start'] for k in kernels)/1e6,
                        kernels=len(kernels), gpu_ops=len(work), groups=dict(groups),
                        api_groups=dict(api_groups), children_ms=children,
                        kernel_names=dict(kernel_names)))

summary = {}
for case in sorted({r['case'] for r in records}):
    summary[case] = {}
    for stage in ('prefill', 'decode', 'reset'):
        rows = [r for r in records if r['case'] == case and r['stage'] == stage]
        if not rows:
            continue
        item = {'n': len(rows)}
        for key in ('wall_ms', 'gpu_busy_ms', 'gpu_idle_ms', 'kernel_sum_ms', 'kernels', 'gpu_ops'):
            item[key] = {'mean': statistics.mean(r[key] for r in rows), 'median': statistics.median(r[key] for r in rows)}
        item['gpu_busy_pct'] = 100*sum(r['gpu_busy_ms'] for r in rows)/sum(r['wall_ms'] for r in rows)
        item['groups'] = {}
        for g in sorted({g for r in rows for g in r['groups']}):
            totals = [r['groups'].get(g, {'gpu_ms':0, 'count':0}) for r in rows]
            item['groups'][g] = dict(mean_gpu_ms=statistics.mean(t['gpu_ms'] for t in totals),
                count_per_stage=statistics.mean(t['count'] for t in totals),
                pct_of_gpu_kernel_time=100*sum(t['gpu_ms'] for t in totals)/sum(r['kernel_sum_ms'] for r in rows))
        item['api_groups'] = {g: {key: statistics.mean(r['api_groups'].get(g, {}).get(key, 0) for r in rows) for key in ('count', 'cpu_ms')} for g in sorted({g for r in rows for g in r['api_groups']})}
        item['children_ms'] = {g: statistics.mean(r['children_ms'].get(g, 0) for r in rows) for g in sorted({g for r in rows for g in r['children_ms']})}
        summary[case][stage] = item
args.output.write_text(json.dumps({'summary': summary, 'stages': records}, indent=2)+'\n')
for case, stages in summary.items():
    for stage, s in stages.items():
        print(case, stage, 'wall', round(s['wall_ms']['mean'],3), 'GPU busy%', round(s['gpu_busy_pct'],1), 'kernels', s['kernels']['mean'])
        print({k: round(v['mean_gpu_ms'],3) for k,v in s['groups'].items()})
