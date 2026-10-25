import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics

root = Path.cwd()
raw = root/'artifacts/benchmark-gb10-flash-20261024'
out = root/'benchmarks/gb10-flash-2026-10-24'
out.mkdir(parents=True, exist_ok=True)
def read(p): return json.loads(p.read_text())
def write(p, obj): p.write_text(json.dumps(obj, indent=2)+'\n')
summary, env, runs = (read(raw/name) for name in ('summary.json','environment.json','runs.json'))
flash = read(root/'artifacts/three-path-flash/summary.json')
base = read(root/'artifacts/three-path-fused/summary.json')
attention = read(root/'artifacts/aoti-flash/attention-validation.json')
smoke = read(root/'artifacts/native-smoke-flash/summary.json')
regression = read(root/'artifacts/three-path-fused-current-native/regression.json')
previous_env = read(root/'artifacts/benchmark-gb10-full-rerun-20261023T150003Z/environment.json')
assert all(env['packages']['aoti_fused'][kind]['sha256'] == previous_env['packages']['aoti_fused'][kind]['sha256']
           for kind in ('decode','prefill'))
profile_dir = root/'artifacts/profile-flash-gb10-20261024'
profile = read(profile_dir/'profile-summary.json')
kernel_rows = list(csv.DictReader((profile_dir/'stats_cuda_gpu_kern_sum.csv').read_text().lstrip().splitlines()))
assert sum(int(r['Instances']) for r in kernel_rows) == sum(s['kernels'] for s in profile['stages'])
assert all('kv_head_expansion' not in s['groups'] for s in profile['stages'])
write(out/'profile-summary.json', {'summary':profile['summary']})
for name in ('workload.json','profiled.json','stats_cuda_gpu_kern_sum.csv','capture.log'):
    shutil.copy2(profile_dir/name,out/('profile-'+name))

for name in ('summary.json','environment.json','runs.json','workload.json'):
    shutil.copy2(raw/name,out/name)
for p in raw.glob('round-*'):
    if p.suffix in ('.json','.csv','.log'): shutil.copy2(p,out/p.name)
for src,name in (
    ('artifacts/three-path-flash/summary.json','validation-flash.json'),
    ('artifacts/three-path-flash/native.json','validation-native.json'),
    ('artifacts/three-path-flash/cases.json','validation-cases.json'),
    ('artifacts/three-path-fused/summary.json','validation-baseline.json'),
    ('artifacts/three-path-fused-current-native/regression.json','baseline-regression.json'),
    ('artifacts/native-smoke-flash/summary.json','native-smoke.json'),
    ('artifacts/aoti-flash/attention-validation.json','attention-validation.json'),
    ('artifacts/aoti-flash/generated-code.json','generated-code.json'),
    ('artifacts/aoti-flash/generated-code-excerpts.txt','generated-code-excerpts.txt'),
    ('artifacts/aoti-flash/export-decode.log','export-decode.log'),
    ('artifacts/aoti-flash/export-prefill.log','export-prefill.log'),
    ('artifacts/aoti-flash/validate.log','validation.log'),
    ('artifacts/aoti-flash/smoke.log','native-smoke.log'),
): shutil.copy2(root/src,out/name)

checks = sorted(raw.glob('*.idle.json')) + sorted(raw.glob('*.after-idle.json'))
assert len(checks)==16
assert all(len(read(p))>=5 and all(s['gpu_percent']<=10 and s['memory_percent']<=10 for s in read(p)[-5:]) for p in checks)
clock = {}
for label in ('aoti_fused','aoti_flash'):
    samples=[]
    for path in raw.glob(f'round-*-{label}.gpu.csv'):
        with path.open() as f:
            rows=list(csv.reader(f))[1:]
        for row in rows:
            try:
                util=float(row[1].strip().split()[0]); sm=float(row[5].strip().split()[0])
                if util>=50: samples.append(sm)
            except (ValueError, IndexError): pass
    assert samples
    clock[label]=dict(active_samples=len(samples),sm_mhz_min=min(samples),sm_mhz_median=statistics.median(samples),sm_mhz_max=max(samples))

paired={}
for case in ('length_25','length_128','length_512'):
    ratios={key:[] for key in ('prefill_ms','decode_ms_per_token','request_ms')}
    for round_index in range(1,5):
        r={label:next(c for c in read(raw/f'round-{round_index}-{label}.json')['cases'] if c['name']==case)
           for label in ('aoti_fused','aoti_flash')}
        for key in ratios:
            ratios[key].append(statistics.median(s[key] for s in r['aoti_fused']['samples']) /
                               statistics.median(s[key] for s in r['aoti_flash']['samples']))
    paired[case]={key:dict(by_round=v,min=min(v),max=max(v),median=statistics.median(v)) for key,v in ratios.items()}
checks_report=dict(idle_checks_passed=16,measured_requests=120,clock=clock,paired_speedups=paired,
                   baseline_regression_bitwise=regression['bitwise_equal'],baseline_package_hashes_unchanged=True,
                   profile_all_kernels_accounted=True,profile_kernels=sum(s['kernels'] for s in profile['stages']))
write(out/'analysis.json',checks_report)

def worst(report, label, key):
    return max(c[label][key] for c in report['cases'])
def cache_worst(report,key):
    return max(c[key] for c in next(c for c in report['cases'] if c['name']=='context_end')['cache_aoti_vs_hf_static'])
def fp32_worst(report,key):
    return max(c['aoti_vs_fp32'][key] for c in report['full_cache_fp32_control'])

table=[]; detail=[]
for length in (25,128,512):
    name=f'length_{length}'
    a,b=(summary['profiles'][p]['cases'][name] for p in ('aoti_fused','aoti_flash'))
    table.append(f"| {length} | {a['prefill_ms']['median']:.3f} | {b['prefill_ms']['median']:.3f} | {a['prefill_ms']['median']/b['prefill_ms']['median']:.2f}× | {a['decode_ms_per_token']['median']:.3f} | {b['decode_ms_per_token']['median']:.3f} | {a['decode_ms_per_token']['median']/b['decode_ms_per_token']['median']:.2f}× | {a['decode_tokens_per_second']:.1f} → {b['decode_tokens_per_second']:.1f} |")
    for label in ('aoti_fused','aoti_flash'):
        m=summary['profiles'][label]['cases'][name]
        detail.append(f"| {length} | {label} | {m['prefill_ms']['p10']:.3f}–{m['prefill_ms']['p90']:.3f} | {m['decode_ms_per_token']['p10']:.3f}–{m['decode_ms_per_token']['p90']:.3f} | {m['request_ms']['median']:.2f} | {m['memory']['peak_allocated_bytes']/2**20:.1f} |")
predictions={name:summary['profiles']['aoti_flash']['cases'][name]['prediction_agreement_with_aoti_fused'] for name in paired}
compile_time={kind:env['packages']['aoti_flash'][kind]['metadata']['export_and_compile_seconds'] for kind in ('decode','prefill')}
sizes={kind:env['packages']['aoti_flash'][kind]['bytes'] for kind in ('decode','prefill')}
numerical=[]
for label,r in (('Fused baseline',base),('Fused + FlashAttention',flash)):
    numerical.append(f"| {label} | {worst(r,'aoti_vs_hf_static','max_relative_l2')*100:.3f}% | {cache_worst(r,'max_relative_l2')*100:.3f}% | {cache_worst(r,'max_peak_relative')*100:.3f}% | {len(r['numerical_failures'])}/56 | {fp32_worst(r,'max_relative_l2')*100:.3f}% | {fp32_worst(r,'max_peak_relative')*100:.3f}% |")
decode_profile=profile['summary']['length_128']['decode']
linear_ms=sum(decode_profile['groups'][g]['mean_gpu_ms'] for g in ('linear_gemm','linear_gemv','lm_head'))

text=f'''# GB10: fused AOTI versus native FlashAttention

Date: 2026-10-24 (Asia/Shanghai). Implementation commit: `{env['git_commit']}` on `dev-gb10`.
Host: `tqxia@192.168.1.36`, repository `/home/tqxia/workspace/xinfer`.

The optional FlashAttention path uses native GQA and attends only to the valid KV
prefix. The original fused packages are unchanged. This comparison uses no CUDA
Graph replay. **Functional checks pass, but numerical acceptance remains failed:**
17/56 full-context cache tensors exceed the unchanged 5% limits, versus 9/56 for
the original fused baseline. The new backend remains experimental and is not the default.

## End-to-end native performance

Medians of 20 measured requests per length/profile. Prefill and decode include
GPU argmax and the host token read. Decode is the per-token average across 64
forced steps. These are runtime measurements, not isolated attention speedups.

| Prompt tokens | Fused prefill ms | Flash prefill ms | Prefill speedup | Fused decode ms/token | Flash decode ms/token | Decode speedup | Decode tokens/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

| Prompt tokens | Profile | Prefill p10–p90 ms | Decode p10–p90 ms/token | Median request ms, including reset | Peak Torch allocated MiB |
|---:|---|---:|---:|---:|---:|
{chr(10).join(detail)}

Four fresh-process rounds alternate baseline/candidate order. Each worker runs
one first-use request, two warmups, and five measured requests at each length.
There are 120 measured requests in total. All 16 before/after idle checks passed;
no `--allow-busy` override was used. Both profiles use the same rebuilt binary,
one CPU thread, checkpoint, tokens, BF16 weights/cache, fusion settings and context
capacity. No weights, dependency versions, power settings or clocks were changed.

Active telemetry (GPU utilization at least 50%): baseline median SM clock
{clock['aoti_fused']['sm_mhz_median']:.0f} MHz (range {clock['aoti_fused']['sm_mhz_min']:.0f}–{clock['aoti_fused']['sm_mhz_max']:.0f}),
Flash median {clock['aoti_flash']['sm_mhz_median']:.0f} MHz
(range {clock['aoti_flash']['sm_mhz_min']:.0f}–{clock['aoti_flash']['sm_mhz_max']:.0f}).
These are automatic-clock results. Per-round speedups and telemetry are in
`analysis.json` and the individual worker files. Candidate/base prediction agreement
is {predictions['length_25']}/65, {predictions['length_128']}/65, and {predictions['length_512']}/65
at lengths 25, 128 and 512, respectively; every repeated request is deterministic.
Torch allocator peaks exclude some AOTI weight allocations; device-memory samples
in `summary.json` are sampled process deltas, not exact transient peaks.

## Correctness and numerical limits

- Strict `AOTI_RUNTIME_CHECK_INPUTS=1` validation: 26 cases and 185 logit rows.
- All 185 top-1 predictions and fixture greedy sequences agree with HF BF16.
- Native/Python AOTI logits and all 56 full-context cache tensors agree bitwise.
- Cache storage, write regions, reset/reuse, poisoned future slots, one-token
  prompts, lengths 2/7/128/511/512, offset chunked prefill, final legal slot and
  overflow rejection pass. Extra chunk boundaries reach valid lengths
  1/3/16/32/63/64/128/256/512/1023/1024, with and without poisoned future storage.
- Native CLI smoke tests pass, including invalid inputs, mixed-package rejection,
  generation limits and Python-free linkage/loading.
- The rebuilt runtime reproduces all 25 recorded baseline logit/cache files
  bitwise (24 cases, 163 rows), independently confirming baseline compatibility.
- Nine isolated attention shapes match an explicit FP32 masked/GQA reference
  within the existing 5% limits, and are bitwise invariant to future-slot poison.

| Profile | Worst logits L2 vs HF BF16 | Full-cache L2 vs HF BF16 | Full-cache peak-relative vs HF BF16 | Failing cache tensors | Full-cache L2 vs FP32 | Full-cache peak-relative vs FP32 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(numerical)}

The original 5% criteria are unchanged. `validate_cpp.py` intentionally exits 1
after recording the complete numerical report. Eager FlashAttention also shows
the large cache difference versus eager efficient attention: worst relative L2
{max(r['max_relative_l2'] for r in attention['cache_comparisons']['flash_vs_efficient'])*100:.3f}% and
peak-relative error {max(r['max_peak_relative'] for r in attention['cache_comparisons']['flash_vs_efficient'])*100:.3f}%.
Thus the large difference is present without AOTI. The isolated FP32 test supports
correct causality/GQA semantics. The full-model FP32 control has mixed results:
Flash has a lower worst peak-relative error but a higher worst relative L2 error.
Neither bitwise equivalence nor an overall quality improvement is claimed.
This finite fixture suite is not a model-quality evaluation.

## Implementation

- `compiler/model.py` registers a small project-owned HF attention adapter, retaining
  Qwen's projections, Q/K normalization, RoPE and output projection. It invokes
  `aten._scaled_dot_product_flash_attention`, the same native operator used by
  PyTorch's lower-right causal-bias dispatcher. Its lower-right alignment handles
  initial prefill, offset chunked prefill and single-token decode uniformly.
- K/V retain eight heads. There is no 8→16 materialized head expansion and no dense
  attention mask. Generated code contains 28 FlashAttention calls per forward and
  zero efficient-attention calls. The calls receive cache views directly;
  generated wrappers have no cache-copy API calls. See `generated-code-excerpts.txt`.
- The native runtime retains 56 caller-owned logical `[1,8,1024,128]` buffers.
  One spare storage position per head keeps prefix strides uniform at capacity
  (98 KiB extra in the native allocation; 112 KiB in Python's export allocation).
  A persistent 1,025-byte CPU buffer provides a view of
  length `valid_length+1`; only its shape is read. The compiled model updates the
  fixed-size caches and constructs the valid-prefix views internally.
- Fixed mutation-input shapes plus the extent tensor avoid AOTI's size-one
  specialization. PyTorch 2.13's experimental `backed_size_oblivious` export mode
  handles intermediate size-one views. Strict input checks remain enabled during
  validation; no guards or error tolerances are disabled. No extra package is needed.
- Metadata enforces matching attention, cache layout, shape contract, numerical
  mode and Torch ABI. Older full-cache packages remain supported.

The direct native attention operator and symbolic export option are internal or
experimental PyTorch interfaces; the exact Torch 2.13/CUDA 13 environment is pinned
and packages must be regenerated and validated after toolchain changes.
Final decode/prefill export+compile times were {compile_time['decode']:.1f}/{compile_time['prefill']:.1f} seconds;
package sizes are {sizes['decode']:,}/{sizes['prefill']:,} bytes. Earlier guarded
prefix-view probes are preserved separately under `artifacts/*probe*`; they are
not the packages benchmarked here.

## Kernel verification and remaining bottleneck

A separate short Nsight capture after the timed benchmark covers three prefills
and 24 decode steps, following two warmup requests per length. All
{checks_report['profile_kernels']:,} GPU kernels are accounted for by the NVTX ranges.
The trace contains `pytorch_flash::flash_fwd_kernel`,
`pytorch_flash::flash_fwd_splitkv_kernel`, and split-combine variants, with no
KV-head-expansion kernels. See `profile-stats_cuda_gpu_kern_sum.csv` for exact names.
Decode launches 397 kernels/token at length 25 and 425 at lengths 128/512;
the latter use an additional split-combine kernel per layer.

At prompt length 128, attention takes {decode_profile['groups']['attention']['mean_gpu_ms']:.3f} ms/token
in the short trace. Linear projections plus the vocabulary head take {linear_ms:.3f} ms/token,
or {100*linear_ms/decode_profile['kernel_sum_ms']['mean']:.1f}% of GPU kernel time. The vocabulary head alone takes
{decode_profile['groups']['lm_head']['mean_gpu_ms']:.3f} ms/token. Linear/weight-read performance remains the main next
opportunity. These trace values are diagnostic and are not substituted for the
unprofiled benchmark results above; no DRAM hardware counters were collected.

## Reproduce

Run from `/home/tqxia/workspace/xinfer`. Choose fresh artifact directories when rerunning.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python compiler/export.py decode --attention-backend flash --output artifacts/aoti-flash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python compiler/export.py prefill --attention-backend flash --output artifacts/aoti-flash
./scripts/build-gb10.sh
AOTI_RUNTIME_CHECK_INPUTS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python compiler/validate_cpp.py --packages artifacts/aoti-flash --output artifacts/three-path-flash
# The validator records the known numerical failure; inspect summary.json.
PYTHONPATH=compiler OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python compiler/tests/flash_attention.py --validation artifacts/three-path-flash --output artifacts/aoti-flash/attention-validation.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python compiler/benchmark.py compare --baseline-packages artifacts/aoti --candidate-packages artifacts/aoti-flash --output artifacts/benchmark-gb10-flash-20261024
```

`environment.json` records source/binary/package hashes, package metadata and the
model revision. Raw logits, cache dumps and source snapshots remain under
`artifacts/three-path-flash`, `artifacts/three-path-fused-current-native`,
`artifacts/native-smoke-flash` and `artifacts/benchmark-gb10-flash-20261024`.
This directory preserves compact results and supporting evidence.
'''
(out/'REPORT.md').write_text(text)
if Path(__file__).resolve() != (out/'analyze.py').resolve():
    shutil.copy2(Path(__file__),out/'analyze.py')
print('\n'.join(table))
print(json.dumps(checks_report,indent=2))
