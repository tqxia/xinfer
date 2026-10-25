"""Run from the project root after export/validation and the C++ build."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

binary = Path('build/qwen_aoti').resolve()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--packages', type=Path, default=Path('artifacts/aoti'))
parser.add_argument('--output', type=Path, default=Path('artifacts/native-smoke'))
args = parser.parse_args()
packages, out = args.packages, args.output
out.mkdir(exist_ok=True)
env = dict(os.environ)
for key in ('VIRTUAL_ENV', 'PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH'):
    env.pop(key, None)
env['PATH'] = '/usr/bin:/bin'
env['PYTHONHOME'] = '/nonexistent-python-home'
report = {'fixtures': [], 'rejections': []}
linked = subprocess.check_output(['ldd', str(binary)], text=True)
(out/'ldd.txt').write_text(linked)
assert 'not found' not in linked
assert 'libpython' not in linked and 'libtorch_python' not in linked

for case in sorted(Path('artifacts/reference').iterdir()):
    if not case.is_dir(): continue
    run_env = env | ({'LD_DEBUG': 'libs'} if case.name == 'capital' else {})
    ids = out/f'{case.name}.ids'
    trace = out/f'{case.name}.f32'
    result = subprocess.run([str(binary), str(packages), str(case/'prompt.ids'), str(ids), '--trace', str(trace)],
                            env=run_env, capture_output=True, text=True, timeout=120)
    (out/f'{case.name}.log').write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    actual = [int(x) for x in ids.read_text().split()]
    expected = [int(x) for x in (case/'generated.ids').read_text().split()]
    assert actual == expected, (case.name, actual, expected)
    a = np.fromfile(trace, dtype='<f4')
    b = np.fromfile(packages/'reference'/f'{case.name}.f32', dtype='<f4')
    assert np.array_equal(a, b), (case.name, a.shape, b.shape, np.max(np.abs(a-b)))
    if case.name == 'capital':
        assert 'libpython' not in result.stderr and 'libtorch_python' not in result.stderr
    report['fixtures'].append(dict(name=case.name, tokens=len(actual), logits_bitwise_equal=True))
    print(report['fixtures'][-1], flush=True)

bad = out/'invalid.ids'
cases = [
    ('empty', '', [], 'Prompt must not be empty'),
    ('negative', '-1', [], 'Token ID out of range'),
    ('vocab', '151936', [], 'Token ID out of range'),
    ('fraction', '1.5', [], 'Invalid integer'),
    ('overflow', '999999999999999999999999999999', [], 'Invalid integer'),
    ('long_prompt', '1 '*513, [], 'Prompt exceeds'),
    ('zero_budget', '1', ['--max-new-tokens','0'], 'max-new-tokens must'),
    ('negative_budget', '1', ['--max-new-tokens','-1'], 'max-new-tokens must'),
    ('context_budget', '1 '*512, ['--max-new-tokens','513'], 'must fit the KV cache'),
    ('unknown_flag', '1', ['--unknown','1'], 'Unknown option'),
    ('missing_value', '1', ['--trace'], 'Missing option value'),
    ('trace_collision', '1', ['--trace',str(out/'invalid-output.ids')], 'Trace path must differ'),
]
for name, content, extra, message in cases:
    bad.write_text(content)
    result = subprocess.run([str(binary), str(packages), str(bad), str(out/'invalid-output.ids'), *extra],
                            env=env | {'CUDA_VISIBLE_DEVICES':''}, capture_output=True, text=True, timeout=20)
    assert result.returncode != 0 and message in result.stderr, (name, result.stderr)
    report['rejections'].append(name)

# Reject mixed compilation profiles from metadata before touching CUDA or weights.
mixed = out/'mixed-packages'
mixed.mkdir(exist_ok=True)
for kind in ('prefill', 'decode'):
    metadata = json.loads((packages/(kind+'.json')).read_text())
    metadata['eager_numerics'] = kind == 'prefill'
    (mixed/(kind+'.json')).write_text(json.dumps(metadata))
result = subprocess.run([str(binary), str(mixed), 'artifacts/reference/capital/prompt.ids',
                         str(out/'mixed.ids')], env=env | {'CUDA_VISIBLE_DEVICES':''},
                        capture_output=True, text=True, timeout=20)
assert result.returncode != 0 and 'Package numerical settings differ' in result.stderr, result.stderr
report['rejections'].append('mixed_numerical_settings')

# Attention/cache contracts must also agree before CUDA initialization.
for kind in ('prefill', 'decode'):
    metadata = json.loads((packages/(kind+'.json')).read_text())
    if kind == 'prefill':
        metadata['attention_backend'] = ('efficient' if metadata.get('attention_backend') == 'flash' else 'flash')
    (mixed/(kind+'.json')).write_text(json.dumps(metadata))
result = subprocess.run([str(binary), str(mixed), 'artifacts/reference/capital/prompt.ids',
                         str(out/'mixed.ids')], env=env | {'CUDA_VISIBLE_DEVICES':''},
                        capture_output=True, text=True, timeout=20)
assert result.returncode != 0 and 'Package attention/cache settings differ' in result.stderr, result.stderr
report['rejections'].append('mixed_attention_settings')

# A real generation-limit stop must produce exactly one prediction, even before EOS.
result = subprocess.run([str(binary), str(packages), 'artifacts/reference/capital/prompt.ids',
                         str(out/'limit.ids'), '--max-new-tokens','1'], env=env,
                        capture_output=True, text=True, timeout=120)
assert result.returncode == 0, result.stderr
assert (out/'limit.ids').read_text().split() == ['785']

# Exercise native T=1 routing and the endpoints of dynamic prefill, comparing
# against Python AOTI under exactly the same fed tokens. Python is only the oracle.
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from custom_ops.attention import load_package_attention_ops

spec = json.loads((packages/'decode.json').read_text())
load_package_attention_ops(spec)
if spec.get('attention_backend') in ('flash', 'flash-custom', 'flashinfer-decode'):
    # Old direct-ATen packages reused the flash label but have no custom-op ABI.
    # Reject them before touching CUDA or attempting to load package weights.
    message = 're-export with --attention-backend flash'
    for kind in ('prefill', 'decode'):
        metadata = json.loads((packages/(kind+'.json')).read_text())
        metadata['attention_backend'] = 'flash'
        del metadata['custom_attention']
        (mixed/(kind+'.json')).write_text(json.dumps(metadata))
        try:
            load_package_attention_ops(metadata)
        except ValueError as error:
            assert message in str(error), str(error)
        else:
            raise AssertionError('Python accepted an obsolete direct-Flash package')
    result = subprocess.run([str(binary),str(mixed),'artifacts/reference/capital/prompt.ids',str(out/'mixed.ids')],
                            env=env | {'CUDA_VISIBLE_DEVICES':''},capture_output=True,text=True,timeout=20)
    assert result.returncode != 0 and message in result.stderr, result.stderr
    report['rejections'].append('obsolete_direct_flash_package')
    report['python_direct_flash_rejection'] = True
    for kind in ('prefill', 'decode'):
        metadata = json.loads((packages/(kind+'.json')).read_text())
        metadata['custom_attention']['abi_version'] = 2
        (mixed/(kind+'.json')).write_text(json.dumps(metadata))
    result = subprocess.run([str(binary),str(mixed),'artifacts/reference/capital/prompt.ids',str(out/'mixed.ids')],
                            env=env | {'CUDA_VISIBLE_DEVICES':''},capture_output=True,text=True,timeout=20)
    assert result.returncode != 0 and 'Unsupported custom attention contract' in result.stderr, result.stderr
    report['rejections'].append('unsupported_custom_attention_contract')
    for name, message in (
            ('mixed_cache_padding', 'Package cache padding settings differ'),
            ('unsupported_cache_padding', 'Unsupported cache padding'),
            ('invalid_cache_stride', 'Unexpected KV cache strides'),
            ('mixed_export_settings', 'Package export settings differ')):
        for kind in ('prefill', 'decode'):
            metadata = json.loads((packages/(kind+'.json')).read_text())
            if name == 'mixed_cache_padding':
                padding = metadata.get('cache_padding', 1)
                metadata['cache_padding'] = 1-padding if kind == 'prefill' else padding
            elif name == 'unsupported_cache_padding':
                metadata['cache_padding'] = 2
            elif name == 'invalid_cache_stride':
                metadata['cache_stride'][1] += 128
            elif kind == 'prefill':
                metadata['export_options']['backed_size_oblivious'] = not metadata['export_options']['backed_size_oblivious']
            (mixed/(kind+'.json')).write_text(json.dumps(metadata))
        result = subprocess.run([str(binary),str(mixed),'artifacts/reference/capital/prompt.ids',str(out/'mixed.ids')],
                                env=env | {'CUDA_VISIBLE_DEVICES':''},capture_output=True,text=True,timeout=20)
        assert result.returncode != 0 and message in result.stderr, (name,result.stderr)
        report['rejections'].append(name)
if spec.get('attention_backend') == 'flashinfer-decode':
    for name, message in (('wrong_prefill_operator', 'Unsupported custom attention contract'),
                          ('wrong_flashinfer_revision', 'Unsupported FlashInfer source contract')):
        for kind in ('prefill', 'decode'):
            metadata = json.loads((packages/(kind+'.json')).read_text())
            if name == 'wrong_prefill_operator' and kind == 'prefill':
                metadata['custom_attention']['name'] = 'xinfer::flashinfer_decode'
            elif name == 'wrong_flashinfer_revision':
                metadata['flashinfer_source']['revision'] = 'unrecognized'
            (mixed/(kind+'.json')).write_text(json.dumps(metadata))
            if kind == 'prefill':
                try:
                    load_package_attention_ops(metadata)
                except ValueError as error:
                    assert message in str(error), str(error)
                else:
                    raise AssertionError(name)
        result = subprocess.run([str(binary),str(mixed),'artifacts/reference/capital/prompt.ids',str(out/'mixed.ids')],
                                env=env | {'CUDA_VISIBLE_DEVICES':''},capture_output=True,text=True,timeout=20)
        assert result.returncode != 0 and message in result.stderr, (name,result.stderr)
        report['rejections'].append(name)
prefill = torch._inductor.aoti_load_package(str(packages/'prefill.pt2'))
decode = torch._inductor.aoti_load_package(str(packages/'decode.pt2'))
seed = [int(x) for x in Path('artifacts/reference/capital/prompt.ids').read_text().split()]
report['boundaries'] = []
with torch.inference_mode():
    for length in (1, 2, 512):
        prompt = (seed * (length//len(seed)+1))[:length]
        input_path, ids_path, trace_path = (out/f'length-{length}.{ext}' for ext in ('input','ids','f32'))
        input_path.write_text(' '.join(map(str,prompt)))
        result = subprocess.run([str(binary),str(packages),str(input_path),str(ids_path),
                                 '--max-new-tokens','3','--trace',str(trace_path)],
                                env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        generated = [int(x) for x in ids_path.read_text().split()]
        assert 1 <= len(generated) <= 3
        rows = np.fromfile(trace_path,dtype='<f4').reshape(len(generated),spec['vocab_size'])
        cache = tuple(torch.empty_strided(spec['cache_shape'],spec['cache_stride'],dtype=torch.bfloat16,device='cuda').zero_()
                      for _ in range(2*spec['layers']))
        tokens, position = prompt, 0
        for token, row in zip(generated,rows):
            ids = torch.tensor([tokens],dtype=torch.int64,device='cuda')
            positions = torch.arange(position,position+len(tokens),device='cuda')
            extra = (torch.empty(position+len(tokens)+1,dtype=torch.uint8),) if spec.get('cache_input') == 'full_capacity_with_extent' else ()
            scores = (decode if len(tokens)==1 else prefill)(ids,positions,cache,*extra)
            assert scores.argmax(-1).item() == token
            assert np.array_equal(scores.float().cpu().numpy()[0],row)
            position += len(tokens)
            tokens = [token]
        report['boundaries'].append(dict(prompt_length=length,tokens=len(generated),logits_bitwise_equal=True))
        print(report['boundaries'][-1],flush=True)

report.update(generation_limit_pass=True, no_python_linkage=True,
              native_loader_no_python_library=True, sanitized_environment_pass=True)
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS: native generation, exact AOTI logits, invalid inputs, generation limit, Python-free linkage', flush=True)
