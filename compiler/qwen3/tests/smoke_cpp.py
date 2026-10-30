"""Validate the native Flash CLI, deployment contract and exact Python AOTI parity."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from compiler.qwen3.custom_ops.attention import load_package_attention_ops
from compiler.qwen3.model import cache_extent
from compiler.qwen3.tests.numerics import ids_from


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native', type=Path, default=Path('build/qwen_aoti'))
parser.add_argument('--reference', type=Path, default=Path('artifacts/reference'))
parser.add_argument('--packages', type=Path, default=Path('artifacts/aoti'))
parser.add_argument('--output', type=Path, default=Path('artifacts/native-smoke'))
args = parser.parse_args()
binary, packages, out = args.native.resolve(), args.packages, args.output
if out.exists():
    parser.error('Output directory already exists; choose a fresh --output')
out.mkdir(parents=True)
env = {key: value for key, value in os.environ.items()
       if key not in ('VIRTUAL_ENV', 'PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH')}
env.update(PATH='/usr/bin:/bin', PYTHONHOME='/nonexistent-python-home')
report = dict(fixtures=[], rejections=[], boundaries=[], native_processes=[])
linked = subprocess.check_output(['ldd', str(binary)], text=True)
(out / 'ldd.txt').write_text(linked)
assert 'not found' not in linked
assert 'libpython' not in linked and 'libtorch_python' not in linked
spec = json.loads((packages / 'decode.json').read_text())
load_package_attention_ops(spec, native=binary)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
native_traces = []


def run_native(arguments, name, environment=env, timeout=120):
    result = subprocess.run([str(binary), *map(str, arguments)], env=environment,
                            capture_output=True, text=True, timeout=timeout)
    (out / f'{name}.log').write_text(result.stderr)
    (out / f'{name}.stdout.log').write_text(result.stdout)
    report['native_processes'].append(dict(name=name, returncode=result.returncode))
    return result


def failure_detail(result, name):
    return dict(case=name, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


@torch.inference_mode()
def compare_native_trace(prompt, generated, trace_path, budget):
    assert 1 <= len(generated) <= budget
    assert not any(token in spec['eos_token_ids'] for token in generated[:-1])
    assert len(generated) == budget or generated[-1] in spec['eos_token_ids']
    rows = np.fromfile(trace_path, dtype='<f4').reshape(len(generated), spec['vocab_size'])
    assert np.isfinite(rows).all()
    cache = tuple(torch.zeros(spec['cache_shape'], dtype=torch.bfloat16, device='cuda')
                  for _ in range(2 * spec['layers']))
    tokens, position = prompt, 0
    for token, row in zip(generated, rows):
        ids = torch.tensor([tokens], dtype=torch.int64, device='cuda')
        positions = torch.arange(position, position + len(tokens), device='cuda')
        scores = (decode if len(tokens) == 1 else prefill)(
            ids, positions, cache, cache_extent(position + len(tokens)))
        assert torch.isfinite(scores).all().item()
        assert scores.argmax(-1).item() == token
        assert scores.float().cpu().numpy()[0].tobytes() == row.tobytes()
        position += len(tokens)
        tokens = [token]
    assert all(torch.isfinite(tensor).all().item() for tensor in cache)


for case in sorted(args.reference.iterdir()):
    if not case.is_dir():
        continue
    run_env = env | ({'LD_DEBUG': 'libs'} if case.name == 'capital' else {})
    ids, trace = out / f'{case.name}.ids', out / f'{case.name}.f32'
    meta = json.loads((case / 'metadata.json').read_text())
    budget = meta['max_new_tokens']
    result = run_native([packages, case / 'prompt.ids', ids, '--max-new-tokens', budget,
                         '--trace', trace], case.name, environment=run_env)
    assert result.returncode == 0, failure_detail(result, case.name)
    actual, expected = ids_from(ids), ids_from(case / 'generated.ids')
    if case.name == 'capital':
        assert 'libpython' not in result.stderr and 'libtorch_python' not in result.stderr
    entry = dict(name=case.name, tokens=len(actual), generated_ids=actual,
                 hf_sequence_matches=actual == expected)
    report['fixtures'].append(entry)
    native_traces.append((entry, ids_from(case / 'prompt.ids'), actual, trace, budget))
    print(report['fixtures'][-1], flush=True)

bad = out / 'invalid.ids'
cases = [
    ('empty', '', [], 'Prompt must not be empty'),
    ('negative', '-1', [], 'Token ID out of range'),
    ('vocab', '151936', [], 'Token ID out of range'),
    ('fraction', '1.5', [], 'Invalid integer'),
    ('overflow', '999999999999999999999999999999', [], 'Invalid integer'),
    ('long_prompt', '1 ' * 513, [], 'Prompt exceeds'),
    ('zero_budget', '1', ['--max-new-tokens', '0'], 'max-new-tokens must'),
    ('negative_budget', '1', ['--max-new-tokens', '-1'], 'max-new-tokens must'),
    ('context_budget', '1 ' * 512, ['--max-new-tokens', '513'], 'must fit the KV cache'),
    ('unknown_flag', '1', ['--unknown', '1'], 'Unknown option'),
    ('missing_value', '1', ['--trace'], 'Missing option value'),
    ('trace_collision', '1', ['--trace', str(out / 'invalid-output.ids')], 'Trace path must differ'),
]
for name, content, extra, message in cases:
    bad.write_text(content)
    result = run_native([packages, bad, out / 'invalid-output.ids', *extra], name,
                        environment=env | {'CUDA_VISIBLE_DEVICES': ''}, timeout=20)
    assert result.returncode != 0 and message in result.stderr, failure_detail(result, name)
    report['rejections'].append(name)

# All invalid package contracts must fail before CUDA or weight loading.
mixed = out / 'invalid-packages'
mixed.mkdir()
package_cases = (
    ('mixed_numerical_settings', 'Package numerical settings differ'),
    ('legacy_eager_numerics', 'Legacy eager-numerics packages are unsupported'),
    ('legacy_efficient', 'Legacy efficient packages are unsupported'),
    ('mixed_attention_settings', 'Package attention/cache settings differ'),
    ('missing_custom_attention_contract', 'Flash packages require the custom-op contract'),
    ('unsupported_custom_attention_contract', 'Unsupported custom attention contract'),
    ('mixed_cache_padding', 'Package cache padding settings differ'),
    ('unsupported_cache_padding', 'Unsupported cache padding'),
    ('invalid_cache_stride', 'Unexpected KV cache strides'),
    ('mixed_export_settings', 'Package export settings differ'),
)
for name, message in package_cases:
    for kind in ('prefill', 'decode'):
        metadata = json.loads((packages / (kind + '.json')).read_text())
        if name == 'mixed_numerical_settings' and kind == 'prefill':
            metadata['triton_enable_fp_fusion'] = not metadata['triton_enable_fp_fusion']
        elif name == 'legacy_eager_numerics':
            metadata['eager_numerics'] = True
        elif name == 'legacy_efficient':
            metadata.update(attention_backend='efficient', cache_input='full_capacity')
        elif name == 'mixed_attention_settings' and kind == 'prefill':
            metadata['attention_backend'] = 'efficient'
        elif name == 'missing_custom_attention_contract':
            del metadata['custom_attention']
        elif name == 'unsupported_custom_attention_contract':
            metadata['custom_attention']['abi_version'] = 2
        elif name == 'mixed_cache_padding' and kind == 'prefill':
            metadata['cache_padding'] = 1
        elif name == 'unsupported_cache_padding':
            metadata['cache_padding'] = 2
        elif name == 'invalid_cache_stride':
            metadata['cache_stride'][1] += 128
        elif name == 'mixed_export_settings' and kind == 'prefill':
            metadata['export_options']['backed_size_oblivious'] = True
        (mixed / (kind + '.json')).write_text(json.dumps(metadata))
        if name in ('legacy_eager_numerics', 'legacy_efficient', 'missing_custom_attention_contract',
                    'unsupported_custom_attention_contract'):
            try:
                load_package_attention_ops(metadata)
            except ValueError as error:
                assert message in str(error), (name, str(error))
            else:
                raise AssertionError(f'Python accepted invalid package: {name}')
    result = run_native([mixed, args.reference / 'capital/prompt.ids', out / 'mixed.ids'], name,
                        environment=env | {'CUDA_VISIBLE_DEVICES': ''}, timeout=20)
    assert result.returncode != 0 and message in result.stderr, failure_detail(result, name)
    report['rejections'].append(name)

seed = ids_from(args.reference / 'capital' / 'prompt.ids')
for length, budget in ((1, 3), (2, 3), (512, 3), (len(seed), 1)):
    prompt = (seed * (length // len(seed) + 1))[:length]
    stem = f'length-{length}-budget-{budget}'
    input_path, ids_path, trace_path = (out / f'{stem}.{ext}' for ext in ('input', 'ids', 'f32'))
    input_path.write_text(' '.join(map(str, prompt)))
    result = run_native([packages, input_path, ids_path, '--max-new-tokens', budget,
                         '--trace', trace_path], stem)
    assert result.returncode == 0, failure_detail(result, stem)
    generated = ids_from(ids_path)
    entry = dict(prompt_length=length, budget=budget, tokens=len(generated))
    report['boundaries'].append(entry)
    native_traces.append((entry, prompt, generated, trace_path, budget))
    print(report['boundaries'][-1], flush=True)

# Finish every native process before loading the Python packages. Each pair of
# loaders owns model weights; retaining both processes' pairs at once needlessly
# increases deployment-test memory pressure.
prefill = torch._inductor.aoti_load_package(str(packages / 'prefill.pt2'))
decode = torch._inductor.aoti_load_package(str(packages / 'decode.pt2'))
for entry, prompt, generated, trace_path, budget in native_traces:
    compare_native_trace(prompt, generated, trace_path, budget)
    entry['logits_bitwise_equal'] = True

report.update(passed=True, generation_limit_pass=True, no_python_linkage=True,
              native_loader_no_python_library=True, sanitized_environment_pass=True)
(out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
print('PASS: native generation/AOTI parity, package/input rejection, limits and Python-free linkage', flush=True)
