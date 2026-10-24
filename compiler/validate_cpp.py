"""Compare HF eager, Python AOTI, and C++ with identical inputs and reused caches."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch
from transformers import StaticCache

from model import CACHE_LENGTH, allocate_cache, load_model, cache_extent
from prepare import DEFAULT_MODEL
from validate_aoti import MAX_RELATIVE_ERROR, compare, error_metrics, ids_from


def trace(step, chunks):
    rows, position = [], 0
    for tokens in chunks:
        rows.append(step(tokens, position)[0].cpu())
        position += len(tokens)
    return torch.stack(rows)


def greedy(step, prompt, limit, eos):
    rows, ids, tokens, position = [], [], prompt, 0
    for _ in range(limit):
        score = step(tokens, position)
        rows.append(score[0].cpu())
        token = score.argmax(-1).item()
        ids.append(token)
        if token in eos:
            break
        position += len(tokens)
        tokens = [token]
    return torch.stack(rows), ids


def save_rows(path, rows):
    rows.numpy().astype('<f4').tofile(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--packages', type=Path, default=Path('artifacts/aoti'))
    parser.add_argument('--reference', type=Path, default=Path('artifacts/reference'))
    parser.add_argument('--native', type=Path, default=Path('build/qwen_validate'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/three-path'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output directory already exists; choose a fresh --output')
    if not args.native.is_file():
        parser.error('Build with -DQWEN_BUILD_VALIDATION=ON first')
    args.output.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = json.loads((args.packages/'decode.json').read_text())
    model = load_model(args.model_dir)
    hf_cache = StaticCache(model.config, 1, CACHE_LENGTH, device='cuda', dtype=torch.bfloat16)
    hf_tensors = tuple(t for pair in zip(hf_cache.key_cache, hf_cache.value_cache) for t in pair)

    def reset(tensors, fill=0):
        for tensor in tensors:
            tensor.fill_(fill)

    def hf_step(tokens, position):
        ids = torch.tensor([tokens], dtype=torch.int64, device='cuda')
        positions = torch.arange(position, position+len(tokens), device='cuda')
        return model(input_ids=ids, cache_position=positions, past_key_values=hf_cache,
                     use_cache=True, logits_to_keep=1).logits[:, -1].float()

    fixtures = {}
    for name in ('capital', 'arithmetic', 'chinese'):
        case = args.reference/name
        fixtures[name] = (ids_from(case/'prompt.ids'), ids_from(case/'generated.ids'))

    def greedy_case(name, fixture, limit=64):
        prompt, expected = fixtures[fixture]
        return dict(name=name, mode='greedy', prompt=prompt, limit=limit,
                    fixture=fixture, expected_ids=expected[:limit])

    cases = [greedy_case('capital_initial', 'capital')]
    for name, (prompt, continuation) in fixtures.items():
        cases.append(dict(name=name+'_teacher', mode='trace', fixture=name,
                          chunks=[prompt]+[[t] for t in continuation[:-1]]))
    seed = fixtures['capital'][0]
    for length in (1, 2, 7, 128, 511, 512):
        prompt = (seed*(length//len(seed)+1))[:length]
        reset(hf_tensors)
        _, continuation = greedy(hf_step, prompt, 4, [])
        chunks = [prompt]+[[t] for t in continuation[:-1]]
        name = f'length_{length}'
        cases.append(dict(name=name, mode='trace', chunks=chunks))
        cases.append(dict(name=name+'_poison', mode='trace', chunks=chunks, fill=3, same_as=name))
    # Feed deliberately different tokens, including EOS, without ending a trace.
    cases.append(dict(name='forced_tokens', mode='trace',
                      chunks=[seed, [0], [151645], [151643], [13]]))
    full = (seed*(CACHE_LENGTH//len(seed)+1))[:CACHE_LENGTH]
    cases.append(dict(name='context_end', mode='trace', dump_cache=True, check_overflow=True,
                      chunks=[full[:512], full[512:1022], full[1022:1023], full[1023:1024]]))
    cases.extend([
        greedy_case('capital_after_full', 'capital'),
        greedy_case('arithmetic_after_capital', 'arithmetic'),
        greedy_case('chinese_after_arithmetic', 'chinese'),
        greedy_case('capital_repeat', 'capital'),
        greedy_case('limit_one', 'capital', 1),
        greedy_case('limit_three', 'capital', 3),
    ])
    if spec.get('attention_backend') == 'flash':
        # Exercise shape boundaries and rectangular causal alignment, including
        # a multi-token append immediately after a one-token initial prompt.
        sizes = (1, 2, 13, 16, 31, 1, 64, 128, 256, 511, 1)
        chunks, offset = [], 0
        for size in sizes:
            chunks.append(full[offset:offset+size])
            offset += size
        assert offset == CACHE_LENGTH
        cases.extend([
            dict(name='prefix_boundaries', mode='trace', chunks=chunks, check_overflow=True),
            dict(name='prefix_boundaries_poison', mode='trace', chunks=chunks, fill=3,
                 same_as='prefix_boundaries', check_overflow=True),
        ])
    manifest = args.output/'cases.json'
    manifest.write_text(json.dumps({'cases': cases}, indent=2)+'\n')
    env = {k: v for k, v in os.environ.items()
           if k not in ('VIRTUAL_ENV', 'PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH')}
    env.update(PATH='/usr/bin:/bin', PYTHONHOME='/nonexistent-python-home')
    with (args.output/'native.log').open('w') as log:
        result = subprocess.run([str(args.native.resolve()), str(args.packages.resolve()),
                                 str(manifest.resolve()), str(args.output.resolve())],
                                env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    assert result.returncode == 0, (args.output/'native.log').read_text()
    native = json.loads((args.output/'native.json').read_text())
    assert len(native['cases']) == len(cases)
    print(f'Native driver passed {len(cases)} cases with one pair of loaders.', flush=True)

    prefill = torch._inductor.aoti_load_package(str(args.packages/'prefill.pt2'))
    decode = torch._inductor.aoti_load_package(str(args.packages/'decode.pt2'))
    py_cache = allocate_cache(model.config, spec.get('attention_backend', 'efficient'))
    pointers = [t.data_ptr() for t in py_cache]

    def py_step(tokens, position):
        ids = torch.tensor([tokens], dtype=torch.int64, device='cuda')
        positions = torch.arange(position, position+len(tokens), device='cuda')
        extra = (cache_extent(position + len(tokens)),) if spec.get('attention_backend') == 'flash' else ()
        return (decode if len(tokens) == 1 else prefill)(ids, positions, py_cache, *extra).float()

    report = dict(cases=[], error_limit=MAX_RELATIVE_ERROR, native_loader_pairs=native['loader_pairs'],
                  numerical_failures=[])
    full_cache = None
    for case, entry in zip(cases, native['cases']):
        name = case['name']
        assert entry['name'] == name and entry['stable_storage']
        reset(hf_tensors, case.get('fill', 0))
        reset(py_cache, case.get('fill', 0))
        if case['mode'] == 'trace':
            eager = trace(hf_step, case['chunks'])
            aoti = trace(py_step, case['chunks'])
            assert entry['write_region_checks'] and entry['reset_check']
        else:
            eager, eager_ids = greedy(hf_step, case['prompt'], case['limit'], spec['eos_token_ids'])
            aoti, aoti_ids = greedy(py_step, case['prompt'], case['limit'], spec['eos_token_ids'])
            assert entry['ids'] == aoti_ids == eager_ids == case['expected_ids'], name
        actual = torch.from_numpy(np.fromfile(args.output/(name+'.f32'), dtype='<f4')
                                  .reshape(entry['rows'], spec['vocab_size']))
        assert pointers == [t.data_ptr() for t in py_cache]
        metrics = dict(name=name, rows=entry['rows'], cpp_vs_aoti=compare(actual, aoti, exact=True),
                       aoti_vs_hf_static=compare(aoti, eager), cpp_vs_hf_static=compare(actual, eager))
        metrics['aoti_vs_hf_static_bitwise'] = torch.equal(aoti.contiguous().view(torch.uint8),
                                                        eager.contiguous().view(torch.uint8))
        different = torch.where(actual.argmax(-1) != eager.argmax(-1))[0].tolist()
        metrics['top1_differences'] = [dict(row=i, hf_id=eager[i].argmax().item(),
                                          compiled_id=actual[i].argmax().item(),
                                          hf_top2_margin=(eager[i].topk(2).values.diff().abs().item()))
                                     for i in different]
        assert not different, metrics['top1_differences']
        if case.get('fixture') and case['mode'] == 'trace':
            baseline = torch.from_numpy(np.fromfile(args.reference/case['fixture']/'logits.f32', dtype='<f4')
                                        .reshape(entry['rows'], spec['vocab_size']))
            metrics['cpp_vs_hf_dynamic'] = compare(actual, baseline)
            assert actual.argmax(-1).tolist() == fixtures[case['fixture']][1]
        for label, rows in (('hf', eager), ('aoti', aoti)):
            save_rows(args.output/(name+'.'+label+'.f32'), rows)
            if 'same_as' in case:
                clean = torch.from_numpy(np.fromfile(args.output/(case['same_as']+'.'+label+'.f32'), dtype='<f4')
                                         .reshape(rows.shape))
                compare(rows, clean, exact=True)
        if 'same_as' in case:
            clean = torch.from_numpy(np.fromfile(args.output/(case['same_as']+'.f32'), dtype='<f4')
                                     .reshape(actual.shape))
            compare(actual, clean, exact=True)
            metrics['future_mask_exact'] = True
        if case.get('dump_cache'):
            bits = np.fromfile(args.output/(name+'.kv.bf16'), dtype='<i2').reshape(2*spec['layers'], *spec['cache_shape'])
            cache_errors = []
            for index, (raw, compiled, original) in enumerate(zip(bits, py_cache, hf_tensors)):
                assert np.array_equal(raw, compiled.cpu().view(torch.int16).numpy())
                error = error_metrics(compiled.flatten(1).float(), original.flatten(1).float())
                error.pop('top1_agreement')  # Argmax has no meaning for a cache tensor.
                error.update(layer=index//2, kind='K' if index % 2 == 0 else 'V')
                error['bitwise_equal'] = torch.equal(compiled.view(torch.int16), original.view(torch.int16))
                cache_errors.append(error)
                if max(error['max_relative_l2'], error['max_peak_relative']) > MAX_RELATIVE_ERROR:
                    report['numerical_failures'].append(dict(case=name, **error))
            full_cache = ([t.cpu().float() for t in py_cache], [t.cpu().float() for t in hf_tensors])
            metrics['cache_cpp_vs_aoti_bitwise'] = True
            metrics['cache_aoti_vs_hf_static_bitwise'] = all(e['bitwise_equal'] for e in cache_errors)
            metrics['cache_aoti_vs_hf_static'] = cache_errors
            metrics['cache_sha256'] = hashlib.sha256(bits.tobytes()).hexdigest()
            assert entry['overflow_rejected_without_mutation'] and entry['position'] == CACHE_LENGTH
        report['cases'].append(metrics)
        print(dict(name=name, rows=entry['rows'], hf_relative_l2=metrics['aoti_vs_hf_static']['max_relative_l2'],
                   top1_differences=metrics['top1_differences']), flush=True)
    # A repeated prompt after different requests must reproduce the initial run exactly.
    first = (args.output/'capital_initial.f32').read_bytes()
    assert first == (args.output/'capital_after_full.f32').read_bytes() == (args.output/'capital_repeat.f32').read_bytes()
    # Keep an independent FP32 control even when eager BF16 parity passes.
    # Do not cast the original BF16 model again: its RoPE buffer must stay FP32.
    if full_cache is not None:
        model.float()
        hf_cache = StaticCache(model.config, 1, CACHE_LENGTH, device='cuda', dtype=torch.float32)
        context = next(c for c in cases if c['name'] == 'context_end')
        trace(hf_step, context['chunks'])
        fp32_cache = (t for pair in zip(hf_cache.key_cache, hf_cache.value_cache) for t in pair)
        report['full_cache_fp32_control'] = []
        for index, (aoti, bf16, fp32) in enumerate(zip(*full_cache, fp32_cache)):
            errors = {label: error_metrics(value.flatten(1), fp32.cpu().flatten(1))
                      for label, value in (('aoti_vs_fp32', aoti), ('bf16_vs_fp32', bf16))}
            for error in errors.values():
                error.pop('top1_agreement')
            report['full_cache_fp32_control'].append(dict(layer=index//2, kind='K' if index % 2 == 0 else 'V', **errors))
    report.update(total_rows=sum(c['rows'] for c in report['cases']), repeated_request_exact=True)
    report['passed'] = not report['numerical_failures']
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    if not report['passed']:
        raise SystemExit(f"FAIL: {len(report['numerical_failures'])} full-cache tensors exceed the unchanged "
                         f"{MAX_RELATIVE_ERROR:.0%} error limits; all cases and FP32 controls recorded in {args.output}/summary.json")
    print('PASS: three execution paths, forced inputs, boundaries, cache contents, masks, repeated requests', flush=True)


if __name__ == '__main__':
    main()
