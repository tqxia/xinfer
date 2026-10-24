"""Validate native FlashAttention causality/GQA and isolate full-model BF16 drift.

Run from the repository root with PYTHONPATH=compiler after validate_cpp.py.
The attention checks assert the existing 5% limits; full-model diagnostics report
each comparison without changing validate_cpp.py's independent acceptance result.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from model import QwenStep, allocate_cache, flash_attention_forward, load_model, cache_extent
from prepare import DEFAULT_MODEL
from validate_aoti import compare, error_metrics


def cache_metrics(actual, expected):
    result = []
    for i, (a, b) in enumerate(zip(actual, expected)):
        metric = error_metrics(a.flatten(1).float(), b.flatten(1).float())
        metric.pop('top1_agreement')
        result.append(dict(layer=i//2, kind='K' if i%2 == 0 else 'V', **metric))
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--validation', type=Path, default=Path('artifacts/three-path-flash'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose an absent output file')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = dict(attention_cases=[])
    # Explicit FP32 scores/mask are independent of the Flash causal dispatcher.
    for nq, nk in ((1,1), (1,2), (1,32), (1,1024), (2,2), (2,3), (16,32), (512,512), (510,1022)):
        q = torch.randn((1,16,nq,128),device='cuda',dtype=torch.bfloat16)
        k = torch.randn((1,8,1025,128),device='cuda',dtype=torch.bfloat16)
        v = torch.randn_like(k)
        actual, _ = flash_attention_forward(SimpleNamespace(sliding_window=None),q,k[:,:,:nk],v[:,:,:nk],None,
                                             scaling=128**-0.5)
        actual = actual.transpose(1,2).float()
        positions = torch.arange(nq,device='cuda') + nk-nq
        mask = torch.arange(nk,device='cuda')[None,:] <= positions[:,None]
        with sdpa_kernel(SDPBackend.MATH):
            expected = torch.nn.functional.scaled_dot_product_attention(
                q.float(),k[:,:,:nk].float(),v[:,:,:nk].float(),attn_mask=mask,enable_gqa=True)
        metric = compare(actual.flatten(1),expected.flatten(1))
        k[:,:,nk:].fill_(100)
        v[:,:,nk:].fill_(-100)
        poisoned, _ = flash_attention_forward(SimpleNamespace(sliding_window=None),q,k[:,:,:nk],v[:,:,:nk],None,
                                               scaling=128**-0.5)
        assert torch.equal(actual,poisoned.transpose(1,2).float())
        report['attention_cases'].append(dict(query_tokens=nq,valid_keys=nk,future_poison_exact=True,**metric))

    cases = json.loads((args.validation/'cases.json').read_text())['cases']
    chunks = next(c for c in cases if c['name']=='context_end')['chunks']
    snapshots, logits = {}, {}
    for backend in ('efficient','flash'):
        model = load_model(args.model_dir, backend)
        fn, cache = QwenStep(model), allocate_cache(model.config, backend)
        position, rows = 0, []
        for tokens in chunks:
            ids = torch.tensor([tokens],device='cuda',dtype=torch.int64)
            pos = torch.arange(position,position+len(tokens),device='cuda')
            extra = (cache_extent(position+len(tokens)),) if backend=='flash' else ()
            rows.append(fn(ids,pos,cache,*extra)[0].float().cpu())
            position += len(tokens)
        snapshots[backend] = [t.float().cpu() for t in cache]
        logits[backend] = torch.stack(rows)
        del fn, cache
        if backend == 'efficient':
            # Independent FP32 control with the original dense-mask implementation.
            model.float()
            fn = QwenStep(model)
            cache = tuple(t.float() for t in allocate_cache(model.config))
            position = 0
            for tokens in chunks:
                ids = torch.tensor([tokens],device='cuda',dtype=torch.int64)
                pos = torch.arange(position,position+len(tokens),device='cuda')
                fn(ids,pos,cache)
                position += len(tokens)
            snapshots['fp32'] = [t.cpu() for t in cache]
            del fn,cache
        del model
        torch.cuda.empty_cache()
    bits = np.fromfile(args.validation/'context_end.kv.bf16',dtype='<i2').reshape(56,1,8,1024,128)
    snapshots['aoti_flash'] = [torch.from_numpy(t.copy()).view(torch.bfloat16).float() for t in bits]
    logits['aoti_flash'] = torch.from_numpy(np.fromfile(args.validation/'context_end.f32',dtype='<f4').reshape(4,-1))
    report['cache_comparisons'] = {}
    for a,b in (('flash','efficient'),('aoti_flash','flash'),('aoti_flash','fp32'),('flash','fp32'),('efficient','fp32')):
        report['cache_comparisons'][a+'_vs_'+b] = cache_metrics(snapshots[a],snapshots[b])
    report['logit_comparisons'] = {a+'_vs_'+b:error_metrics(logits[a],logits[b])
                                 for a,b in (('flash','efficient'),('aoti_flash','flash'))}
    report['attention_checks_passed'] = True
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['logit_comparisons'],indent=2),flush=True)
    for name, rows in report['cache_comparisons'].items():
        print(name,{k:max(r[k] for r in rows) for k in ('max_relative_l2','max_peak_relative')},flush=True)
    print('PASS: native attention matches explicit FP32 causality/GQA; full-model drift recorded separately',flush=True)


if __name__ == '__main__':
    main()
