"""Synthetic attention-only opportunity probe; not an end-to-end model benchmark."""
import argparse
import json
import statistics
from pathlib import Path
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention as sdpa

p = argparse.ArgumentParser()
p.add_argument('output', type=Path)
args = p.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
torch.set_num_threads(1)
torch.manual_seed(1234)
results = []
with torch.inference_mode():
    for phase in ('decode', 'prefill'):
        for length in (25,128,512):
            qlen = 1 if phase == 'decode' else length
            valid = length+1 if phase == 'decode' else length
            q = torch.randn((1,16,qlen,128), device='cuda', dtype=torch.bfloat16)
            k = torch.randn((1,8,1024,128), device='cuda', dtype=torch.bfloat16)
            v = torch.randn_like(k)
            positions = torch.tensor([length],device='cuda') if phase == 'decode' else torch.arange(length,device='cuda')
            mask = torch.zeros((qlen,1024), device='cuda',dtype=torch.bfloat16)
            mask.masked_fill_(torch.arange(1024,device='cuda')[None,:] > positions[:,None], torch.finfo(torch.bfloat16).min)
            mask = mask[None,None]
            def baseline():
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    return sdpa(q,k.repeat_interleave(2,dim=1),v.repeat_interleave(2,dim=1),attn_mask=mask)
            def candidate():
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    return sdpa(q,k[:,:,:valid],v[:,:,:valid],is_causal=(phase=='prefill'),enable_gqa=True)
            ref, alt = baseline(), candidate()
            diff=(ref.float()-alt.float())
            row=dict(phase=phase, prompt_length=length, valid_keys=valid,
                     max_abs_error=diff.abs().max().item(),
                     relative_l2=(diff.norm()/ref.float().norm()).item(),
                     finite=bool(torch.isfinite(alt).all().item()), timings_us={})
            assert row['finite']
            # Capture 16 calls per replay to amortize Python dispatch. These tensors
            # are reused, so cache behavior is not representative of 28 model layers.
            for name, fn in [('efficient_full_cache_repeat_kv',baseline),('flash_valid_prefix_native_gqa',candidate)]:
                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(16):
                        result=fn()
                graph.replay()
                torch.cuda.synchronize()
                times=[]
                for _ in range(3):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(50):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    times.append(start.elapsed_time(end)*1000/(16*50))
                row['timings_us'][name]=dict(samples=times, median=statistics.median(times))
                del graph,result
            results.append(row)
            print(json.dumps(row),flush=True)
args.output.write_text(json.dumps(dict(torch_version=torch.__version__, device=torch.cuda.get_device_name(), cases=results),indent=2)+'\n')
