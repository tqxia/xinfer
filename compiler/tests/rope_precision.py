"""Reproduce the layer-0 RoPE rounding difference with four scalar inputs."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--disable-fp-fusion', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose an absent output directory')
    args.output.mkdir(parents=True)
    # Set before importing/compiling; this option changes generated GPU code.
    os.environ['TRITON_DEFAULT_FP_FUSION'] = '0' if args.disable_fp_fusion else '1'
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str((args.output/'inductor').resolve())
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '2')

    import torch
    from torch._inductor.runtime.triton_helpers import get_backend_options

    class RotaryArithmetic(torch.nn.Module):
        def forward(self, q, cos, rotated_q, sin):
            a, b = q*cos, rotated_q*sin
            return a, b, a+b

    with torch.inference_mode():
        # Actual layer-0 Q values: head 0, position 423, channel 32, in the
        # first 512-token chunk of the context_end validation case.
        values = dict(q=18.5, cos=0.91015625, rotated_q=0.76171875, sin=0.41015625)
        # Repeat the same four values so Triton emits packed BF16 arithmetic,
        # as it does in the model. A one-element kernel takes a different path.
        inputs = tuple(torch.full((256,), x, dtype=torch.bfloat16, device='cuda') for x in values.values())
        module = RotaryArithmetic()
        options = {'emulate_precision_casts': True}
        graph = torch.export.export(module, inputs, strict=False)
        package = args.output/'rotary.pt2'
        torch._inductor.aoti_compile_and_package(graph, package_path=str(package), inductor_configs=options)
        compiled = torch._inductor.aoti_load_package(str(package))
        eager, actual = module(*inputs), compiled(*inputs)
        fp32 = module(*(x.float() for x in inputs))
        rows = []
        for label, a, b, f in zip(('q*cos', 'rotated_q*sin', 'sum'), actual, eager, fp32):
            assert all(torch.equal(t, t[0].expand_as(t)) for t in (a, b, f))
            rows.append(dict(operation=label, aoti=a[0].item(), eager_bf16=b[0].item(), fp32=f[0].item(),
                             aoti_fp32_abs_error=abs(a[0].item()-f[0].item()),
                             eager_fp32_abs_error=abs(b[0].item()-f[0].item())))
        assert all(torch.equal(a, b) for a, b in zip(actual[:2], eager[:2]))
        if args.disable_fp_fusion:
            assert torch.equal(actual[2], eager[2])
        report = dict(torch_version=torch.__version__, inputs=values, repetitions=256, inductor_configs=options,
                      enable_fp_fusion=get_backend_options()['enable_fp_fusion'], rows=rows,
                      sum_matches_eager=torch.equal(actual[2], eager[2]))
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
