"""Compare the adapter with the original HF block and verify its exported state contract."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from adapters.qwen3_flash import FlashAttention, FlashStep, PrefixCache
from custom_ops.attention import load_attention_ops
from model import CACHE_LENGTH, TensorCache, cache_extent, load_model
from prepare import DEFAULT_MODEL
from validate_aoti import compare


class AttentionProbe(torch.nn.Module):
    def __init__(self, attention, rotary, adapter=False):
        super().__init__()
        self.attention = FlashAttention(attention) if adapter else attention
        self.rotary = rotary
        self.adapter = adapter
        self.eval()

    def forward(self, hidden, cache_position, kv, kv_extent):
        mask = None
        if self.adapter:
            cache = PrefixCache(kv, cache_position, kv_extent)
        else:
            cache = TensorCache(kv, cache_position)
            future = torch.arange(CACHE_LENGTH, device=hidden.device).unsqueeze(0) > cache_position.unsqueeze(1)
            mask = torch.zeros_like(future, dtype=hidden.dtype)
            mask = mask.masked_fill(future, torch.finfo(hidden.dtype).min)[None, None]
        rotary = self.rotary(hidden, cache_position.unsqueeze(0))
        return self.attention(hidden, position_embeddings=rotary, attention_mask=mask,
                              past_key_value=cache, cache_position=cache_position)[0]


def inputs(tokens, length, hidden_size):
    return (torch.randn(1, tokens, hidden_size, device="cuda", dtype=torch.bfloat16),
            torch.arange(length - tokens, length, device="cuda"),
            tuple(torch.randn(1, 8, CACHE_LENGTH, 128, device="cuda", dtype=torch.bfloat16)
                  for _ in range(2)), cache_extent(length))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    load_attention_ops()
    model = load_model(args.model_dir)
    original = model.model.layers[0].self_attn
    hf = AttentionProbe(original, model.model.rotary_emb)
    adapter = AttentionProbe(original, model.model.rotary_emb, adapter=True)
    report = {"cases": [], "rejections": []}
    for kind in ("decode", "prefill"):
        example = inputs(1 if kind == "decode" else 16, 32, model.config.hidden_size)
        tokens = torch.export.Dim("tokens", min=2, max=512)
        extent = torch.export.Dim("extent", min=2, max=1025)
        shapes = ({} if kind == "decode" else {1: tokens},
                  {} if kind == "decode" else {0: tokens}, ({}, {}), {0: extent})
        adapter_export = torch.export.export(adapter, example, dynamic_shapes=shapes, strict=False)
        adapter_export.validate()
        targets = [n.target for n in adapter_export.graph.nodes if n.op == "call_function"]
        assert targets.count(torch.ops.xinfer.flash_attention.default) == 1
        assert targets.count(torch.ops.aten.index_copy_.default) == 2
        assert torch.ops.aten.scaled_dot_product_attention.default not in targets
        site = next(n for n in adapter_export.graph.nodes if n.target == torch.ops.xinfer.flash_attention.default)
        assert site.args[0].meta["val"].is_contiguous()
        exported_adapter = adapter_export.module()
        (args.output / f"{kind}.graph.txt").write_text(adapter_export.graph_module.code)
        cases = ((1, 1), (1, 2), (1, 1024)) if kind == "decode" else (
            (2, 2), (2, 3), (16, 32), (510, 1022), (512, 512), (512, 1024))
        for count, length in cases:
            hidden, positions, initial, valid = inputs(count, length, model.config.hidden_size)
            outputs, caches = [], []
            for fn in (hf, adapter, exported_adapter):
                kv = tuple(t.clone() for t in initial)
                pointers = [t.data_ptr() for t in kv]
                outputs.append(fn(hidden, positions, kv, valid))
                assert pointers == [t.data_ptr() for t in kv]
                start = length - count
                for before, after in zip(initial, kv):
                    assert torch.equal(before[:, :, :start], after[:, :, :start])
                    assert torch.equal(before[:, :, length:], after[:, :, length:])
                caches.append(kv)
            hf_error = compare(outputs[1].flatten(0, 1).float(), outputs[0].flatten(0, 1).float())
            hf_error.pop("top1_agreement")  # Hidden-state argmax is not a token prediction.
            assert torch.equal(outputs[2], outputs[1])
            for kv in caches[1:]:
                assert all(torch.equal(a, b) for a, b in zip(kv, caches[0]))
            poisoned = tuple(t.clone() for t in initial)
            for t in poisoned:
                t[:, :, length:].fill_(100)
            valid.fill_(255)
            assert torch.equal(adapter(hidden, positions, poisoned, valid), outputs[1])
            report["cases"].append(dict(kind=kind, tokens=count, length=length,
                                       adapter_vs_hf=hf_error, adapter_export_exact=True, cache_exact=True,
                                       write_region_and_storage=True, future_poison_exact=True))
            print(report["cases"][-1], flush=True)

    hidden, positions, kv, extent = inputs(1, 1, model.config.hidden_size)
    arguments = dict(position_embeddings=model.model.rotary_emb(hidden, positions[None]),
                     past_key_value=PrefixCache(kv, positions, extent), cache_position=positions)
    for name, changes in (("mask", {"attention_mask": torch.zeros(1, 1, 1, 1024, device="cuda")}),
                          ("cache", {"past_key_value": TensorCache(kv, positions)}),
                          ("positions", {"cache_position": positions.clone()}),
                          ("attention_weights", {"output_attentions": True}),
                          ("training", {})):
        adapter.attention.train(name == "training")
        try:
            adapter.attention(hidden, **(arguments | changes))
        except ValueError:
            report["rejections"].append(name)
        else:
            raise AssertionError(f"Accepted unsupported {name}")
    original_weights = {id(p) for p in model.parameters()}
    layers = list(model.model.layers)
    step = FlashStep(model)
    assert original_weights == {id(p) for p in step.parameters()}
    assert all(a is b for a, b in zip(layers, model.model.layers))
    assert model.model.layers[0].self_attn.original is original
    report.update(shared_weights=True, decoder_layers_preserved=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
