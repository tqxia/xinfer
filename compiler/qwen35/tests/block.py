"""Layer-0 DeltaNet, with explicit state tensors around the unmodified HF module."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

from compiler.common.artifacts import sha256
from compiler.qwen35.config import MODEL_ID, REVISIONS, deltanet_state_shapes


MODEL_DIR = Path("work/qwen35/0.8b/block")


def prepare(directory, model=MODEL_ID):
    """Download the pinned shard, then retain only this block's original weights."""
    from huggingface_hub import HfApi, hf_hub_download

    revision = REVISIONS[model]

    def download(name):
        return Path(hf_hub_download(model, name, revision=revision,
                                   cache_dir="work/qwen35/hub"))

    directory.mkdir(parents=True, exist_ok=False)
    config = download("config.json")
    index = json.loads(download("model.safetensors.index.json").read_text())
    prefix = "model.language_model.layers.0.linear_attn."
    names = {k: v for k, v in index["weight_map"].items() if k.startswith(prefix)}
    if not names:
        raise ValueError(f"No layer-0 DeltaNet weights in {model}")
    info = HfApi().model_info(model, revision=revision, files_metadata=True)
    expected = {s.rfilename: s.lfs.sha256 for s in info.siblings if s.lfs}
    weights, shards = {}, {}
    for shard in sorted(set(names.values())):
        path = download(shard)
        shards[shard] = sha256(path)
        if shards[shard] != expected[shard]:
            raise ValueError(f"Hub checksum mismatch: {shard}")
        with safe_open(path, framework="pt", device="cpu") as reader:
            for name, filename in names.items():
                if filename == shard:
                    weights[name.removeprefix(prefix)] = reader.get_tensor(name)
    save_file(weights, directory / "block.safetensors")
    (directory / "config.json").write_bytes(config.read_bytes())
    manifest = dict(model=model, revision=revision, layer=0, source_shards=shards,
                    files={n: sha256(directory / n) for n in ("config.json", "block.safetensors")})
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


class TensorState:
    """The small part of HF's cache protocol used by this one block.

    Zero tensors represent a fresh request. Clone convolution storage because
    HF's decode convolution mutates it; the exported interface is functional.
    """

    def __init__(self, conv, recurrent):
        self.layers = [self]
        self.record_past = False
        self.conv_states = [conv.clone()]
        self.recurrent_states = [recurrent]

    def has_previous_state(self, layer_idx, state_idx=0):
        return True

    def update_conv_state(self, value, layer_idx, conv_kernel_size):
        combined = torch.cat((self.conv_states[0], value), dim=-1)
        self.conv_states[0] = combined[..., -conv_kernel_size:].contiguous()
        return combined

    def update_recurrent_state(self, value, layer_idx):
        self.recurrent_states[0] = value
        return value


class BlockStep(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden, conv, recurrent):
        cache = TensorState(conv, recurrent)
        output = self.block(hidden, cache_params=cache)
        return output, cache.conv_states[0], cache.recurrent_states[0]


def load_block(directory=MODEL_DIR, dtype=torch.bfloat16):
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    config = Qwen3_5TextConfig(**json.loads((directory / "config.json").read_text())["text_config"])
    if config.layer_types[0] != "linear_attention":
        raise ValueError("Layer 0 must be a DeltaNet block")
    block = Qwen3_5GatedDeltaNet(config, layer_idx=0)
    block.load_state_dict(load_file(directory / "block.safetensors"), strict=True)
    return block.eval().requires_grad_(False).to(device="cuda", dtype=dtype), config


def input_shapes(config, tokens):
    return ((1, tokens, config.hidden_size), *deltanet_state_shapes(config))

def zero_state(config, dtype=torch.bfloat16):
    _, conv, recurrent = input_shapes(config, 1)
    return (torch.zeros(conv, device="cuda", dtype=dtype),
            torch.zeros(recurrent, device="cuda", dtype=torch.float32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=REVISIONS, default=MODEL_ID)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.output or Path("work/qwen35") / args.model.rsplit("-", 1)[1].lower() / "block"
    prepare(directory, args.model)
