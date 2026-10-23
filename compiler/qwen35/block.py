"""Layer-0 DeltaNet, with explicit state tensors around the unmodified HF module."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet


MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MODEL_DIR = Path("work/qwen35/block")
HIDDEN = 2560
CONV_SHAPE = (1, 8192, 4)
STATE_SHAPE = (1, 32, 128, 128)
PREFILL = 64


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(directory):
    """Download the pinned shard, then retain only this block's original weights."""
    from huggingface_hub import HfApi, hf_hub_download

    def download(name):
        return Path(hf_hub_download(MODEL_ID, name, revision=REVISION,
                                   cache_dir="work/qwen35/hub"))

    directory.mkdir(parents=True, exist_ok=True)
    config = download("config.json")
    index = json.loads(download("model.safetensors.index.json").read_text())
    prefix = "model.language_model.layers.0.linear_attn."
    names = {k: v for k, v in index["weight_map"].items() if k.startswith(prefix)}
    info = HfApi().model_info(MODEL_ID, revision=REVISION, files_metadata=True)
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
    manifest = dict(model=MODEL_ID, revision=REVISION, layer=0, source_shards=shards,
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
    assert (config.hidden_size, config.linear_num_value_heads, config.linear_key_head_dim,
            config.linear_value_head_dim, config.linear_conv_kernel_dim) == (2560, 32, 128, 128, 4)
    block = Qwen3_5GatedDeltaNet(config, layer_idx=0)
    block.load_state_dict(load_file(directory / "block.safetensors"), strict=True)
    return block.eval().requires_grad_(False).to(device="cuda", dtype=dtype), config


def zero_state(dtype=torch.bfloat16):
    return (torch.zeros(CONV_SHAPE, device="cuda", dtype=dtype),
            torch.zeros(STATE_SHAPE, device="cuda", dtype=torch.float32))


def configure():
    if importlib.metadata.version("transformers") != "5.20.2":
        raise RuntimeError("This adapter targets transformers==5.20.2")
    for package in ("fla", "causal_conv1d"):
        if importlib.util.find_spec(package) is not None:
            raise RuntimeError(f"Use an environment without {package} for the PyTorch fallback baseline")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MODEL_DIR)
    prepare(parser.parse_args().output)
