"""Original HF text model with a flat, functional tensor-state interface."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import Qwen3_5ForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextConfig, Qwen3_5TextRotaryEmbedding

from compiler.qwen35.config import CACHE_LENGTH, MODEL_ID, REVISIONS, TEXT_DIR, configure, deltanet_state_shapes
from compiler.common.artifacts import sha256


def prepare(directory):
    from huggingface_hub import HfApi, hf_hub_download

    directory.mkdir(parents=True, exist_ok=False)
    revision = REVISIONS[MODEL_ID]

    def download(name):
        return Path(hf_hub_download(MODEL_ID, name, revision=revision, cache_dir="work/qwen35/hub"))

    info = HfApi().model_info(MODEL_ID, revision=revision, files_metadata=True)
    expected = {s.rfilename: s.lfs.sha256 for s in info.siblings if s.lfs}
    index = json.loads(download("model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.language_model."
    names = {name: shard for name, shard in index.items() if name.startswith(prefix)}
    weights, shards = {}, {}
    for shard in sorted(set(names.values())):
        path = download(shard)
        shards[shard] = sha256(path)
        if shards[shard] != expected[shard]:
            raise ValueError(f"Hub checksum mismatch: {shard}")
        with safe_open(path, framework="pt", device="cpu") as reader:
            for name, filename in names.items():
                if filename == shard:
                    weights["model." + name.removeprefix(prefix)] = reader.get_tensor(name)
    if not weights:
        raise ValueError("No text weights found")
    save_file(weights, directory / "text.safetensors")
    files = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
    for name in files:
        (directory / name).write_bytes(download(name).read_bytes())
    manifest = dict(model=MODEL_ID, revision=revision, text_parameters=len(weights), source_shards=shards,
                    files={name: sha256(directory / name) for name in files + ["text.safetensors"]})
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def load_model(directory=TEXT_DIR, dtype=torch.bfloat16):
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    config = Qwen3_5TextConfig(**json.loads((directory / "config.json").read_text())["text_config"])
    config._attn_implementation = "sdpa"
    if not config.tie_word_embeddings or config.rope_parameters["rope_type"] != "default":
        raise ValueError("This probe expects tied embeddings and default RoPE")
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    weights = {name: value.to(dtype) for name, value in load_file(directory / "text.safetensors").items()}
    weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    model.load_state_dict(weights, strict=True, assign=True)
    model.lm_head.weight = model.model.embed_tokens.weight
    # Recreate nonpersistent meta buffers in FP32, including the RoPE frequencies.
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
    return model.eval().requires_grad_(False).to("cuda")


def state_specs(config):
    conv_shape, recurrent_shape = deltanet_state_shapes(config)
    kv_shape = (1, config.num_key_value_heads, CACHE_LENGTH, config.head_dim)
    specs = []
    for layer, kind in enumerate(config.layer_types):
        if kind == "linear_attention":
            tensors = (("conv", conv_shape, "bfloat16"), ("recurrent", recurrent_shape, "float32"))
        elif kind == "full_attention":
            tensors = (("key", kv_shape, "bfloat16"), ("value", kv_shape, "bfloat16"))
        else:
            raise ValueError(f"Unsupported layer type: {kind}")
        specs.extend(dict(name=f"layer_{layer}.{name}", layer=layer, kind=name, shape=list(shape), dtype=dtype)
                     for name, shape, dtype in tensors)
    return specs


def zero_state(config):
    return [torch.zeros(s["shape"], device="cuda", dtype=getattr(torch, s["dtype"])) for s in state_specs(config)]


class LayerState:
    def __init__(self, kind, a, b):
        if kind == "linear_attention":
            self.conv_states = [a.clone()]
            self.recurrent_states = [b]
            self.record_past = False
        else:
            self.keys, self.values = a, b


class TextCache:
    def __init__(self, kinds, position, state):
        self.position = position
        self.kinds = kinds
        self.layers = []
        for i, kind in enumerate(kinds):
            a, b = state[2 * i:2 * i + 2]
            self.layers.append(LayerState(kind, a, b))

    def has_previous_state(self, layer_idx, state_idx=0):
        return True

    def update_conv_state(self, value, layer_idx, conv_kernel_size):
        layer = self.layers[layer_idx]
        combined = torch.cat((layer.conv_states[0], value), dim=-1)
        layer.conv_states[0] = combined[..., -conv_kernel_size:].contiguous()
        return combined

    def update_recurrent_state(self, value, layer_idx):
        self.layers[layer_idx].recurrent_states[0] = value
        return value

    def update(self, key, value, layer_idx):
        layer = self.layers[layer_idx]
        layer.keys = layer.keys.index_copy(2, self.position, key)
        layer.values = layer.values.index_copy(2, self.position, value)
        return layer.keys, layer.values

    def tensors(self):
        result = []
        for kind, layer in zip(self.kinds, self.layers):
            result.extend((layer.conv_states[0], layer.recurrent_states[0]) if kind == "linear_attention"
                          else (layer.keys, layer.values))
        return result


class TextStep(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.original = model

    def forward(self, input_ids, cache_position, state):
        model = self.original.model
        cache = TextCache(model.config.layer_types, cache_position, state)
        hidden = model.embed_tokens(input_ids)
        positions = cache_position.view(1, 1, -1).expand(3, 1, -1)
        rotary = model.rotary_emb(hidden, positions)
        allowed = torch.arange(CACHE_LENGTH, device=input_ids.device)[None, :] <= cache_position[:, None]
        mask = torch.zeros_like(allowed, dtype=hidden.dtype).masked_fill(~allowed, torch.finfo(hidden.dtype).min)
        mask = mask[None, None, :, :]
        for kind, layer in zip(model.config.layer_types, model.layers):
            hidden = layer(hidden, position_embeddings=rotary,
                           attention_mask=mask if kind == "full_attention" else None,
                           position_ids=cache_position[None, :], past_key_values=cache, use_cache=True)
        hidden = model.norm(hidden)
        logits = self.original.lm_head(hidden[:, -1:, :])[:, 0, :]
        return (logits, *cache.tensors())


def configure_text():
    configure()
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=TEXT_DIR)
    prepare(parser.parse_args().output)
