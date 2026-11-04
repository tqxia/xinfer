"""HF text model with caller-owned KV buffers and functional DeltaNet state."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import Qwen3_5ForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextConfig,
    Qwen3_5TextRotaryEmbedding,
)

from compiler.qwen35.config import (
    CACHE_LENGTH,
    MODEL_ID,
    REVISIONS,
    TEXT_DIR,
    configure,
    deltanet_state_shapes,
)
from compiler.common.artifacts import sha256
from compiler.common.metadata import state_spec


MODEL_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def _text_config(path):
    config = Qwen3_5TextConfig(**json.loads(path.read_text())["text_config"])
    if not config.tie_word_embeddings or config.rope_parameters["rope_type"] != "default":
        raise ValueError("Qwen3.5 text export requires tied embeddings and default RoPE")
    config._attn_implementation = "sdpa"
    return config


def prepare(directory, model=MODEL_ID):
    """Extract text weights from either supported, pinned multimodal checkpoint."""
    if not isinstance(model, str) or model not in REVISIONS:
        raise ValueError(f"Unsupported checkpoint: {model}")
    from huggingface_hub import HfApi, hf_hub_download

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    revision = REVISIONS[model]

    def download(name):
        return Path(hf_hub_download(model, name, revision=revision, cache_dir="work/qwen35/hub"))

    config_path = download("config.json")
    _text_config(config_path)
    info = HfApi().model_info(model, revision=revision, files_metadata=True)
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
    for name in MODEL_FILES:
        path = config_path if name == "config.json" else download(name)
        (directory / name).write_bytes(path.read_bytes())
    manifest = dict(
        model=model,
        revision=revision,
        text_parameters=len(weights),
        source_shards=shards,
        files={name: sha256(directory / name) for name in (*MODEL_FILES, "text.safetensors")},
    )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def load_model(directory=TEXT_DIR, dtype=torch.bfloat16):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Prepared checkpoint manifest must be an object")
    model_id = manifest.get("model")
    if (
        not isinstance(model_id, str)
        or model_id not in REVISIONS
        or manifest.get("revision") != REVISIONS[model_id]
    ):
        raise ValueError("Prepared checkpoint must match a supported pinned revision")
    if not isinstance(manifest.get("files"), dict) or set(manifest["files"]) != {
        *MODEL_FILES,
        "text.safetensors",
    }:
        raise ValueError("Prepared checkpoint file manifest is incomplete or unsupported")
    for name, expected in manifest["files"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    config = _text_config(directory / "config.json")
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    weights = {
        name: value.to(dtype) for name, value in load_file(directory / "text.safetensors").items()
    }
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
        specs.extend(
            state_spec(
                layer, name, shape, dtype, "inplace" if kind == "full_attention" else "replace"
            )
            for name, shape, dtype in tensors
        )
    return specs


def zero_state(config):
    return [
        torch.zeros(s["shape"], device="cuda", dtype=getattr(torch, s["dtype"]))
        for s in state_specs(config)
    ]


class LayerState:
    def __init__(self, kind, a, b):
        if kind == "linear_attention":
            # DeltaNet reads caller state and returns fresh replacement tensors.
            self.conv_states = [a]
            self.recurrent_states = [b]
            self.record_past = False
        else:
            # Keep updates in list slots, as for DeltaNet state. Torch 2.13
            # export can lose attribute assignments after a symbolic cond.
            self.kv = [a, b]


class TextCache:
    def __init__(self, kinds, position, state, extent=None):
        self.position = position
        self.extent = extent
        self.kinds = kinds
        self.layers = []
        for i, kind in enumerate(kinds):
            a, b = state[2 * i : 2 * i + 2]
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
        layer.kv[0].index_copy_(2, self.position, key)
        layer.kv[1].index_copy_(2, self.position, value)
        return tuple(layer.kv)

    def tensors(self):
        result = []
        for kind, layer in zip(self.kinds, self.layers):
            result.extend(
                (layer.conv_states[0], layer.recurrent_states[0])
                if kind == "linear_attention"
                else layer.kv
            )
        return result


class TextStep(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        from compiler.qwen35.adapters.attention import install_attention

        install_attention(model)
        self.original = model

    def forward(self, input_ids, cache_position, state, kv_extent):
        model = self.original.model
        cache = TextCache(model.config.layer_types, cache_position, state, kv_extent)
        hidden = model.embed_tokens(input_ids)
        positions = cache_position.view(1, 1, -1).expand(3, 1, -1)
        rotary = model.rotary_emb(hidden, positions)
        for kind, layer in zip(model.config.layer_types, model.layers):
            hidden = layer(
                hidden,
                position_embeddings=rotary,
                attention_mask=None,
                position_ids=cache_position[None, :],
                past_key_values=cache,
                use_cache=True,
            )
        hidden = model.norm(hidden)
        logits = self.original.lm_head(hidden[:, -1:, :])[:, 0, :]
        return (logits, *cache.tensors())


def configure_text():
    configure()
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=REVISIONS, default=MODEL_ID)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    directory = (
        args.output or TEXT_DIR.parent.parent / args.model.rsplit("-", 1)[1].lower() / "text"
    )
    prepare(directory, args.model)


if __name__ == "__main__":
    main()
