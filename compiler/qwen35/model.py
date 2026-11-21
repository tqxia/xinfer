"""HF text model with caller-owned KV buffers and functional DeltaNet state."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import Qwen3_5ForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextConfig,
    Qwen3_5TextRotaryEmbedding,
)

from compiler.qwen35.config import TEXT_DIR
from compiler.common.attention import batched_attention_metadata
from compiler.common.kernels.kv_cache import append_kv_batched
from compiler.common.metadata import state_spec
from compiler.common.context import resolve_context_length


def _text_config(path):
    document = json.loads(path.read_text())
    if document.get("quantization_config") is not None:
        raise ValueError("Qwen3.5 text export requires an unquantized checkpoint")
    config = Qwen3_5TextConfig(**document["text_config"])
    if not config.tie_word_embeddings or config.rope_parameters["rope_type"] != "default":
        raise ValueError("Qwen3.5 text export requires tied embeddings and default RoPE")
    config._attn_implementation = "sdpa"
    return config


def create_text_model(config):
    """Construct the shared text architecture without allocating weight storage."""
    with torch.device("meta"):
        return Qwen3_5ForCausalLM(config)


def finalize_text_model(model):
    """Restore FP32 nonpersistent buffers and place the loaded model on CUDA."""
    # Recreate nonpersistent meta buffers in FP32, including the RoPE frequencies.
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(model.config)
    return model.eval().requires_grad_(False).to("cuda")


def load_model(directory=TEXT_DIR, dtype=torch.bfloat16):
    directory = Path(directory)
    config = _text_config(directory / "config.json")
    model = create_text_model(config)
    weights = {
        name: value.to(dtype) for name, value in load_file(directory / "text.safetensors").items()
    }
    weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    model.load_state_dict(weights, strict=True, assign=True)
    model.lm_head.weight = model.model.embed_tokens.weight
    return finalize_text_model(model)


def deltanet_state_shapes(config, batch_size=1):
    key_width = config.linear_num_key_heads * config.linear_key_head_dim
    value_width = config.linear_num_value_heads * config.linear_value_head_dim
    return (
        (batch_size, 2 * key_width + value_width, config.linear_conv_kernel_dim),
        (
            batch_size,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
        ),
    )


def state_specs(config, capacity=None, *, batch_size=1):
    capacity = resolve_context_length(config, capacity)
    conv_shape, recurrent_shape = deltanet_state_shapes(config, batch_size)
    kv_shape = (batch_size, config.num_key_value_heads, capacity, config.head_dim)
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


def zero_state(config, capacity=None, *, batch_size=1):
    return [
        torch.zeros(s["shape"], device="cuda", dtype=getattr(torch, s["dtype"]))
        for s in state_specs(config, capacity, batch_size=batch_size)
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


class HybridCache:
    """Caller-owned KV buffers and replacement DeltaNet state."""

    def __init__(self, kinds, position, state, kv_lengths, query_lengths):
        self.position = position
        self.offsets, self.cuda_offsets, self.blocks = batched_attention_metadata(
            kv_lengths, query_lengths, position.device
        )
        self.query_lengths = self.cuda_offsets[0, 1:] - self.cuda_offsets[0, :-1]
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
        append_kv_batched(key, value, self.position, self.cuda_offsets[0], *layer.kv)
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


def text_rotary(rotary, positions, dtype):
    # Text uses identical positions in all three MRoPE grids. HF's in-place
    # recomposition is redundant here and meta_copy_ introduces a B >= 2 guard.
    # Compute the same frequencies without those cross-grid slice assignments.
    frequencies = positions[..., None].float() * rotary.inv_freq.float()
    cos = frequencies.cos() * rotary.attention_scaling
    sin = frequencies.sin() * rotary.attention_scaling
    return tuple(torch.cat((value, value), dim=-1).to(dtype) for value in (cos, sin))


class ModelStep(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        from compiler.qwen35.adapters.attention import install_attention

        install_attention(model)
        self.original = model

    def forward(self, input_ids, cache_position, state, kv_lengths, query_lengths):
        model = self.original.model
        cache = HybridCache(
            model.config.layer_types, cache_position, state, kv_lengths, query_lengths
        )
        hidden = model.embed_tokens(input_ids)
        rotary = text_rotary(model.rotary_emb, cache_position, hidden.dtype)
        for layer in model.layers:
            hidden = layer(
                hidden,
                position_embeddings=rotary,
                attention_mask=None,
                position_ids=cache_position,
                past_key_values=cache,
                use_cache=True,
            )
        hidden = model.norm(hidden)
        counts = cache.query_lengths.to(torch.int64)
        indices = (counts - 1).clamp_min(0)
        last = hidden.gather(1, indices[:, None, None].expand(-1, 1, hidden.shape[-1]))
        logits = self.original.lm_head(last)[:, 0, :]
        return (logits.masked_fill(counts[:, None] == 0, 0), *cache.tensors())
