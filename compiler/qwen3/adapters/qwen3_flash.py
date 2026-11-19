"""Qwen3-0.6B FlashAttention through child-module replacement, without FX passes.

The tensor entry point uses right-padded independent rows at every batch size,
with compact KV prefixes. The C++ runtime validates positions and capacity.
Weights are loaded with HF first; wrappers reuse the original modules in place.
"""

import importlib.metadata

try:
    _transformers_version = importlib.metadata.version("transformers")
except importlib.metadata.PackageNotFoundError as error:
    raise RuntimeError("Qwen3 Flash adapter requires transformers==4.51.3") from error
if _transformers_version != "4.51.3":
    raise RuntimeError(
        f"Qwen3 Flash adapter requires transformers==4.51.3; found {_transformers_version}"
    )

import torch
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, apply_rotary_pos_emb

from compiler.common.attention import load_attention_ops, batched_attention_metadata
from compiler.common.kernels.kv_cache import append_kv_batched


class KVCache(Cache):
    """Caller-owned KV tensors with per-row positions and contiguous valid lengths."""

    def __init__(self, tensors, position, kv_lengths, query_lengths):
        super().__init__()
        self.tensors = tensors
        self.position = position
        self.offsets, self.cuda_offsets, self.blocks = batched_attention_metadata(
            kv_lengths, query_lengths, position.device
        )

    def update(self, key, value, layer_idx, cache_kwargs=None):
        k, v = self.tensors[2 * layer_idx : 2 * layer_idx + 2]
        append_kv_batched(key, value, self.position, self.cuda_offsets[0], k, v)
        return k, v


class FlashAttention(torch.nn.Module):
    """Keep HF's weights and arithmetic around one explicit custom-op call."""

    def __init__(self, original):
        super().__init__()
        if type(original) is not Qwen3Attention or original.sliding_window is not None:
            raise ValueError("Expected full-attention Qwen3Attention")
        config = original.config
        if (config.num_attention_heads, config.num_key_value_heads, original.head_dim) != (
            16,
            8,
            128,
        ):
            raise ValueError("The native Flash op targets Qwen3-0.6B head dimensions")
        self.original = original

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        if self.training or kwargs.get("output_attentions", False):
            raise ValueError("Flash adapter supports inference without attention weights")
        if attention_mask is not None:
            raise ValueError("Flash adapter uses KVCache lengths instead of attention masks")
        if not isinstance(past_key_value, KVCache) or cache_position is not past_key_value.position:
            raise ValueError("Flash adapter requires KVCache with the same position tensor")
        a = self.original
        input_shape = hidden_states.shape[:-1]
        head_shape = (*input_shape, -1, a.head_dim)
        q = a.q_norm(a.q_proj(hidden_states).view(head_shape)).transpose(1, 2)
        k = a.k_norm(a.k_proj(hidden_states).view(head_shape)).transpose(1, 2)
        v = a.v_proj(hidden_states).view(head_shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k, v = past_key_value.update(
            k, v, a.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position}
        )
        output = torch.ops.xinfer.batched_flash_attention(
            q,
            k,
            v,
            past_key_value.offsets,
            past_key_value.cuda_offsets,
            past_key_value.blocks,
            a.scaling,
        )
        output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return a.o_proj(output), None


class FlashStep(torch.nn.Module):
    """Install adapters once; drive unchanged HF decoder layers with tensor state.

    This small loop avoids HF's automatic mask construction. Causality lives in the
    prefix op; arbitrary masks are deliberately outside this inference interface.
    """

    def __init__(self, model):
        super().__init__()
        # Construct all replacements before modifying the model. No weight copies.
        replacements = [FlashAttention(layer.self_attn) for layer in model.model.layers]
        load_attention_ops(model="qwen3")
        for layer, attention in zip(model.model.layers, replacements):
            layer.self_attn = attention
        self.model = model
        self.eval()

    def forward(self, input_ids, cache_position, kv, kv_lengths, query_lengths):
        cache = KVCache(kv, cache_position, kv_lengths, query_lengths)
        base = self.model.model
        hidden = base.embed_tokens(input_ids)
        # HF's batched RoPE matmul specializes B=1 during export. Flatten only
        # its independent positions, then restore the model's batch layout.
        rotary = tuple(
            value.reshape(*cache_position.shape, -1)
            for value in base.rotary_emb(hidden, cache_position.reshape(1, -1))
        )
        for layer in base.layers:
            hidden = layer(
                hidden,
                attention_mask=None,
                position_ids=cache_position,
                past_key_value=cache,
                cache_position=cache_position,
                position_embeddings=rotary,
                use_cache=True,
            )[0]
        hidden = base.norm(hidden)
        counts = query_lengths.to(device=hidden.device, dtype=torch.int64)
        indices = (counts - 1).clamp_min(0)
        last = hidden.gather(1, indices[:, None, None].expand(-1, 1, hidden.shape[-1]))
        logits = self.model.lm_head(last)[:, -1, :]
        return logits.masked_fill(counts[:, None] == 0, 0)
