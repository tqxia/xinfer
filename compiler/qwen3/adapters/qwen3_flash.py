"""Qwen3-0.6B FlashAttention through child-module replacement, without FX passes.

The tensor entry point supports only batch-one, unpadded, contiguous appends to
the populated cache prefix. The C++ runtime validates positions and capacity.
Weights are loaded with HF first; wrappers reuse the original modules in place.
"""

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, apply_rotary_pos_emb

from compiler.qwen3.custom_ops.attention import load_attention_ops
from compiler.qwen3.model import TensorCache


class PrefixCache(TensorCache):
    def __init__(self, tensors, position, extent):
        super().__init__(tensors, position)
        self.extent = extent


class FlashAttention(torch.nn.Module):
    """Keep HF's weights and arithmetic around one explicit custom-op call."""

    def __init__(self, original):
        super().__init__()
        if type(original) is not Qwen3Attention or original.sliding_window is not None:
            raise ValueError("Expected full-attention Qwen3Attention")
        config = original.config
        if (config.num_attention_heads, config.num_key_value_heads, original.head_dim) != (16, 8, 128):
            raise ValueError("The native Flash op targets Qwen3-0.6B head dimensions")
        self.original = original

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_value=None, cache_position=None, **kwargs):
        if self.training or kwargs.get("output_attentions", False):
            raise ValueError("Flash adapter supports inference without attention weights")
        if attention_mask is not None:
            raise ValueError("Flash adapter accepts only the unpadded PrefixCache contract")
        if not isinstance(past_key_value, PrefixCache) or cache_position is not past_key_value.position:
            raise ValueError("Flash adapter requires PrefixCache with the same position tensor")
        a = self.original
        input_shape = hidden_states.shape[:-1]
        head_shape = (*input_shape, -1, a.head_dim)
        q = a.q_norm(a.q_proj(hidden_states).view(head_shape)).transpose(1, 2)
        k = a.k_norm(a.k_proj(hidden_states).view(head_shape)).transpose(1, 2)
        v = a.v_proj(hidden_states).view(head_shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k, v = past_key_value.update(k, v, a.layer_idx,
                                    {"sin": sin, "cos": cos, "cache_position": cache_position})
        # Preserve HF SDPA's query layout for a controlled backend comparison.
        output = torch.ops.xinfer.flash_attention(q.contiguous(), k, v, past_key_value.extent, a.scaling)
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
        load_attention_ops()
        for layer, attention in zip(model.model.layers, replacements):
            layer.self_attn = attention
        self.model = model
        self.eval()

    def forward(self, input_ids, cache_position, kv, kv_extent):
        cache = PrefixCache(kv, cache_position, kv_extent)
        base = self.model.model
        hidden = base.embed_tokens(input_ids)
        positions = cache_position.unsqueeze(0)
        rotary = base.rotary_emb(hidden, positions)
        for layer in base.layers:
            hidden = layer(hidden, attention_mask=None, position_ids=positions,
                           past_key_value=cache, cache_position=cache_position,
                           position_embeddings=rotary, use_cache=True)[0]
        return self.model.lm_head(base.norm(hidden)[:, -1:, :])[:, -1, :]
