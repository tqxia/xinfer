"""Qwen3 FlashAttention replacement and installation using the original HF weights."""

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, apply_rotary_pos_emb

from compiler.common.attention import load_attention_ops
from compiler.qwen3.model import KVCache


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


def install_attention(model):
    """Replace attention modules while retaining their original weights."""
    # Construct all replacements before modifying the model. No weight copies.
    replacements = [FlashAttention(layer.self_attn) for layer in model.model.layers]
    load_attention_ops(model="qwen3")
    for layer, attention in zip(model.model.layers, replacements):
        layer.self_attn = attention
