"""Qwen3.5 gated full attention over the populated caller-owned KV prefix."""

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, apply_rotary_pos_emb

from compiler.common.attention import load_attention_ops
from compiler.qwen35.model import TextCache


class FlashAttention(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        if type(original) is not Qwen3_5Attention:
            raise ValueError("Expected Qwen3.5 full attention")
        config = original.config
        if (config.num_attention_heads, config.num_key_value_heads, original.head_dim) != (
            8,
            2,
            256,
        ):
            raise ValueError("The native Flash op targets Qwen3.5-0.8B head dimensions")
        self.original = original

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        if self.training or kwargs.get("output_attentions", False):
            raise ValueError("Flash adapter supports inference without attention weights")
        if (
            attention_mask is not None
            or not isinstance(past_key_values, TextCache)
            or past_key_values.extent is None
        ):
            raise ValueError(
                "Flash adapter requires an unpadded TextCache with its valid-prefix extent"
            )
        a = self.original
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, a.head_dim)
        query, gate = torch.chunk(
            a.q_proj(hidden_states).view(*input_shape, -1, a.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)
        query = a.q_norm(query.reshape(hidden_shape)).transpose(1, 2)
        key = a.k_norm(a.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value = a.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        key, value = past_key_values.update(key, value, a.layer_idx)
        output = torch.ops.xinfer.flash_attention(
            query.contiguous(), key, value, past_key_values.extent, a.scaling
        )
        output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        output = output * torch.sigmoid(gate)
        return a.o_proj(output), None


def install_attention(model):
    """Reuse original HF weights and keep repeated TextStep construction safe."""
    replacements = []
    for kind, layer in zip(model.config.layer_types, model.model.layers):
        if kind == "full_attention" and not isinstance(layer.self_attn, FlashAttention):
            replacement = FlashAttention(layer.self_attn)
            replacement.train(layer.self_attn.training)
            replacements.append((layer, replacement))
    load_attention_ops(model="qwen35")
    for layer, replacement in replacements:
        layer.self_attn = replacement
