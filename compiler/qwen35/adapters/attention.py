"""Qwen3.5 gated attention over independent, right-padded KV prefixes."""

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, apply_rotary_pos_emb

from compiler.common.kernels.flash_attention.attention import load_attention_ops
from compiler.qwen35.model import HybridCache


class FlashAttention(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        if type(original) is not Qwen3_5Attention:
            raise ValueError("Expected Qwen3.5 full attention")
        config = original.config
        heads, kv_heads, dim = (
            config.num_attention_heads,
            config.num_key_value_heads,
            original.head_dim,
        )
        if heads <= 0 or kv_heads <= 0 or heads % kv_heads:
            raise ValueError("Query heads must be a positive multiple of KV heads")
        if dim < 8 or dim > 256 or dim % 8:
            raise ValueError("Flash head dimension must be a multiple of 8 in 8..256")
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
        if attention_mask is not None or not isinstance(past_key_values, HybridCache):
            raise ValueError("Flash adapter requires HybridCache with per-row lengths")
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
        output = torch.ops.xinfer.batched_flash_attention(
            query,
            key,
            value,
            past_key_values.offsets,
            past_key_values.cuda_offsets,
            past_key_values.blocks,
            a.scaling,
        )
        output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        output = output * torch.sigmoid(gate)
        return a.o_proj(output), None


def install_attention(model):
    """Reuse original HF weights and keep repeated ModelStep construction safe."""
    replacements = []
    for kind, layer in zip(model.config.layer_types, model.model.layers):
        if kind == "full_attention" and not isinstance(layer.self_attn, FlashAttention):
            replacement = FlashAttention(layer.self_attn)
            # Inherit the wrapped layer's mode so eval models stay eval and train mode still fails.
            replacement.train(layer.self_attn.training)
            replacements.append((layer, replacement))
    load_attention_ops()
    for layer, replacement in replacements:
        layer.self_attn = replacement
