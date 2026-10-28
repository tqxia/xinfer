"""Qwen3.5 DeltaNet adapter with a Triton recurrent core."""

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import causal_conv1d_fn, causal_conv1d_update

from compiler.qwen35.kernels.recurrent import recurrent


class DeltaCore(torch.nn.Module):
    """Explicit hidden/conv/recurrent interface for prefill and decode.

    Reuse all original parameters. The GDN core uses a Triton operator;
    projections, convolution and gated normalization remain visible to export.
    """

    def __init__(self, block):
        super().__init__()
        self.block = block
        if (block.num_k_heads, block.num_v_heads, block.head_k_dim, block.head_v_dim) != (16, 16, 128, 128):
            raise ValueError("The Triton baseline requires Hq=Hk=Hv=16 and K=V=128")

    def forward(self, hidden, conv, state):
        block = self.block
        batch, tokens, _ = hidden.shape
        mixed = block.in_proj_qkv(hidden).transpose(1, 2)
        if tokens == 1:
            # HF's decode convolution mutates its state; preserve our functional ABI.
            next_conv = conv.clone()
            mixed = causal_conv1d_update(mixed, next_conv, block.conv1d.weight.squeeze(1),
                                        block.conv1d.bias, block.activation)
        else:
            combined = torch.cat((conv, mixed), dim=-1)
            next_conv = combined[..., -block.conv_kernel_size:].contiguous()
            mixed = causal_conv1d_fn(combined, block.conv1d.weight.squeeze(1),
                                    block.conv1d.bias, activation=block.activation)
            mixed = mixed[..., -tokens:]
        mixed = mixed.transpose(1, 2)
        q, k, v = mixed.split((block.key_dim, block.key_dim, block.value_dim), dim=-1)
        q = q.reshape(batch, tokens, block.num_k_heads, block.head_k_dim)
        k = k.reshape(batch, tokens, block.num_k_heads, block.head_k_dim)
        v = v.reshape(batch, tokens, block.num_v_heads, block.head_v_dim)
        beta = block.in_proj_b(hidden).sigmoid()
        g = -block.A_log.float().exp() * F.softplus(block.in_proj_a(hidden).float() + block.dt_bias)
        core, next_recurrent = recurrent(q, k, v, g, beta, state)
        z = block.in_proj_z(hidden).reshape(-1, block.head_v_dim)
        core = block.norm(core.reshape(-1, block.head_v_dim), z)
        output = block.out_proj(core.reshape(batch, tokens, block.value_dim))
        return output, next_conv, next_recurrent


class DeltaNet(DeltaCore):
    """HF module interface for the explicit TextCache used by TextStep."""

    def forward(self, hidden_states, cache_params, attention_mask=None, **kwargs):
        if attention_mask is not None:
            raise ValueError("DeltaNet expects unpadded token inputs")
        layer = cache_params.layers[self.block.layer_idx]
        output, conv, recurrent = super().forward(
            hidden_states, layer.conv_states[0], layer.recurrent_states[0])
        layer.conv_states[0] = conv
        cache_params.update_recurrent_state(recurrent, self.block.layer_idx)
        return output


def install_delta(model):
    """Reuse HF parameters and replace every DeltaNet core with Triton."""
    for kind, layer in zip(model.config.layer_types, model.model.layers):
        if kind == "linear_attention" and not isinstance(layer.linear_attn, DeltaNet):
            layer.linear_attn = DeltaNet(layer.linear_attn)
