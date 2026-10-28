"""Experimental prefill adapter: native chunk rule, ordinary HF outer operations."""

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import causal_conv1d_fn

_fake_registered = False


def load_library(path):
    """Load the same dispatcher library used by the standalone C++ runner."""
    torch.ops.load_library(str(path.resolve()))
    global _fake_registered
    if not _fake_registered:
        torch.library.register_fake("xinfer::delta_prefill", _fake_delta)
        _fake_registered = True


def _fake_delta(q, k, v, g, beta, state):
    # The native implementation returns fresh, contiguous tensors.
    return v.new_empty(v.shape), state.new_empty(state.shape)


class NativeDeltaPrefill(torch.nn.Module):
    """Single-block T>=2 interface: hidden, convolution state, recurrent state.

    Reuse all original parameters. Only the chunk rule becomes an opaque op;
    projections, convolution and gated normalization remain visible to export.
    """

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden, conv, recurrent):
        block = self.block
        batch, tokens, _ = hidden.shape
        mixed = block.in_proj_qkv(hidden).transpose(1, 2)
        combined = torch.cat((conv, mixed), dim=-1)
        next_conv = combined[..., -block.conv_kernel_size:].contiguous()
        mixed = causal_conv1d_fn(combined, block.conv1d.weight.squeeze(1),
                                block.conv1d.bias, activation=block.activation)
        mixed = mixed[..., -tokens:].transpose(1, 2)
        q, k, v = mixed.split((block.key_dim, block.key_dim, block.value_dim), dim=-1)
        q = q.reshape(batch, tokens, block.num_k_heads, block.head_k_dim)
        k = k.reshape(batch, tokens, block.num_k_heads, block.head_k_dim)
        v = v.reshape(batch, tokens, block.num_v_heads, block.head_v_dim)
        if block.num_v_heads != block.num_k_heads:
            repeats = block.num_v_heads // block.num_k_heads
            q, k = (x.repeat_interleave(repeats, dim=2) for x in (q, k))
        beta = block.in_proj_b(hidden).sigmoid()
        g = -block.A_log.float().exp() * F.softplus(block.in_proj_a(hidden).float() + block.dt_bias)
        core, next_recurrent = torch.ops.xinfer.delta_prefill(q, k, v, g, beta, recurrent)
        z = block.in_proj_z(hidden).reshape(-1, block.head_v_dim)
        core = block.norm(core.reshape(-1, block.head_v_dim), z)
        output = block.out_proj(core.reshape(batch, tokens, block.value_dim))
        return output, next_conv, next_recurrent


class NativeDeltaNet(NativeDeltaPrefill):
    """HF module interface for the explicit TextCache used by TextStep."""

    def forward(self, hidden_states, cache_params, attention_mask=None, **kwargs):
        if hidden_states.shape[1] == 1:
            return self.block(hidden_states, cache_params=cache_params,
                              attention_mask=attention_mask, **kwargs)
        if attention_mask is not None:
            raise ValueError("Native DeltaNet prefill expects unpadded token inputs")
        layer = cache_params.layers[self.block.layer_idx]
        output, conv, recurrent = super().forward(
            hidden_states, layer.conv_states[0], layer.recurrent_states[0])
        layer.conv_states[0] = conv
        cache_params.update_recurrent_state(recurrent, self.block.layer_idx)
        return output


def install_native_delta(model):
    """Replace selected modules in place; preserve all original parameters."""
    for kind, layer in zip(model.config.layer_types, model.model.layers):
        if kind == "linear_attention" and not isinstance(layer.linear_attn, NativeDeltaNet):
            layer.linear_attn = NativeDeltaNet(layer.linear_attn)
