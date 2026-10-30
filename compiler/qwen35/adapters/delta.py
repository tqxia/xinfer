"""Qwen3.5 DeltaNet adapter with a Triton recurrent core."""

import torch

from compiler.qwen35.kernels.recurrent import fused


class DeltaCore(torch.nn.Module):
    """Explicit hidden/conv/recurrent interface for prefill and decode.

    Reuse original parameters with shape-dispatched prefill and fused decode.
    Projections and gated normalization stay outside the conditional kernels.
    """

    def __init__(self, block):
        super().__init__()
        self.block = block
        if (block.num_k_heads, block.num_v_heads, block.head_k_dim, block.head_v_dim) != (16, 16, 128, 128):
            raise ValueError("The Triton baseline requires Hq=Hk=Hv=16 and K=V=128")
        if block.conv_kernel_size != 4 or block.conv1d.bias is not None or block.activation != "silu":
            raise ValueError("Fused DeltaNet requires a bias-free width-4 SiLU convolution")

    def forward(self, hidden, conv, state):
        block = self.block
        batch, tokens, _ = hidden.shape
        core, next_conv, next_recurrent = fused(
            block.in_proj_qkv(hidden), block.in_proj_b(hidden), block.in_proj_a(hidden),
            conv, state, block.conv1d.weight.squeeze(1), block.A_log, block.dt_bias)
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
