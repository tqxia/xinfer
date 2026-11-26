"""Shared Qwen Gated DeltaNet adapter backed by the native in-place CUDA operator."""

import torch

from compiler.common.kernels.gated_delta_net.deltanet import batched_deltanet


class DeltaCore(torch.nn.Module):
    """Explicit hidden/conv/recurrent interface for prefill and decode.

    Reuse original parameters and mutate caller-owned conv/recurrent state in
    place through xinfer::batched_deltanet. Projections and gated normalization
    stay outside the native operator.
    """

    def __init__(self, block):
        super().__init__()
        self.block = block
        if (
            block.num_k_heads != 16
            or block.num_v_heads not in (16, 32, 48)
            or block.head_k_dim != 128
            or block.head_v_dim != 128
        ):
            raise ValueError("Fused DeltaNet requires Hk=16, Hv=16, 32 or 48 and K=V=128")
        if (
            block.conv_kernel_size != 4
            or block.conv1d.bias is not None
            or block.activation != "silu"
        ):
            raise ValueError("Fused DeltaNet requires a bias-free width-4 SiLU convolution")

    def forward(
        self, hidden, conv, state, cu_query=None, cuda_cu_query=None, *, query_lengths=None
    ):
        block = self.block
        batch, tokens, _ = hidden.shape
        core = batched_deltanet(
            block.in_proj_qkv(hidden),
            block.in_proj_b(hidden),
            block.in_proj_a(hidden),
            conv,
            state,
            block.conv1d.weight.squeeze(1),
            block.A_log,
            block.dt_bias,
            cu_query,
            cuda_cu_query,
            query_lengths=query_lengths,
        )
        z = block.in_proj_z(hidden).reshape(-1, block.head_v_dim)
        core = block.norm(core.reshape(-1, block.head_v_dim), z)
        return block.out_proj(core.reshape(batch, tokens, block.value_dim))


class DeltaNet(DeltaCore):
    """HF module interface for the explicit HybridCache used by ModelStep."""

    def forward(self, hidden_states, cache_params, attention_mask=None, **kwargs):
        if attention_mask is not None:
            raise ValueError("DeltaNet uses HybridCache lengths instead of attention masks")
        layer = cache_params.layers[self.block.layer_idx]
        return super().forward(
            hidden_states,
            layer.conv_states[0],
            layer.recurrent_states[0],
            cache_params.offsets[0],
            cache_params.cuda_offsets[0],
        )


def install_delta(model):
    """Reuse HF parameters and replace every DeltaNet core with the native adapter."""
    for kind, layer in zip(model.config.layer_types, model.model.layers):
        if kind == "linear_attention" and not isinstance(layer.linear_attn, DeltaNet):
            layer.linear_attn = DeltaNet(layer.linear_attn)
