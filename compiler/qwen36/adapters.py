"""Small inference adapters retaining HF's quantized projection/expert storage."""

import torch
from transformers.integrations.finegrained_fp8 import (
    FP8Experts as HFExperts,
    FP8Linear as HFLinear,
)

from compiler.common import fp8
from compiler.common.artifacts import require


def _format(module):
    require(
        tuple(module.block_size or ()) == (128, 128)
        and module.activation_scheme == "dynamic"
        and not module.has_bias,
        "FP8 adapters require bias-free dynamic 128x128 block quantization",
    )


class FP8Linear(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        require(type(original) is HFLinear, "Expected the ordinary HF FP8Linear module")
        _format(original)
        require(
            original.bias is None and original.activation_scale is None,
            "Unsupported FP8 bias/scale",
        )
        fp8._weight(original.weight, original.weight_scale_inv, 2)
        self.original = original
        self.train(original.training)

    def forward(self, hidden_states):
        require(not self.training, "FP8 adapters support inference only")
        module = self.original
        return fp8.linear(hidden_states, module.weight, module.weight_scale_inv)


class FP8Experts(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        require(type(original) is HFExperts, "Expected the ordinary HF FP8Experts module")
        _format(original)
        require(
            original.has_gate
            and original.config.hidden_act == "silu"
            and original.swiglu_alpha is None
            and original.limit is None
            and not getattr(original, "is_transposed", False)
            and getattr(original, "is_concatenated", True)
            and not getattr(original, "_is_expert_parallel", False),
            "FP8 experts require local concatenated SiLU gate/up weights without clipping",
        )
        fp8._weight(original.gate_up_proj, original.gate_up_proj_scale_inv, 3)
        fp8._weight(original.down_proj, original.down_proj_scale_inv, 3)
        require(
            original.gate_up_proj.shape
            == (original.num_experts, 2 * original.intermediate_dim, original.hidden_dim)
            and original.down_proj.shape
            == (original.num_experts, original.hidden_dim, original.intermediate_dim),
            "HF expert geometry differs from its quantized weights",
        )
        self.original = original
        self.train(original.training)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        require(not self.training, "FP8 adapters support inference only")
        module = self.original
        return fp8.experts(
            hidden_states,
            top_k_index,
            top_k_weights,
            module.gate_up_proj,
            module.gate_up_proj_scale_inv,
            module.down_proj,
            module.down_proj_scale_inv,
        )


def install_fp8(model):
    """Replace only HF FP8 projections/experts; preserve routers and BF16 modules."""
    replacements = []

    def visit(parent):
        for name, module in parent.named_children():
            if isinstance(module, (FP8Linear, FP8Experts)):
                continue
            if isinstance(module, HFExperts):
                replacements.append((parent, name, FP8Experts(module)))
            elif isinstance(module, HFLinear):
                replacements.append((parent, name, FP8Linear(module)))
            else:
                visit(module)

    visit(model)
    # An invalid module or failed preload leaves the original hierarchy intact.
    if replacements:
        fp8.load_fp8_kernels()
    counts = dict(linear=0, experts=0)
    for parent, name, replacement in replacements:
        setattr(parent, name, replacement)
        counts["experts" if isinstance(replacement, FP8Experts) else "linear"] += 1
    return counts
