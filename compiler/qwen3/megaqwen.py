"""Qwen3-0.6B MegaQwen operator registration and decode export adapter.

The export wrapper retains canonical parameters and preserves the model ABI,
passing weights, KV and scratch explicitly to the shared native operator.
"""

import os
from pathlib import Path

import torch

from compiler.common.artifacts import default_build_dir, require

MEGAQWEN_LIBRARY = "libxinfer_megaqwen.so"
MEGAQWEN_C_SHIM = """AOTITorchError aoti_torch_cuda_megaqwen_decode(
    AtenTensorHandle ids, const AtenTensorHandle* weights, int64_t weights_len,
    const AtenTensorHandle* kv, int64_t kv_len, AtenTensorHandle cos, AtenTensorHandle sin,
    const AtenTensorHandle* workspace, int64_t workspace_len,
    AtenTensorHandle extent, AtenTensorHandle* output)"""
VOCAB_SIZE = 151936
LAYER_WEIGHTS = (
    ("input_layernorm.weight", (1024,)),
    ("self_attn.original.q_proj.weight", (2048, 1024)),
    ("self_attn.original.k_proj.weight", (1024, 1024)),
    ("self_attn.original.v_proj.weight", (1024, 1024)),
    ("self_attn.original.q_norm.weight", (128,)),
    ("self_attn.original.k_norm.weight", (128,)),
    ("self_attn.original.o_proj.weight", (1024, 2048)),
    ("post_attention_layernorm.weight", (1024,)),
    ("mlp.gate_proj.weight", (3072, 1024)),
    ("mlp.up_proj.weight", (3072, 1024)),
    ("mlp.down_proj.weight", (1024, 3072)),
)
WEIGHT_SPECS = (
    ("model.model.embed_tokens.weight", (VOCAB_SIZE, 1024)),
    ("model.model.norm.weight", (1024,)),
    ("model.lm_head.weight", (VOCAB_SIZE, 1024)),
    *(
        (f"model.model.layers.{layer}.{name}", shape)
        for layer in range(28)
        for name, shape in LAYER_WEIGHTS
    ),
)
WORKSPACE_SHAPES = (
    (1024,),
    (1024,),
    (1024,),
    (2048,),
    (1024,),
    (1024,),
    (2048,),
    (3072,),
    (1024,),
    (16, 4),
    (16, 4),
    (16, 4, 128),
)
_loaded = None


def fake_decode(ids, weights, kv, cos, sin, workspace, extent):
    """Metadata-only contract; data values and raw addresses are never inspected."""
    torch._check(ids.shape == (1, 1) and ids.dtype == torch.int64)
    torch._check(ids.device.type == "cuda")
    torch._check(len(weights) == len(WEIGHT_SPECS) and len(kv) == 56)
    torch._check(len(workspace) == len(WORKSPACE_SHAPES))
    capacity = kv[0].shape[2]
    torch._check(capacity >= 512)
    torch._check(capacity <= 40960)
    for tensor, (_, shape) in zip(weights, WEIGHT_SPECS):
        torch._check(tensor.shape == shape and tensor.dtype == torch.bfloat16)
        torch._check(tensor.device == ids.device and tensor.is_contiguous())
    for tensor in (*kv, cos, sin):
        torch._check(tensor.dtype == torch.bfloat16 and tensor.device == ids.device)
        torch._check(tensor.is_contiguous())
    for tensor in kv:
        torch._check(tensor.shape == (1, 8, capacity, 128))
    torch._check(cos.shape == sin.shape == (capacity, 128))
    for index, (tensor, shape) in enumerate(zip(workspace, WORKSPACE_SHAPES)):
        dtype = torch.bfloat16 if index == 0 else torch.float32
        torch._check(tensor.shape == shape and tensor.dtype == dtype)
        torch._check(tensor.device == ids.device and tensor.is_contiguous())
    torch._check(extent.device.type == "cpu" and extent.dtype == torch.uint8)
    torch._check(extent.dim() == 1)
    torch._check(extent.shape[0] >= 2)
    torch._check(extent.shape[0] <= capacity + 1)
    return ids.new_empty((1, VOCAB_SIZE), dtype=torch.bfloat16)


def load_megaqwen_ops(library=None):
    """Load the C++ operator once and attach its export/FakeTensor contract."""
    global _loaded
    if library is None and _loaded is not None:
        return
    selected = library or os.environ.get("XINFER_MEGAQWEN_LIBRARY")
    library = Path(selected or default_build_dir("qwen3") / MEGAQWEN_LIBRARY).resolve()
    if _loaded is not None:
        require(_loaded == library, "Cannot load two different MegaQwen libraries")
        return
    require(library.is_file(), f"Build the native MegaQwen library first: {library}")
    torch.ops.load_library(str(library))
    torch.library.register_fake("xinfer::megaqwen_decode")(fake_decode)
    _loaded = library


def megaqwen_library():
    """Return the implementation selected by the eager/export adapter."""
    require(_loaded is not None, "Load the native MegaQwen library before export")
    return _loaded


def megaqwen_c_shims():
    """Declare the direct native call used by the compiled decode wrapper."""
    return {torch.ops.xinfer.megaqwen_decode.default: [MEGAQWEN_C_SHIM]}


def allocate_workspace(device="cuda"):
    """Allocate per-runtime scratch; every tensor is an explicit mutable input."""
    return [
        torch.empty(shape, device=device, dtype=torch.bfloat16 if index == 0 else torch.float32)
        for index, shape in enumerate(WORKSPACE_SHAPES)
    ]


def rope_tables(capacity, device="cuda"):
    """Construct RoPE tables in CPU FP32, then convert them to BF16 on the device."""
    require(type(capacity) is int and 512 <= capacity <= 40960, "Invalid MegaQwen capacity")
    exponents = torch.arange(0, 128, 2, dtype=torch.float32) / 128.0
    inverse_frequency = torch.reciprocal(torch.pow(1000000.0, exponents))
    frequencies = torch.arange(capacity, dtype=torch.float32).unsqueeze(1) * inverse_frequency
    return tuple(
        torch.cat((value, value), dim=1).to(dtype=torch.bfloat16, device=device).contiguous()
        for value in (torch.cos(frequencies), torch.sin(frequencies))
    )


def validate_geometry(config, capacity):
    def field(name, default=None):
        return (
            config.get(name, default)
            if isinstance(config, dict)
            else getattr(config, name, default)
        )

    require(type(capacity) is int and 512 <= capacity <= 40960, "Invalid MegaQwen capacity")
    expected = {
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": VOCAB_SIZE,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
    }
    for name, value in expected.items():
        require(field(name) == value, f"Unsupported MegaQwen model configuration: {name}")
    require(
        field("rope_scaling") is None and field("attention_bias", False) is False,
        "MegaQwen requires unscaled RoPE and bias-free attention",
    )


def _adapt_flash_model(model):
    from compiler.qwen3.adapters.qwen3_flash import FlashAttention, FlashStep

    if all(isinstance(layer.self_attn, FlashAttention) for layer in model.model.layers):
        return model
    return FlashStep(model).model


class MegaDecodeStep(torch.nn.Module):
    """Shipping ABI-1 decode graph with shared parameters and graph-local scratch.

    The optional compiled profile uses MegaQwen for every singleton, including
    position zero. Multi-token prefill remains a separate FlashStep graph.
    """

    def __init__(self, model, capacity):
        super().__init__()
        validate_geometry(model.config, capacity)
        load_megaqwen_ops()
        self.model = _adapt_flash_model(model)
        self.model.requires_grad_(False)
        device = next(self.model.parameters()).device
        cos, sin = rope_tables(capacity, device)
        self.register_buffer("megaqwen_cos", cos)
        self.register_buffer("megaqwen_sin", sin)
        self.eval()

    def forward(self, input_ids, cache_position, kv, kv_extent):
        # Preserve the canonical .model parameter names used by FlashStep and
        # the shared-weight archive; retaining these tensors never copies them.
        weights = [self.get_parameter(name) for name, _ in WEIGHT_SPECS]
        workspace = allocate_workspace(input_ids.device)
        return torch.ops.xinfer.megaqwen_decode(
            input_ids, weights, list(kv), self.megaqwen_cos, self.megaqwen_sin, workspace, kv_extent
        )
