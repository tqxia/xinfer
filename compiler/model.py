"""Tensor-only entry point around the original Hugging Face Qwen3 model."""

from types import MethodType

import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


CACHE_LENGTH = 1024
MAX_PROMPT = 512


def flash_attention_forward(module, query, key, value, attention_mask,
                            dropout=0.0, scaling=None, **kwargs):
    """Native GQA FlashAttention, with causal alignment at the end of the KV prefix.

    Use the same ATen operator as PyTorch's causal_lower_right SDPA dispatcher.
    Unlike public SDPA's is_causal=True (upper-left alignment), this operator's
    causal flag is lower-right: it handles initial prefill, appended chunks and
    single-token decode uniformly. It also avoids shape-equality branches in the
    CausalBias dispatcher that would specialize a dynamic AOTI export.
    This internal operator is covered by the pinned Torch/native ABI contract.
    """
    if attention_mask is not None or dropout != 0.0 or module.sliding_window is not None:
        raise ValueError("Flash path requires unpadded inference without sliding-window attention")
    output = torch.ops.aten._scaled_dot_product_flash_attention.default(
        query, key, value, 0.0, True, return_debug_mask=False, scale=scaling)[0]
    return output.transpose(1, 2).contiguous(), None


def _prefix_causal_mask(self, attention_mask, *args, **kwargs):
    if attention_mask is not None:
        raise ValueError("Flash path takes a valid KV prefix, not a padding/dense mask")
    # Causality is implemented inside FlashAttention. No dense mask is needed.
    return None


def cache_extent(valid_length):
    """Only this CPU tensor's shape is read; +1 avoids AOTI's size-one guards."""
    return torch.empty(valid_length + 1, dtype=torch.uint8)


class TensorCache(Cache):
    """Adapt caller-owned (K0, V0, K1, V1, ...) tensors to HF's cache API."""

    def __init__(self, tensors, position, valid_length=None):
        super().__init__()
        self.tensors = tensors
        self.position = position
        self.valid_length = valid_length

    def get_seq_length(self, layer_idx=0):
        return self.position[0]

    def get_max_cache_shape(self):
        return self.tensors[0].shape[2]

    def update(self, key, value, layer_idx, cache_kwargs=None):
        k, v = self.tensors[2 * layer_idx:2 * layer_idx + 2]
        position = cache_kwargs["cache_position"]
        k.index_copy_(2, position, key)
        v.index_copy_(2, position, value)
        if self.valid_length is not None:
            return k[:, :, :self.valid_length, :], v[:, :, :self.valid_length, :]
        return k, v


class QwenStep(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.flash = model.config._attn_implementation == "xinfer_flash"

    def forward(self, input_ids, cache_position, kv, kv_extent=None):
        # Absolute positions handle both multi-token prefill and single-token decode.
        mask = None
        if not self.flash:
            columns = torch.arange(kv[0].shape[2], device=input_ids.device)
            future = columns.unsqueeze(0) > cache_position.unsqueeze(1)
            mask = torch.zeros_like(future, dtype=kv[0].dtype)
            mask = mask.masked_fill(future, torch.finfo(kv[0].dtype).min)[None, None]
        result = self.model(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=cache_position.unsqueeze(0),
            past_key_values=TensorCache(kv, cache_position, kv_extent.shape[0] - 1 if self.flash else None),
            cache_position=cache_position,
            use_cache=True,
            logits_to_keep=1,
        )
        return result.logits[:, -1, :]


def load_model(model_dir, attention_backend="efficient"):
    if attention_backend not in ("efficient", "flash"):
        raise ValueError(f"Unknown attention backend: {attention_backend}")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval().to("cuda")
    if attention_backend == "flash":
        ALL_ATTENTION_FUNCTIONS.register("xinfer_flash", flash_attention_forward)
        model.config._attn_implementation = "xinfer_flash"
        model.model._update_causal_mask = MethodType(_prefix_causal_mask, model.model)
    return model


def allocate_cache(config, attention_backend="efficient"):
    # A spare position per head keeps every Flash prefix non-contiguous, even
    # at capacity. Otherwise export guards specialize at the 1024-token boundary.
    storage_length = CACHE_LENGTH + (attention_backend == "flash")
    shape = (1, config.num_key_value_heads, storage_length, config.head_dim)
    return tuple(torch.zeros(shape, dtype=torch.bfloat16, device="cuda")[:, :, :CACHE_LENGTH, :]
                 for _ in range(2 * config.num_hidden_layers))
