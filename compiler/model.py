"""Tensor-only entry point around the original Hugging Face Qwen3 model."""

import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache


CACHE_LENGTH = 1024
MAX_PROMPT = 512


def cache_extent(valid_length):
    """Only this CPU tensor's shape is read; +1 avoids AOTI's size-one guards."""
    return torch.empty(valid_length + 1, dtype=torch.uint8)


class TensorCache(Cache):
    """Adapt caller-owned (K0, V0, K1, V1, ...) tensors to HF's cache API."""

    def __init__(self, tensors, position):
        super().__init__()
        self.tensors = tensors
        self.position = position

    def get_seq_length(self, layer_idx=0):
        return self.position[0]

    def get_max_cache_shape(self):
        return self.tensors[0].shape[2]

    def update(self, key, value, layer_idx, cache_kwargs=None):
        k, v = self.tensors[2 * layer_idx:2 * layer_idx + 2]
        position = cache_kwargs["cache_position"]
        k.index_copy_(2, position, key)
        v.index_copy_(2, position, value)
        return k, v


class QwenStep(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, cache_position, kv):
        # The efficient path attends to the full cache with an explicit mask.
        columns = torch.arange(kv[0].shape[2], device=input_ids.device)
        future = columns.unsqueeze(0) > cache_position.unsqueeze(1)
        mask = torch.zeros_like(future, dtype=kv[0].dtype)
        mask = mask.masked_fill(future, torch.finfo(kv[0].dtype).min)[None, None]
        result = self.model(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=cache_position.unsqueeze(0),
            past_key_values=TensorCache(kv, cache_position),
            cache_position=cache_position,
            use_cache=True,
            logits_to_keep=1,
        )
        return result.logits[:, -1, :]


def load_model(model_dir):
    return AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval().to("cuda")


def allocate_cache(config):
    shape = (1, config.num_key_value_heads, CACHE_LENGTH, config.head_dim)
    return tuple(torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
                 for _ in range(2 * config.num_hidden_layers))
