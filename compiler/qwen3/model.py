"""Load Qwen3 models and adapt caller-owned KV tensors to the HF cache API."""

import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache


CACHE_LENGTH = 1024
MAX_PROMPT = 512


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
        k, v = self.tensors[2 * layer_idx : 2 * layer_idx + 2]
        position = cache_kwargs["cache_position"]
        k.index_copy_(2, position, key)
        v.index_copy_(2, position, value)
        return k, v


def load_model(model_dir):
    return (
        AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .eval()
        .to("cuda")
    )


def allocate_cache(config):
    shape = (1, config.num_key_value_heads, CACHE_LENGTH, config.head_dim)
    return tuple(
        torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
        for _ in range(2 * config.num_hidden_layers)
    )
