"""Load Qwen3 models and allocate caller-owned KV tensors."""

import torch
from transformers import AutoModelForCausalLM

from compiler.common.context import resolve_context_length


MAX_PREFILL = 512
MAX_BATCH_SIZE = 8


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


def allocate_cache(config, capacity=None, batch_size=1):
    capacity = resolve_context_length(config, capacity)
    shape = (batch_size, config.num_key_value_heads, capacity, config.head_dim)
    return tuple(
        torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
        for _ in range(2 * config.num_hidden_layers)
    )
