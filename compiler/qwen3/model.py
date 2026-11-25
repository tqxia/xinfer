"""Qwen3 model loading, caller-owned KV state and explicit inference steps."""

import torch
from transformers import AutoModelForCausalLM

from compiler.common.kernels.flash_attention.attention import batched_attention_metadata
from compiler.common.context import resolve_context_length
from compiler.common.kernels.kv.kv_cache import append_kv_batched


MAX_PREFILL = 512
MAX_BATCH_SIZE = 8


def load_model(model_dir):
    return (
        AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            dtype="auto",
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


class KVCache:
    """Caller-owned KV tensors with per-row positions and contiguous valid lengths."""

    def __init__(self, tensors, position, kv_lengths, query_lengths):
        self.tensors = tensors
        self.position = position
        self.offsets, self.cuda_offsets, self.blocks = batched_attention_metadata(
            kv_lengths, query_lengths, position.device
        )

    def update(self, key, value, layer_idx, cache_kwargs=None):
        k, v = self.tensors[2 * layer_idx : 2 * layer_idx + 2]
        append_kv_batched(key, value, self.position, self.cuda_offsets[0], k, v)
        return k, v


class ModelStep(torch.nn.Module):
    """Install adapters once; drive unchanged HF decoder layers with tensor state.

    This small loop avoids HF's automatic mask construction. Causality lives in the
    prefix op; arbitrary masks are deliberately outside this inference interface.
    """

    def __init__(self, model):
        super().__init__()
        from compiler.qwen3.adapters.attention import install_attention

        install_attention(model)
        self.model = model
        self.eval()

    def forward(self, input_ids, cache_position, kv, kv_lengths, query_lengths):
        cache = KVCache(kv, cache_position, kv_lengths, query_lengths)
        base = self.model.model
        hidden = base.embed_tokens(input_ids)
        # HF's batched RoPE matmul specializes B=1 during export. Flatten only
        # its independent positions, then restore the model's batch layout.
        rotary = tuple(
            value.reshape(*cache_position.shape, -1)
            for value in base.rotary_emb(hidden, cache_position.reshape(1, -1))
        )
        for layer in base.layers:
            hidden = layer(
                hidden,
                attention_mask=None,
                position_ids=cache_position,
                past_key_values=cache,
                cache_position=cache_position,
                position_embeddings=rotary,
                use_cache=True,
            )
        hidden = base.norm(hidden)
        counts = (cache.cuda_offsets[0, 1:] - cache.cuda_offsets[0, :-1]).to(torch.int64)
        indices = (counts - 1).clamp_min(0)
        last = hidden.gather(1, indices[:, None, None].expand(-1, 1, hidden.shape[-1]))
        logits = self.model.lm_head(last)[:, -1, :]
        return logits.masked_fill(counts[:, None] == 0, 0)
