"""Unmodified HF execution with hybrid KV and recurrent caches."""

import torch
from transformers import DynamicCache, StaticCache

from compiler.common.artifacts import require


class HybridHFStage:
    """Expose StaticCache and padded DynamicCache through one runtime interface."""

    def __init__(self, model, specs, capacity, *, dynamic=False, max_prefill=512):
        self.model = model
        self.vocab_size = model.config.vocab_size
        self.dtype = next(model.parameters()).dtype
        self.specs = specs
        self.dynamic = dynamic
        self.capacity = capacity
        self.max_prefill = max_prefill
        self.cache = None
        self.position = 0

    def reset(self, fill=0):
        if self.dynamic:
            require(fill == 0, "DynamicCache control requires an unpoisoned initial cache")
            self.cache = DynamicCache(config=self.model.config)
        else:
            self.cache = StaticCache(self.model.config, max_cache_len=self.capacity)
        self.position = 0
        if fill:
            config = self.model.config
            sample = torch.zeros(
                (1, config.num_key_value_heads, 1, config.head_dim),
                device="cuda",
                dtype=self.dtype,
            )
            for kind, layer in zip(config.layer_types, self.cache.layers):
                if kind == "full_attention":
                    layer.lazy_initialization(sample, sample)
                    layer.keys.fill_(fill)
                    layer.values.fill_(fill)

    @property
    def state(self):
        tensors = []
        for kind, layer in zip(self.model.config.layer_types, self.cache.layers):
            tensors.extend(
                (layer.conv_states[0], layer.recurrent_states[0])
                if kind == "linear_attention"
                else (layer.keys, layer.values)
            )
        if not self.dynamic:
            return tensors
        return [
            torch.nn.functional.pad(tensor, (0, 0, 0, self.capacity - self.position))
            if spec["kind"] in ("key", "value")
            else tensor
            for spec, tensor in zip(self.specs, tensors)
        ]

    def step(self, tokens):
        length = len(tokens)
        require(1 <= length <= 512, "Expected 1..512 input tokens")
        require(self.position + length <= self.capacity, "Context capacity exceeded")
        ids = torch.tensor([tokens], device="cuda")
        positions = torch.arange(self.position, self.position + length, device="cuda")
        width = self.position + length if self.dynamic else self.capacity
        mask = torch.full(
            (1, 1, length, width),
            torch.finfo(self.dtype).min,
            device="cuda",
            dtype=self.dtype,
        ).triu(self.position + 1)
        logits = self.model(
            ids,
            position_ids=positions[None, :],
            attention_mask={"full_attention": mask, "linear_attention": None},
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=1,
        ).logits[:, 0, :]
        self.position += length
        return logits

    def close(self):
        self.cache = None
        self.model = None
