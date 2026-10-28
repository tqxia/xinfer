"""HF and Python AOTI runtimes with explicit state and package-driven dispatch."""

import json
import torch
from transformers import StaticCache

from compiler.qwen35.config import CACHE_LENGTH, PREFILL


class HFRuntime:
    def __init__(self, model, chunked=True):
        self.model = model
        self.chunked = chunked
        self.dtype = next(model.parameters()).dtype

    def reset(self, poison=0):
        self.cache = StaticCache(self.model.config, max_cache_len=CACHE_LENGTH)
        self.position = 0
        if poison:
            config = self.model.config
            sample = torch.zeros((1, config.num_key_value_heads, 1, config.head_dim),
                                 device="cuda", dtype=self.dtype)
            for kind, layer in zip(config.layer_types, self.cache.layers):
                if kind == "full_attention":
                    layer.lazy_initialization(sample, sample)
                    layer.keys.fill_(poison)
                    layer.values.fill_(poison)

    @property
    def state(self):
        tensors = []
        for kind, layer in zip(self.model.config.layer_types, self.cache.layers):
            tensors.extend((layer.conv_states[0], layer.recurrent_states[0]) if kind == "linear_attention"
                           else (layer.keys, layer.values))
        return tensors

    def step(self, tokens):
        assert 1 <= len(tokens) <= 512 and self.position + len(tokens) <= CACHE_LENGTH
        offset = 0
        while offset < len(tokens):
            remaining = len(tokens) - offset
            length = (PREFILL if remaining >= PREFILL else 1) if self.chunked else remaining
            ids = torch.tensor([tokens[offset:offset + length]], device="cuda")
            positions = torch.arange(self.position, self.position + length, device="cuda")
            mask = torch.full((1, 1, length, CACHE_LENGTH), torch.finfo(self.dtype).min,
                              device="cuda", dtype=self.dtype).triu(self.position + 1)
            logits = self.model(ids, position_ids=positions[None, :],
                                attention_mask={"full_attention": mask, "linear_attention": None},
                                past_key_values=self.cache, use_cache=True, logits_to_keep=1).logits[:, 0, :]
            self.position += length
            offset += length
        return logits


class PythonRuntime:
    def __init__(self, packages, specs):
        metadata = json.loads((packages / "prefill.json").read_text())
        self.dynamic_prefill = metadata.get("dynamic_prefill", False)
        if self.dynamic_prefill:
            from compiler.common.artifacts import sha256
            from compiler.qwen35.adapters.native_delta import load_library
            library = packages / "libxinfer_delta_ops.so"
            assert sha256(library) == metadata["delta_library"]["sha256"], "DeltaNet library checksum mismatch"
            assert metadata["tokens"] is None and metadata["token_range"] == [2, 512]
            load_library(library)
        self.functions = {kind: torch._inductor.aoti_load_package(str(packages / f"{kind}.pt2"))
                          for kind in ("prefill", "decode")}
        self.specs = specs

    def reset(self):
        self.position = 0
        self.state = [torch.zeros(s["shape"], device="cuda", dtype=getattr(torch, s["dtype"])) for s in self.specs]

    def step(self, tokens):
        assert 1 <= len(tokens) <= 512 and self.position + len(tokens) <= CACHE_LENGTH
        offset = 0
        while offset < len(tokens):
            remaining = len(tokens) - offset
            length = remaining if self.dynamic_prefill else (PREFILL if remaining >= PREFILL else 1)
            ids = torch.tensor([tokens[offset:offset + length]], device="cuda")
            positions = torch.arange(self.position, self.position + length, device="cuda")
            result = self.functions["prefill" if length > 1 else "decode"](ids, positions, self.state)
            self.state = list(result[1:])
            self.position += length
            offset += length
        return result[0]
