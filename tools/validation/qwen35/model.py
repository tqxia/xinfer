"""Qwen3.5 stages and fixtures for the shared numerical validation chain."""

from contextlib import contextmanager
import gc
from pathlib import Path

import torch
from transformers import AutoTokenizer, DynamicCache, StaticCache

from compiler.common.artifacts import MODEL_MANIFEST_NAME, read_json, require
from compiler.qwen35.config import TEXT_DIR
from tools.prepare.qwen35 import REVISIONS
from compiler.qwen35.model import TextStep, configure_text, load_model, state_specs
from compiler.qwen35.packages import validate_model_metadata
from tools.prepare.common import encode_prompt
from tools.validation.common.prompts import PARAGRAPH, chat_prompts
from tools.validation.common.runner import ExplicitRuntime, Stage, greedy
from tools.validation.qwen35.common import fixtures as model_fixtures
from tools.common import prompt_chunks
from tools.validation.common.context import capture_long_state, long_context_cases


def default_model_dir(pair):
    """Select the prepared checkpoint matching the package's pinned model ID."""
    model_id = pair["decode"]["checkpoint"]["model_id"]
    require(model_id in REVISIONS, "Unsupported Qwen3.5 checkpoint")
    return TEXT_DIR.parent.parent / model_id.rsplit("-", 1)[1].lower() / "text"


class HFStage:
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


class Suite:
    model_type = "qwen35"
    state_required = True
    tokens_required = True

    def __init__(self, model_dir, pair):
        self.model_dir = Path(model_dir)
        self.pair = pair
        self.specs = pair["decode"]["state"]
        self.vocab_size = pair["decode"]["vocab_size"]
        self.capacity = pair["decode"]["cache_length"]
        self.max_prefill = pair["prefill"]["token_range"][1]
        self.eos = pair["decode"]["eos_token_ids"]
        configure_text()
        manifest = read_json(self.model_dir / MODEL_MANIFEST_NAME)
        for metadata in pair.values():
            validate_model_metadata(metadata)
            require(metadata["torch_version"] == torch.__version__, "Torch version mismatch")
            require(metadata["checkpoint"]["manifest"] == manifest, "Checkpoint manifest differs")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)

    def fixtures(self, hf_runtime):
        """Retain the numerical-model and end-to-end fixtures without renaming origins."""
        cases = []
        for original in model_fixtures(self.tokenizer):
            if sum(map(len, original["steps"])) > self.capacity:
                continue
            name = original["name"]
            case = dict(original, name="model_" + name, origin=dict(suite="model", name=name))
            if name == "repeat":
                case["same_as"] = "model_prefill_decode"
            elif name == "mixed_dispatch_repeat":
                case["same_as"] = "model_mixed_dispatch"
            cases.append(case)

        paragraph = self.tokenizer.encode(PARAGRAPH)
        stream = (paragraph * (self.capacity // len(paragraph) + 1))[: self.capacity]
        teacher = []
        for length in (1, 2, 3, 22, 31, 32, 33, 63, 64, 65, 96, 97, 98, 127, 128, 129, 511, 512):
            decode = 64 if length == 512 else 8
            if length + decode > self.capacity:
                continue
            teacher.append(
                dict(
                    name=f"length_{length}",
                    steps=[stream[:length]] + [[t] for t in stream[length : length + decode]],
                )
            )
        teacher.append(
            dict(
                name="context_end",
                steps=(
                    [stream[:512], stream[512:960], stream[960:1023], stream[1023:]]
                    if self.capacity == 1024
                    else prompt_chunks(stream[:-1], self.max_prefill) + [stream[-1:]]
                ),
                long_context=self.capacity > 1024,
                check_overflow=True,
            )
        )
        sizes = (96, 98, 97, 99, 1)
        offset, steps = 0, []
        for length in sizes:
            steps.append(stream[offset : offset + length])
            offset += length
        mixed = dict(name="mixed_dispatch", steps=steps)
        teacher.extend((mixed, dict(mixed, name="mixed_dispatch_repeat", same_as="mixed_dispatch")))
        for name in ("length_2", "length_65", "length_98", "context_end"):
            clean = next(case for case in teacher if case["name"] == name)
            teacher.append(dict(clean, name=name + "_poison", same_as=name, poison_kv=3))
        teacher.append(
            dict(
                next(case for case in teacher if case["name"] == "length_65"),
                name="repeat_65",
                same_as="length_65",
            )
        )
        teacher.extend(long_context_cases(paragraph, self.capacity, self.max_prefill))
        chats = [
            dict(
                name=prompt["name"],
                prompt=encode_prompt(self.tokenizer, prompt["text"]),
                text=prompt["text"],
                limit=64,
            )
            for prompt in chat_prompts(self.model_type)
        ]
        for chat in chats:
            generated = greedy(hf_runtime, chat["prompt"], chat["limit"], self.eos)
            teacher.append(
                dict(
                    name=chat["name"] + "_teacher",
                    steps=[chat["prompt"]] + [[token] for token in generated["ids"][:-1]],
                )
            )
        cases += [
            dict(case, origin=dict(suite="end_to_end", name=case["name"])) for case in teacher
        ]
        for case in teacher:
            if case["name"] in ("long_prompt_513", "long_prompt_8191", "long_prompt_8192"):
                prompt = [token for chunk in case["steps"][:-2] for token in chunk]
                chats.append(dict(name=case["name"] + "_generate", prompt=prompt, limit=3))
        return cases, chats

    def capture_states(self, case, step):
        return capture_long_state(case, step) if case.get("long_context") else True

    def fp32_cases(self, cases):
        selected = {
            "length_96",
            "length_97",
            "length_98",
            "length_512",
            "context_end",
            "capital_teacher",
            "long_prompt_513",
            "long_prompt_8192",
            "long_context_end",
        }
        return [
            case
            for case in cases
            if case["origin"]["suite"] == "model" or case["origin"]["name"] in selected
        ]

    def native_state_keys(self, cases):
        return {
            (case["name"], step)
            for case in cases
            for step in range(len(case["steps"]))
            if self.capture_states(case, step)
        }

    def stage_cases(self, name, cases):
        if name == "hf_dynamic":
            return [case for case in cases if case["name"] == "model_prefill_decode"]
        return cases

    def stages(self):
        """Record HF before applying the adapters and package-selected graph passes."""
        model, wrapper, functions = load_model(self.model_dir), None, None
        try:
            require(
                state_specs(model.config, self.capacity) == self.specs, "State descriptors differ"
            )
            bounds = dict(
                max_prefill=getattr(self, "max_prefill", 512),
            )
            yield Stage("hf", HFStage(model, self.specs, self.capacity, **bounds))
            yield Stage(
                "hf_dynamic", HFStage(model, self.specs, self.capacity, dynamic=True, **bounds)
            )
            from compiler.qwen35.adapters.delta import install_delta

            install_delta(model)
            if (
                self.pair["prefill"]["model_specific"]
                .get("linear_packing", {})
                .get("enabled", False)
            ):
                from compiler.qwen35.export import export_step

                functions = {
                    kind: export_step(model, kind, self.capacity)[0].module()
                    for kind in ("prefill", "decode")
                }
            else:
                wrapper = TextStep(model)
                functions = dict(prefill=wrapper, decode=wrapper)
            yield Stage(
                "eager",
                ExplicitRuntime(
                    functions, self.specs, self.vocab_size, self.capacity, owner=model, **bounds
                ),
            )
        finally:
            wrapper, functions, model = None, None, None
            gc.collect()
            torch.cuda.empty_cache()

    @contextmanager
    def reference(self, dtype):
        model = load_model(self.model_dir, dtype=dtype)
        runtime = HFStage(
            model,
            self.specs,
            self.capacity,
            max_prefill=self.max_prefill,
        )
        try:
            require(
                state_specs(model.config, self.capacity) == self.specs, "State descriptors differ"
            )
            yield runtime
        finally:
            runtime.close()
            model = None
            gc.collect()
            torch.cuda.empty_cache()
