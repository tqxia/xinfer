"""Qwen3 stages and live HF fixtures for the common validation chain."""

from contextlib import contextmanager
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import DynamicCache, GenerationConfig, StaticCache

from compiler.common.artifacts import MODEL_MANIFEST_NAME, require
from compiler.qwen3.adapters.qwen3_flash import FlashStep
from compiler.qwen3.model import load_model
from compiler.common.context import model_context_limit
from compiler.qwen3.packages import validate_model_metadata
from compiler.qwen3.config import DEFAULT_MODEL
from tools.prepare.common import encode_prompt, load_tokenizer
from tools.validation.common.prompts import chat_prompts
from tools.validation.common.runner import ExplicitRuntime, Stage, greedy
from tools.common import prompt_chunks
from tools.validation.common.context import capture_long_state, long_context_cases
from tools.validation.qwen3.cases import extra_cases


def default_model_dir(pair):
    return DEFAULT_MODEL


class HFRuntime:
    """Original HF execution with static or dynamic cache and explicit positions."""

    def __init__(self, model, specs, *, dynamic=False, capacity=None, max_prefill=512):
        self.model = model
        self.specs = specs
        self.dynamic = dynamic
        if capacity is None:
            capacity = next(
                (spec["shape"][2] for spec in specs if spec["update"] == "inplace"), None
            )
            if capacity is None:
                capacity = model_context_limit(model.config)
        self.capacity = capacity
        self.max_prefill = max_prefill
        self.vocab_size = model.config.vocab_size
        self.dtype = model.dtype
        self.cache = None
        self.position = 0

    def reset(self, fill=0):
        if self.dynamic:
            require(fill == 0, "The HF DynamicCache control starts with an empty cache")
            self.cache = DynamicCache()
        else:
            self.cache = StaticCache(
                self.model.config, 1, self.capacity, device="cuda", dtype=self.dtype
            )
            for tensor in (*self.cache.key_cache, *self.cache.value_cache):
                tensor.fill_(fill)
        self.position = 0

    @property
    def state(self):
        values = []
        for spec in self.specs:
            cache = self.cache.key_cache if spec["kind"] == "key" else self.cache.value_cache
            if spec["layer"] >= len(cache):
                value = torch.zeros(spec["shape"], device="cuda", dtype=self.dtype)
            else:
                value = cache[spec["layer"]]
                if self.dynamic:
                    value = F.pad(value, (0, 0, 0, self.capacity - value.shape[2]))
            values.append(value)
        return values

    @torch.inference_mode()
    def step(self, tokens):
        require(1 <= len(tokens) <= 512, "Expected 1..512 input tokens")
        require(self.position + len(tokens) <= self.capacity, "Context capacity exceeded")
        ids = torch.tensor([tokens], dtype=torch.int64, device="cuda")
        positions = torch.arange(self.position, self.position + len(tokens), device="cuda")
        score = self.model(
            input_ids=ids,
            cache_position=positions,
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=1,
        ).logits[:, -1]
        self.position += len(tokens)
        return score

    @torch.inference_mode()
    def generate(self, prompt, limit, eos):
        """Cross-check manual decoding with HF generation on a separate cache."""
        original = self.model.generation_config
        generation = GenerationConfig(
            do_sample=False,
            max_new_tokens=limit,
            eos_token_id=list(eos),
            pad_token_id=original.pad_token_id,
            bos_token_id=original.bos_token_id,
            use_cache=True,
            disable_compile=True,
        )
        cache = (
            DynamicCache()
            if self.dynamic
            else StaticCache(
                self.model.config, 1, self.capacity, device=self.model.device, dtype=self.dtype
            )
        )
        ids = torch.tensor([prompt], dtype=torch.int64, device=self.model.device)
        generated = self.model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            past_key_values=cache,
            generation_config=generation,
            # Transformers 4.51 otherwise merges checkpoint sampling defaults.
            use_model_defaults=False,
        )
        return generated[0, len(prompt) :].tolist()

    def close(self):
        self.cache = None
        self.model = None


class Suite:
    model_type = "qwen3"
    state_required = False
    tokens_required = False

    def __init__(self, model_dir, pair):
        self.model_dir = Path(model_dir)
        self.pair = pair
        self.specs = pair["decode"]["state"]
        self.capacity = pair["decode"]["cache_length"]
        self.vocab_size = pair["decode"]["vocab_size"]
        self.eos = pair["decode"]["eos_token_ids"]
        self.max_prefill = pair["prefill"]["token_range"][1]
        self.megaqwen = pair["decode"]["model_specific"].get("decode_backend") == "megaqwen"
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        manifest = json.loads((self.model_dir / MODEL_MANIFEST_NAME).read_text())
        for entry in pair.values():
            validate_model_metadata(entry)
            require(entry["torch_version"] == torch.__version__, "Package Torch version differs")
            require(entry["checkpoint"]["manifest"] == manifest, "Checkpoint manifest differs")

        self.tokenizer = load_tokenizer(self.model_dir)

    def fixtures(self, hf_runtime):
        references = {}
        for definition in chat_prompts(self.model_type):
            name = definition["name"]
            prompt = encode_prompt(self.tokenizer, definition["text"])
            require(len(prompt) + 64 <= self.capacity, f"Context budget exceeded for {name}")
            generated = greedy(hf_runtime, prompt, 64, self.eos)["ids"]
            hf_ids = hf_runtime.generate(prompt, 64, self.eos)
            require(
                generated == hf_ids,
                f"Manual HF decoding disagrees with HF.generate() for {name}",
            )
            references[name] = prompt, generated

        teacher = []
        for order, (name, (prompt, continuation)) in enumerate(references.items(), start=1):
            teacher.append(
                dict(
                    name=name + "_teacher",
                    steps=[prompt] + [[token] for token in continuation[:-1]],
                    fixture=name,
                    order=order,
                )
            )
        seed = references["capital"][0]
        order = 4
        for length in (1, 2, 7, 128, 511, 512):
            if length + 4 > self.capacity:
                continue
            prompt = (seed * (length // len(seed) + 1))[:length]
            continuation = greedy(hf_runtime, prompt, 4, [])["ids"]
            steps = [prompt] + [[token] for token in continuation[:-1]]
            name = f"length_{length}"
            teacher.extend(
                (
                    dict(name=name, steps=steps, order=order),
                    dict(
                        name=name + "_poison",
                        steps=steps,
                        poison_kv=3,
                        same_as=name,
                        order=order + 1,
                    ),
                )
            )
            order += 2
        teacher.append(
            dict(name="forced_tokens", steps=[seed, [0], [151645], [151643], [13]], order=16)
        )
        full = (seed * (self.capacity // len(seed) + 1))[: self.capacity]
        teacher.append(
            dict(
                name="context_end",
                steps=prompt_chunks(full[:-2], self.max_prefill) + [full[-2:-1], full[-1:]],
                long_context=self.capacity > 1024,
                dump_cache=True,
                check_overflow=True,
                order=17,
            )
        )
        sizes = (1, 2, 13, 16, 31, 1, 64, 128, 256, 511, 1)
        steps, offset = [], 0
        for length in sizes:
            chunk = full[offset : offset + length]
            if chunk:
                steps.append(chunk)
                offset += len(chunk)
        steps.extend(
            prompt_chunks(full[offset:], self.max_prefill) if offset < self.capacity else []
        )
        require(
            sum(map(len, steps)) == self.capacity, "Prefix boundary fixture must fill the context"
        )
        teacher.extend(
            (
                dict(
                    name="prefix_boundaries",
                    steps=steps,
                    check_overflow=True,
                    order=24,
                    long_context=self.capacity > 1024,
                ),
                dict(
                    name="prefix_boundaries_poison",
                    steps=steps,
                    poison_kv=3,
                    same_as="prefix_boundaries",
                    check_overflow=True,
                    order=25,
                    long_context=self.capacity > 1024,
                ),
            )
        )

        def chat(name, fixture, order, limit=64, same_as=None):
            prompt, expected = references[fixture]
            result = dict(
                name=name,
                prompt=prompt,
                fixture=fixture,
                limit=limit,
                expected_ids=expected[:limit],
                order=order,
            )
            if same_as:
                result["same_as"] = same_as
            return result

        chats = [
            chat("capital_initial", "capital", 0),
            chat("capital_after_full", "capital", 18, same_as="capital_initial"),
            chat("arithmetic_after_capital", "arithmetic", 19),
            chat("chinese_after_arithmetic", "chinese", 20),
            chat("capital_repeat", "capital", 21, same_as="capital_initial"),
            chat("limit_one", "capital", 22, limit=1),
            chat("limit_three", "capital", 23, limit=3),
        ]
        long_cases = long_context_cases(seed, self.capacity, self.max_prefill)
        teacher.extend(long_cases)
        for case in long_cases:
            if case["name"] in ("long_prompt_513", "long_prompt_8191", "long_prompt_8192"):
                prompt = [token for chunk in case["steps"][:-2] for token in chunk]
                chats.append(dict(name=case["name"] + "_generate", prompt=prompt, limit=3))
        if self.megaqwen:
            teacher = extra_cases(teacher, seed)
        return teacher, chats

    def stage_cases(self, name, cases):
        if name == "hf_dynamic":
            return [case for case in cases if case.get("fixture")]
        return cases

    def capture_states(self, case, step):
        if case.get("long_context"):
            return capture_long_state(case, step)
        return step == len(case["steps"]) - 1 and (
            bool(case.get("fixture")) or case.get("dump_cache", False)
        )

    def native_state_keys(self, cases):
        return {
            (case["name"], step)
            for case in cases
            for step in range(len(case["steps"]))
            if self.capture_states(case, step)
            and (case.get("dump_cache") or case.get("long_context"))
        }

    @staticmethod
    def state_rows(tensor, spec):
        return tensor.float().flatten(1)

    def stages(self):
        model = load_model(self.model_dir)
        try:
            runtime = HFRuntime(
                model,
                self.specs,
                capacity=self.capacity,
                max_prefill=self.max_prefill,
            )
            try:
                yield Stage("hf", runtime)
            finally:
                runtime.close()
            runtime = HFRuntime(
                model,
                self.specs,
                dynamic=True,
                capacity=self.capacity,
                max_prefill=self.max_prefill,
            )
            try:
                yield Stage("hf_dynamic", runtime)
            finally:
                runtime.close()
            adapter = FlashStep(model)
            decode = adapter
            if self.megaqwen:
                from compiler.qwen3.megaqwen import MegaDecodeStep

                decode = MegaDecodeStep(adapter.model, self.capacity)
            runtime = ExplicitRuntime(
                dict(prefill=adapter, decode=decode),
                self.specs,
                self.vocab_size,
                self.capacity,
                logits_only=True,
                owner=(adapter, decode) if self.megaqwen else adapter,
                require_write=True,
                max_prefill=self.max_prefill,
            )
            try:
                yield Stage("eager", runtime)
            finally:
                runtime.close()
            del adapter, decode, runtime
        finally:
            del model
            torch.cuda.empty_cache()

    @contextmanager
    def reference(self):
        """Reload unmodified BF16 HF for matched greedy-trajectory replay."""
        model = load_model(self.model_dir)
        runtime = HFRuntime(
            model,
            self.specs,
            capacity=self.capacity,
            max_prefill=self.max_prefill,
        )
        try:
            yield runtime
        finally:
            runtime.close()
            del runtime, model
            torch.cuda.empty_cache()
