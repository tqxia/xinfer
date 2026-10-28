"""Compare full-text HF, the tensor adapter and Python-loaded AOTI packages."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, DynamicCache, StaticCache

from compiler.qwen35.config import ARTIFACT_DIR, CACHE_LENGTH, PREFILL, TEXT_DIR
from compiler.common.artifacts import sha256, ROOT
from compiler.qwen35.model import TextStep, configure_text, load_model, state_specs, zero_state
from compiler.common.validation import bits_equal
from compiler.qwen35.tests.common import fixtures, compare, LIMIT
from compiler.qwen35.reference import PythonRuntime


def cache_tensors(cache, config):
    tensors = []
    for kind, layer in zip(config.layer_types, cache.layers):
        tensors.extend((layer.conv_states[0], layer.recurrent_states[0]) if kind == "linear_attention"
                       else (layer.keys, layer.values))
    return tensors


def hf_trace(model, cases, dynamic=False):
    trace = {}
    dtype = next(model.parameters()).dtype
    for case in cases:
        cache = DynamicCache(config=model.config) if dynamic else StaticCache(model.config, max_cache_len=CACHE_LENGTH)
        offset = 0
        for index, tokens in enumerate(case["steps"]):
            ids = torch.tensor([tokens], device="cuda")
            position = torch.arange(offset, offset + len(tokens), device="cuda")
            width = offset + len(tokens) if dynamic else CACHE_LENGTH
            mask = torch.full((1, 1, len(tokens), width), torch.finfo(dtype).min,
                              device="cuda", dtype=dtype).triu(offset + 1)
            logits = model(ids, position_ids=position[None, :],
                           attention_mask={"full_attention": mask, "linear_attention": None},
                           past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, 0, :]
            tensors = (logits, *cache_tensors(cache, model.config))
            trace[(case["name"], index)] = tuple(t.cpu().clone() for t in tensors)
            offset += len(tokens)
    return trace


def explicit_trace(functions, config, cases):
    trace = {}
    specs = state_specs(config)
    for case in cases:
        state, offset = zero_state(config), 0
        for index, tokens in enumerate(case["steps"]):
            ids = torch.tensor([tokens], device="cuda")
            position = torch.arange(offset, offset + len(tokens), device="cuda")
            before = [t.clone() for t in (ids, position, *state)]
            kind = "prefill" if len(tokens) > 1 else "decode"
            output = functions[kind](ids, position, state)
            assert len(output) == 1 + len(specs)
            assert output[0].shape == (1, config.vocab_size) and output[0].dtype == torch.bfloat16
            assert all(bits_equal(a, b) for a, b in zip((ids, position, *state), before)), "Input mutation"
            for tensor, spec in zip(output[1:], specs):
                assert list(tensor.shape) == spec["shape"] and str(tensor.dtype) == "torch." + spec["dtype"]
            assert all(torch.isfinite(t).all() for t in output)
            trace[(case["name"], index)] = tuple(t.cpu().clone() for t in output)
            state, offset = list(output[1:]), offset + len(tokens)
    return trace


def check_mask_and_writes(functions, config, tokens):
    length = len(tokens)
    specs = state_specs(config)
    ids = torch.tensor([tokens], device="cuda")
    position = torch.arange(length, device="cuda")
    state = zero_state(config)
    clean = functions["prefill"](ids, position, state)
    for tensor, spec in zip(state, specs):
        if spec["kind"] in ("key", "value"):
            tensor.fill_(7.0 if spec["kind"] == "key" else -3.0)
    poison = functions["prefill"](ids, position, state)
    assert bits_equal(clean[0], poison[0]), "Masked future cache affects logits"
    for index, spec in enumerate(specs):
        a, b = clean[index + 1], poison[index + 1]
        if spec["kind"] in ("key", "value"):
            assert bits_equal(a[:, :, :length], b[:, :, :length])
            assert bits_equal(state[index][:, :, length:], b[:, :, length:]), "Future cache overwritten"
        else:
            assert bits_equal(a, b)
    next_state = list(clean[1:])
    decoded = functions["decode"](ids[:, :1].contiguous(), position.new_tensor([length]), next_state)
    for index, spec in enumerate(specs):
        if spec["kind"] in ("key", "value"):
            assert bits_equal(next_state[index][:, :, :length], decoded[index + 1][:, :, :length])
            assert bits_equal(next_state[index][:, :, length + 1:], decoded[index + 1][:, :, length + 1:])
    return dict(poisoned_future_ignored=True, kv_write_regions_preserved=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=TEXT_DIR)
    parser.add_argument("--packages", type=Path, default=ARTIFACT_DIR / "text")
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "text-validation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_text()
    metadata = {kind: json.loads((args.packages / f"{kind}.json").read_text()) for kind in ("prefill", "decode")}
    for kind, meta in metadata.items():
        assert meta["torch"] == torch.__version__ and meta["cache_length"] == CACHE_LENGTH
        assert meta["package_sha256"] == sha256(args.packages / f"{kind}.pt2")
        assert meta["checkpoint"] == json.loads((args.model_dir / "manifest.json").read_text())
    for field in ("checkpoint", "state", "source_sha256", "inductor_configs", "attention_backend"):
        assert metadata["prefill"][field] == metadata["decode"][field], field
    assert metadata["prefill"].get("eager_numerics", False) == metadata["decode"].get("eager_numerics", False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    dynamic_prefill = metadata["prefill"].get("dynamic_prefill", False)
    cases = fixtures(tokenizer, dynamic_prefill)
    (args.output / "cases.json").write_text(json.dumps(dict(cases=cases), indent=2) + "\n")
    model = load_model(args.model_dir)
    config, specs = model.config, state_specs(model.config)
    assert metadata["prefill"]["state"] == specs
    print("HF StaticCache reference", flush=True)
    reference = hf_trace(model, cases)
    print("HF DynamicCache control", flush=True)
    dynamic = hf_trace(model, [cases[1]], dynamic=True)
    # DynamicCache has only the valid prefix; pad unused storage for comparison.
    for key, tensors in dynamic.items():
        offset = sum(len(t) for t in cases[1]["steps"][:key[1] + 1])
        padded = list(tensors)
        for i, spec in enumerate(specs, start=1):
            if spec["kind"] in ("key", "value"):
                padded[i] = torch.nn.functional.pad(tensors[i], (0, 0, 0, CACHE_LENGTH - offset))
        dynamic[key] = tuple(padded)
    print("Functional tensor adapter", flush=True)
    if dynamic_prefill:
        from compiler.qwen35.adapters.native_delta import install_native_delta, load_library
        load_library(args.packages / "libxinfer_delta_ops.so")
        install_native_delta(model)
    wrapper = TextStep(model)
    adapted = explicit_trace(dict(prefill=wrapper, decode=wrapper), config, cases)
    del wrapper, model
    torch.cuda.empty_cache()
    print("Independent FP32 reference", flush=True)
    fp32_model = load_model(args.model_dir, dtype=torch.float32)
    fp32 = hf_trace(fp32_model, cases)
    del fp32_model
    torch.cuda.empty_cache()
    print("Python AOTI", flush=True)
    functions = PythonRuntime(args.packages, specs).functions
    compiled = explicit_trace(functions, config, cases)
    report = dict(limit=LIMIT, cases=cases, comparisons=[], failures=[], top1_differences=[],
                  checkpoint=metadata["prefill"]["checkpoint"], package_sha256={k: m["package_sha256"] for k, m in metadata.items()},
                  source_sha256={name: sha256(ROOT / name) for name in ('compiler/qwen35/config.py', 'compiler/qwen35/model.py', 'compiler/qwen35/tests/validate_model.py', 'compiler/qwen35/tests/common.py', 'compiler/common/validation.py', 'compiler/common/artifacts.py')},
                  checks=check_mask_and_writes(functions, config, cases[1]["steps"][0]))
    if dynamic_prefill:
        seed = cases[1]["steps"][0] * 8
        report["checks"]["dynamic_mask_lengths"] = [2, 22, 65, 511, 512]
        for length in report["checks"]["dynamic_mask_lengths"]:
            check_mask_and_writes(functions, config, seed[:length])
    for name, actual, expected, enforce in (
        ("adapter_vs_hf", adapted, reference, True),
        ("aoti_vs_hf", compiled, reference, True),
        ("aoti_vs_adapter", compiled, adapted, True),
        ("dynamic_vs_static_hf", dynamic, reference, True),
        ("aoti_vs_fp32", compiled, fp32, False),
        ("hf_vs_fp32", reference, fp32, False),
    ):
        compare(report, name, actual, expected, specs, enforce)
    for i in range(len(cases[1]["steps"])):
        assert all(bits_equal(a, b) for a, b in zip(compiled[("prefill_decode", i)], compiled[("repeat", i)]))
    report["checks"].update(reset_repeat_bitwise=True, input_preservation=True, finite_shapes_dtypes=True)
    report["passed"] = not report["failures"]
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save({name: trace[("long_prefix", len(cases[4]["steps"]) - 1)]
                for name, trace in dict(hf=reference, aoti=compiled, fp32=fp32).items()}, args.output / "final_states.pt")
    print(json.dumps(dict(passed=report["passed"], checked_steps=len(compiled),
                          numerical_failures=len(report["failures"]), top1_differences=report["top1_differences"]), indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
