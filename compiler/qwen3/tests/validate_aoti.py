"""Check Flash AOTI handoff/cache state and report drift from independent HF references."""

import argparse
import json
from pathlib import Path

import torch
from transformers import StaticCache

from compiler.common.validation import bits_equal
from compiler.qwen3.custom_ops.attention import load_package_attention_ops
from compiler.qwen3.model import CACHE_LENGTH, QwenStep, allocate_cache, cache_extent, load_model
from compiler.qwen3.prepare import DEFAULT_MODEL
from compiler.qwen3.tests.numerics import (
    MAX_RELATIVE_ERROR, compare, error_metrics, ids_from, policy, token_differences,
    within_logit_limits,
)


def call(fn, tokens, start, cache):
    ids = torch.tensor([tokens], dtype=torch.int64, device="cuda")
    positions = torch.arange(start, start + len(tokens), device="cuda")
    return fn(ids, positions, cache, cache_extent(start + len(tokens))).float()


def trace(fn, prompt, continuation, cache):
    rows = [call(fn, prompt, 0, cache)[0].cpu()]
    for i, token in enumerate(continuation[:-1]):
        rows.append(call(fn, [token], len(prompt) + i, cache)[0].cpu())
    return torch.stack(rows)


def generate(fn, prompt, cache, eos_ids, limit):
    for tensor in cache:
        tensor.zero_()
    score = call(fn, prompt, 0, cache)
    result = []
    for i in range(limit):
        assert torch.isfinite(score).all().item(), "Nonfinite compiled logits"
        token = score.argmax(-1).item()
        result.append(token)
        if token in eos_ids or i + 1 == limit:
            break
        score = call(fn, [token], len(prompt) + i, cache)
    return result


def check_write(fn, tokens, start, cache):
    """A call may change only the requested cache slots, with stable storage."""
    before = [tensor.clone() for tensor in cache]
    pointers = [tensor.data_ptr() for tensor in cache]
    score = call(fn, tokens, start, cache)
    end = start + len(tokens)
    assert pointers == [tensor.data_ptr() for tensor in cache]
    for old, new in zip(before, cache):
        assert torch.isfinite(new).all().item(), "Nonfinite compiled cache"
        assert torch.equal(old[:, :, :start], new[:, :, :start])
        assert torch.equal(old[:, :, end:], new[:, :, end:])
        assert not torch.equal(old[:, :, start:end], new[:, :, start:end])
    return score


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--packages", type=Path, default=Path("artifacts/aoti"))
    parser.add_argument("--reference", type=Path, default=Path("artifacts/reference"))
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    metadata = {kind: json.loads((args.packages / (kind + ".json")).read_text())
                for kind in ("prefill", "decode")}
    for spec in metadata.values():
        load_package_attention_ops(spec)
    for key in ("model_revision", "attention_backend", "custom_attention", "cache_input", "kv_extent",
                "torch_version", "cxx11_abi", "cache_shape", "cache_stride", "flat_input_count",
                "inductor_configs", "triton_enable_fp_fusion"):
        assert metadata["prefill"][key] == metadata["decode"][key], f"Package settings differ: {key}"
    model = load_model(args.model_dir)
    wrapper = QwenStep(model)
    prefill = torch._inductor.aoti_load_package(str(args.packages / "prefill.pt2"))
    decode = torch._inductor.aoti_load_package(str(args.packages / "decode.pt2"))

    def compiled(ids, positions, kv, extent):
        result = (decode if ids.shape[1] == 1 else prefill)(ids, positions, kv, extent)
        assert result.shape == (1, model.config.vocab_size) and result.dtype == torch.bfloat16
        return result

    def eager_wrapper(ids, positions, kv, extent):
        return wrapper(ids, positions, kv)

    def hf_static(ids, positions, cache, extent):
        return model(input_ids=ids, cache_position=positions, past_key_values=cache,
                     use_cache=True, logits_to_keep=1).logits[:, -1]

    report = dict(fixtures=[], boundaries=[], context_end=[], numerical_policy=policy(),
                  numerical_failures=[], hf_cache_differences=[], hf_token_differences=[],
                  hf_greedy_differences=[])
    report["numerical_policy"]["native_python_aoti"] = "checked_by_validate_cpp"
    hf_parity = True

    def compare_logits(actual, expected, name):
        nonlocal hf_parity
        metrics = error_metrics(actual, expected)
        hf_parity &= bits_equal(actual, expected)
        if not within_logit_limits(metrics):
            report["numerical_failures"].append(dict(case=name, **metrics))
        differences = token_differences(actual, expected)
        if differences:
            report["hf_token_differences"].append(dict(case=name, differences=differences))
        return metrics

    def compare_cache(actual, expected, name):
        nonlocal hf_parity
        metrics = []
        for index, (a, b) in enumerate(zip(actual, expected)):
            error = error_metrics(a.flatten(1).float(), b.flatten(1).float())
            error.pop("top1_agreement")
            error.update(layer=index // 2, kind="K" if index % 2 == 0 else "V",
                         bitwise_equal=bits_equal(a, b))
            hf_parity &= error["bitwise_equal"]
            if not within_logit_limits(error):
                report["hf_cache_differences"].append(dict(case=name, **error))
            metrics.append(error)
        return metrics

    precision_cases, compiled_sequences = [], {}
    for case in sorted(args.reference.iterdir()):
        if not case.is_dir():
            continue
        prompt, continuation = ids_from(case / "prompt.ids"), ids_from(case / "generated.ids")
        meta = json.loads((case / "metadata.json").read_text())
        eager_kv, aoti_kv = allocate_cache(model.config), allocate_cache(model.config)
        hf_kv = StaticCache(model.config, 1, CACHE_LENGTH, device="cuda", dtype=torch.bfloat16)
        eager = trace(eager_wrapper, prompt, continuation, eager_kv)
        official = trace(hf_static, prompt, continuation, hf_kv)
        compare(eager, official, exact=True)
        for index, (k, v) in enumerate(zip(hf_kv.key_cache, hf_kv.value_cache)):
            compare(eager_kv[2 * index], k, exact=True)
            compare(eager_kv[2 * index + 1], v, exact=True)
        pointers = [tensor.data_ptr() for tensor in aoti_kv]
        actual = trace(compiled, prompt, continuation, aoti_kv)
        metrics = compare_logits(actual, eager, case.name)
        assert pointers == [tensor.data_ptr() for tensor in aoti_kv]
        cache_metrics = compare_cache(aoti_kv, eager_kv, case.name)
        generated = generate(compiled, prompt, aoti_kv, meta["eos_token_ids"], meta["max_new_tokens"])
        compiled_sequences[case.name] = generated
        if generated != continuation:
            report["hf_greedy_differences"].append(dict(case=case.name, hf_ids=continuation, compiled_ids=generated))
        report["fixtures"].append(dict(name=case.name, aoti_vs_static=metrics,
                                       cache_metrics=cache_metrics, hf_static_exact=True,
                                       compiled_generated_ids=generated))
        precision_cases.append((prompt, continuation, eager, actual))
        print(dict(name=case.name, aoti_vs_static=metrics), flush=True)

    seed = ids_from(args.reference / "capital" / "prompt.ids")
    reused = allocate_cache(model.config)
    for length in (1, 2, 7, 128, 511, 512):
        tokens = (seed * (length // len(seed) + 1))[:length]
        expected_kv = allocate_cache(model.config)
        for tensor in reused:
            tensor.zero_()
        expected = call(eager_wrapper, tokens, 0, expected_kv)
        actual = check_write(compiled, tokens, 0, reused)
        metrics = compare_logits(actual, expected, f"length_{length}_prefill")
        poisoned = allocate_cache(model.config)
        for tensor in poisoned:
            tensor.fill_(3)
        compare(check_write(compiled, tokens, 0, poisoned), actual, exact=True)
        token = expected.argmax(-1).item()
        expected = call(eager_wrapper, [token], length, expected_kv)
        actual = check_write(compiled, [token], length, reused)
        handoff = compare_logits(actual, expected, f"length_{length}_decode")
        report["boundaries"].append(dict(prompt_length=length, prefill=metrics, decode=handoff,
                                         address_and_write_region_pass=True, future_mask_pass=True))

    full = (seed * (CACHE_LENGTH // len(seed) + 1))[:CACHE_LENGTH]
    eager_kv = allocate_cache(model.config)
    for tensor in reused:
        tensor.zero_()
    for start, count in ((0, 512), (512, 510), (1022, 1), (1023, 1)):
        expected = call(eager_wrapper, full[start:start + count], start, eager_kv)
        actual = check_write(compiled, full[start:start + count], start, reused)
        report["context_end"].append(dict(position=start, tokens=count,
                                           **compare_logits(actual, expected, f"context_{start}")))
    report["full_cache_aoti_vs_hf"] = compare_cache(reused, eager_kv, "context_end")
    capital_meta = json.loads((args.reference / "capital" / "metadata.json").read_text())
    assert generate(compiled, seed, reused, capital_meta["eos_token_ids"],
                    capital_meta["max_new_tokens"]) == compiled_sequences["capital"]
    report["reset_after_full_context_pass"] = True
    model.float()
    for fixture, (prompt, continuation, eager, actual) in zip(report["fixtures"], precision_cases):
        fp32_kv = tuple(t.float() for t in allocate_cache(model.config))
        fp32 = trace(eager_wrapper, prompt, continuation, fp32_kv)
        fixture["bf16_vs_fp32"] = error_metrics(eager, fp32)
        fixture["aoti_vs_fp32"] = error_metrics(actual, fp32)
    report["passed"] = not report["numerical_failures"]
    report["hf_parity_passed"] = bool(hf_parity) and not report["hf_greedy_differences"]
    report["hf_reference_limits_passed"] = not any(report[key] for key in (
        "numerical_failures", "hf_cache_differences", "hf_token_differences", "hf_greedy_differences"))
    (args.packages / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(f"FAIL: {len(report['numerical_failures'])} logit comparisons exceed {MAX_RELATIVE_ERROR:.0%}; "
                         f"see {args.packages / 'validation.json'}")
    print("PASS: bounded logits, Flash handoff, cache state/reuse and FP32 controls; HF drift recorded", flush=True)


if __name__ == "__main__":
    main()
