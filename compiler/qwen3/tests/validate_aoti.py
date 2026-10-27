"""Check the static-cache wrapper, compiled handoff, shapes, and cache lifetime."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import StaticCache
from compiler.common.validation import row_errors, bits_equal

from compiler.qwen3.model import CACHE_LENGTH, QwenStep, allocate_cache, load_model
from compiler.qwen3.prepare import DEFAULT_MODEL

# Calibrated with an independent FP32 control, then checked on boundary cases.
# Pointwise relative error near zero is not useful for accumulated BF16 error.
MAX_RELATIVE_ERROR = 0.05


def ids_from(path):
    return [int(word) for word in path.read_text().split()]


def call(fn, tokens, start, cache):
    ids = torch.tensor([tokens], dtype=torch.int64, device="cuda")
    positions = torch.arange(start, start + len(tokens), device="cuda")
    return fn(ids, positions, cache).float()


def trace(fn, prompt, continuation, cache):
    rows = [call(fn, prompt, 0, cache)[0].cpu()]
    for i, token in enumerate(continuation[:-1]):
        rows.append(call(fn, [token], len(prompt) + i, cache)[0].cpu())
    return torch.stack(rows)


def error_metrics(actual, expected):
    error = row_errors(actual, expected)
    return {"max_abs": error["max_abs"], "max_relative_l2": error["relative_l2"],
            "max_peak_relative": error["peak_relative"],
            "top1_agreement": (actual.argmax(-1) == expected.argmax(-1)).sum().item(),
            "rows": actual.shape[0]}


def compare(actual, expected, exact=False):
    metrics = error_metrics(actual, expected)
    if exact:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert bits_equal(actual, expected)
    else:
        assert metrics["max_relative_l2"] <= MAX_RELATIVE_ERROR, metrics
        assert metrics["max_peak_relative"] <= MAX_RELATIVE_ERROR, metrics
    return metrics


def generate(fn, prompt, cache, eos_ids, limit):
    for tensor in cache:
        tensor.zero_()
    score = call(fn, prompt, 0, cache)
    result = []
    for i in range(limit):
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
    model = load_model(args.model_dir)
    wrapper = QwenStep(model)
    prefill = torch._inductor.aoti_load_package(str(args.packages / "prefill.pt2"))
    decode = torch._inductor.aoti_load_package(str(args.packages / "decode.pt2"))

    def compiled(ids, positions, kv):
        fn = decode if ids.shape[1] == 1 else prefill
        result = fn(ids, positions, kv)
        assert result.shape == (1, model.config.vocab_size)
        assert result.dtype == torch.bfloat16
        return result

    def hf_static(ids, positions, cache):
        return model(input_ids=ids, cache_position=positions, past_key_values=cache,
                     use_cache=True, logits_to_keep=1).logits[:, -1]

    report = {"fixtures": [], "boundaries": [], "context_end": []}
    precision_cases = []
    for case in sorted(args.reference.iterdir()):
        if not case.is_dir():
            continue
        prompt = ids_from(case / "prompt.ids")
        continuation = ids_from(case / "generated.ids")
        meta = json.loads((case / "metadata.json").read_text())
        eager_kv, aoti_kv = allocate_cache(model.config), allocate_cache(model.config)
        hf_kv = StaticCache(model.config, 1, CACHE_LENGTH, device="cuda", dtype=torch.bfloat16)
        eager = trace(wrapper, prompt, continuation, eager_kv)
        official = trace(hf_static, prompt, continuation, hf_kv)
        compare(eager, official, exact=True)
        for index, (k, v) in enumerate(zip(hf_kv.key_cache, hf_kv.value_cache)):
            torch.testing.assert_close(eager_kv[2 * index], k, rtol=0, atol=0)
            torch.testing.assert_close(eager_kv[2 * index + 1], v, rtol=0, atol=0)
        pointers = [tensor.data_ptr() for tensor in aoti_kv]
        actual = trace(compiled, prompt, continuation, aoti_kv)
        metrics = compare(actual, eager)
        assert metrics["top1_agreement"] == len(continuation)
        assert pointers == [tensor.data_ptr() for tensor in aoti_kv]
        # Check the contents of every cache tensor after the compiled handoff.
        cache_metrics = [compare(a.flatten(1).float(), b.flatten(1).float())
                         for a, b in zip(aoti_kv, eager_kv)]
        dynamic = torch.from_numpy(np.fromfile(case / "logits.f32", dtype="<f4").reshape(meta["logits_shape"]))
        assert actual.argmax(-1).tolist() == continuation
        assert dynamic.argmax(-1).tolist() == continuation
        # This resets and reuses the same cache; generation follows its own tokens.
        assert generate(compiled, prompt, aoti_kv, meta["eos_token_ids"], meta["max_new_tokens"]) == continuation
        report["fixtures"].append({"name": case.name, "aoti_vs_static": metrics,
                                   "cache_max_relative_l2": max(m["max_relative_l2"] for m in cache_metrics),
                                   "cache_max_abs": max(m["max_abs"] for m in cache_metrics),
                                   "hf_static_exact": True, "greedy_and_reset_pass": True})
        precision_cases.append((prompt, continuation, eager, actual))
        trace_dir = args.packages / "reference"
        trace_dir.mkdir(exist_ok=True)
        actual.numpy().astype("<f4").tofile(trace_dir / f"{case.name}.f32")
        print(report["fixtures"][-1], flush=True)

    seed = ids_from(args.reference / "capital" / "prompt.ids")
    reused = allocate_cache(model.config)
    for length in (1, 2, 7, 128, 511, 512):
        tokens = (seed * (length // len(seed) + 1))[:length]
        expected_kv = allocate_cache(model.config)
        for tensor in reused:
            tensor.zero_()
        expected = call(wrapper, tokens, 0, expected_kv)
        actual = check_write(compiled, tokens, 0, reused)
        metrics = compare(actual, expected)
        # Nonzero future entries must be ignored, and must remain untouched.
        poisoned = allocate_cache(model.config)
        for tensor in poisoned:
            tensor.fill_(3)
        poisoned_score = check_write(compiled, tokens, 0, poisoned)
        compare(poisoned_score, actual, exact=True)
        token = expected.argmax(-1).item()
        expected = call(wrapper, [token], length, expected_kv)
        actual = check_write(compiled, [token], length, reused)
        handoff = compare(actual, expected)
        report["boundaries"].append({"prompt_length": length, "prefill": metrics, "decode": handoff,
                                     "address_and_write_region_pass": True, "future_mask_pass": True})
        print(report["boundaries"][-1], flush=True)

    # Fill a real 1,022-token prefix in chunks, then exercise both final slots.
    full = (seed * (CACHE_LENGTH // len(seed) + 1))[:CACHE_LENGTH]
    eager_kv = allocate_cache(model.config)
    for tensor in reused:
        tensor.zero_()
    for start, count in ((0, 512), (512, 510), (1022, 1), (1023, 1)):
        tokens = full[start:start + count]
        expected = call(wrapper, tokens, start, eager_kv)
        actual = check_write(compiled, tokens, start, reused)
        report["context_end"].append({"position": start, "tokens": count, **compare(actual, expected)})
        print(report["context_end"][-1], flush=True)
    # A different prompt after a full context must start from empty cache state.
    capital_meta = json.loads((args.reference / "capital" / "metadata.json").read_text())
    assert generate(compiled, seed, reused, capital_meta["eos_token_ids"], 64) == ids_from(args.reference / "capital" / "generated.ids")
    report["reset_after_full_context_pass"] = True
    # Promote the same checkpoint weights to FP32 as an independent precision
    # control. Neither compilation nor cache adaptation should hide BF16 drift.
    model.float()
    for fixture, (prompt, continuation, eager, actual) in zip(report["fixtures"], precision_cases):
        fp32_kv = tuple(t.float() for t in allocate_cache(model.config))
        fp32 = trace(wrapper, prompt, continuation, fp32_kv)
        fixture["bf16_vs_fp32"] = compare(eager, fp32)
        fixture["aoti_vs_fp32"] = compare(actual, fp32)
        assert fp32.argmax(-1).tolist() == continuation
        print({"name": fixture["name"], "fp32_control": fixture["aoti_vs_fp32"]}, flush=True)
    report["error_limits"] = {"max_relative_l2": MAX_RELATIVE_ERROR,
                               "max_peak_relative": MAX_RELATIVE_ERROR,
                               "scope": "per logit row and per whole K/V tensor"}
    (args.packages / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: static wrapper, AOTI handoff, greedy sequences, shapes, masks, cache reuse, FP32 control", flush=True)


if __name__ == "__main__":
    main()
