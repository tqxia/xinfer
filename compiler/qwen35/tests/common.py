"""Full-model fixtures and per-state acceptance policy shared by Qwen3.5 tests."""

from compiler.qwen35.config import PREFILL
from compiler.common.validation import bits_equal, row_errors

LIMIT = 0.05


def metrics(actual, expected, index):
    assert actual.shape == expected.shape
    width = expected.shape[-1] if index < 2 else expected.shape[-2] * expected.shape[-1]
    a, b = (t.float().reshape(-1, width) for t in (actual, expected))
    return row_errors(a, b)


def fixtures(tokenizer, dynamic=False):
    paragraph = tokenizer.encode("The capital of France is Paris. A small compiler can run a language model. ",
                                 add_special_tokens=False)
    stream = (paragraph * (1024 // len(paragraph) + 1))[:1024]

    def case(name, prefix, decode):
        sizes, remaining = ([prefix], 0) if dynamic else ([], prefix)
        while remaining:
            length = PREFILL if remaining >= PREFILL else 1
            sizes.append(length)
            remaining -= length
        sizes += [1] * decode
        offset, steps = 0, []
        for length in sizes:
            steps.append(stream[offset:offset + length])
            offset += length
        return dict(name=name, prefix=prefix, steps=steps)

    cases = [case("decode_zero", 1, 3), case("prefill_decode", 64, 8),
             case("remainder_65", 65, 4), case("two_prefills", 128, 4),
             case("long_prefix", 512, 8), case("repeat", 64, 8)]
    if dynamic:
        cases += [case(f"dynamic_{length}", length, 2) for length in (2, 22, 31, 32, 33, 63, 127, 129, 511)]
    return cases


def compare(report, name, actual, expected, specs, enforce=True):
    for key, values in actual.items():
        row = dict(comparison=name, case=key[0], step=key[1], tensors={})
        for i, (a, b) in enumerate(zip(values, expected[key])):
            label = "logits" if i == 0 else specs[i - 1]["name"]
            axis = 0 if i == 0 else (1 if specs[i - 1]["kind"] == "conv" else 2)
            error = dict(**metrics(a, b, axis), bitwise_equal=bits_equal(a, b))
            row["tensors"][label] = error
            if enforce and max(error["relative_l2"], error["peak_relative"]) > LIMIT:
                report["failures"].append(dict(comparison=name, case=key[0], step=key[1], tensor=label, **error))
        a, b = values[0], expected[key][0]
        row["top1_equal"] = a.argmax(-1).item() == b.argmax(-1).item()
        if not row["top1_equal"]:
            top = b.float().topk(2).values[0]
            row["reference_top2_margin"] = (top[0] - top[1]).item()
            report["top1_differences"].append({k: v for k, v in row.items() if k != "tensors"})
        report["comparisons"].append(row)
