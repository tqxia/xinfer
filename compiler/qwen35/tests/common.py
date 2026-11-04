"""Full-model fixtures and per-state acceptance policy shared by Qwen3.5 tests."""

from compiler.common.metadata import state_name
from compiler.common.tests.validation import (
    DEFAULT_RELATIVE_LIMIT,
    record_failure,
    relative_limits,
    tensor_metrics,
    token_differences,
)

LIMIT = DEFAULT_RELATIVE_LIMIT


def metrics(actual, expected, index):
    assert actual.shape == expected.shape
    width = expected.shape[-1] if index < 2 else expected.shape[-2] * expected.shape[-1]
    a, b = (t.float().reshape(-1, width) for t in (actual, expected))
    return tensor_metrics(actual, expected, rows=(a, b))


def fixtures(tokenizer):
    paragraph = tokenizer.encode(
        "The capital of France is Paris. A small compiler can run a language model. ",
        add_special_tokens=False,
    )
    stream = (paragraph * (1024 // len(paragraph) + 1))[:1024]

    def case(name, prefix, decode):
        sizes = [prefix] + [1] * decode
        offset, steps = 0, []
        for length in sizes:
            steps.append(stream[offset : offset + length])
            offset += length
        return dict(name=name, prefix=prefix, steps=steps)

    cases = [
        case("decode_zero", 1, 3),
        case("prefill_decode", 64, 8),
        case("remainder_65", 65, 4),
        case("two_prefills", 128, 4),
        case("long_prefix", 512, 8),
        case("repeat", 64, 8),
    ]
    cases += [
        case(f"dynamic_{length}", length, 2)
        for length in (2, 22, 31, 32, 33, 63, 96, 97, 98, 127, 129, 511)
    ]
    sizes = (96, 98, 97, 99, 1)
    offset, steps = 0, []
    for length in sizes:
        steps.append(stream[offset : offset + length])
        offset += length
    mixed = dict(name="mixed_dispatch", steps=steps)
    cases.extend((mixed, dict(mixed, name="mixed_dispatch_repeat")))
    return cases


def compare(report, name, actual, expected, specs, enforce=True):
    for key, values in actual.items():
        row = dict(comparison=name, case=key[0], step=key[1], tensors={})
        for i, (a, b) in enumerate(zip(values, expected[key])):
            label = "logits" if i == 0 else state_name(specs[i - 1])
            axis = 0 if i == 0 else (1 if specs[i - 1]["kind"] == "conv" else 2)
            error = metrics(a, b, axis)
            row["tensors"][label] = error
            record_failure(
                report["failures"],
                error,
                limit=LIMIT,
                required=enforce,
                comparison=name,
                case=key[0],
                step=key[1],
                tensor=label,
            )
        a, b = values[0], expected[key][0]
        differences = token_differences(a, b)
        row["top1_equal"] = not differences
        if differences:
            row["reference_top2_margin"] = differences[0]["reference_top2_margin"]
            row["token_differences"] = differences
            report["top1_differences"].append({k: v for k, v in row.items() if k != "tensors"})
        report["comparisons"].append(row)


def policy(*, end_to_end=False):
    return dict(
        limits=relative_limits(LIMIT),
        hf_state_errors="required",
        hf_token_differences="required" if end_to_end else "reported",
        native_python_aoti="bitwise" if end_to_end else "checked_by_validate_end_to_end",
        finite_values_and_cache_state="required",
    )


def model_passed(report):
    return not report["failures"]


def end_to_end_passed(report):
    return not (
        report["failures"]
        or report["greedy_failures"]
        or any(row["comparison"].startswith("aoti_vs_hf") for row in report["top1_differences"])
    )
