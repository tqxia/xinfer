"""Qwen3 policy: bounded logit drift, diagnostic HF cache/token drift."""

from compiler.common.tests.validation import (
    DEFAULT_RELATIVE_LIMIT, compare_tensors, relative_limits, tensor_metrics,
    token_differences, within_limits,
)

# HF cache differences and token choices are reported separately; neither changes
# the native/Python bitwise contract.
MAX_RELATIVE_ERROR = DEFAULT_RELATIVE_LIMIT


def ids_from(path):
    return [int(word) for word in path.read_text().split()]


def error_metrics(actual, expected):
    return tensor_metrics(actual, expected, logits=True)


def within_logit_limits(metrics):
    return within_limits(metrics, MAX_RELATIVE_ERROR)


def compare(actual, expected, exact=False):
    return compare_tensors(actual, expected, limit=MAX_RELATIVE_ERROR, exact=exact, logits=True)


def policy():
    return dict(limits=relative_limits(MAX_RELATIVE_ERROR),
                hf_cache_and_token_differences="reported",
                native_python_aoti="bitwise",
                finite_values_and_cache_state="required")


def model_passed(report):
    return not report["failures"]


def reference_limits_passed(report):
    return not any(report[key] for key in (
        "failures", "hf_cache_differences", "hf_token_differences", "hf_greedy_differences"))
