"""Qwen3 validation policy: bounded logit drift, diagnostic HF cache/token drift."""

import torch

from compiler.common.validation import bits_equal, row_errors


# Keep the existing per-row logit limits. HF cache differences and token choices
# are reported separately; neither changes the native/Python bitwise contract.
MAX_RELATIVE_ERROR = 0.05


def ids_from(path):
    return [int(word) for word in path.read_text().split()]


def error_metrics(actual, expected):
    error = row_errors(actual, expected)  # Also rejects nonfinite inputs.
    return {"max_abs": error["max_abs"], "max_relative_l2": error["relative_l2"],
            "max_peak_relative": error["peak_relative"],
            "top1_agreement": (actual.argmax(-1) == expected.argmax(-1)).sum().item(),
            "rows": actual.shape[0]}


def within_logit_limits(metrics):
    return max(metrics["max_relative_l2"], metrics["max_peak_relative"]) <= MAX_RELATIVE_ERROR


def compare(actual, expected, exact=False):
    metrics = error_metrics(actual, expected)
    if exact:
        assert bits_equal(actual, expected), metrics
    else:
        assert within_logit_limits(metrics), metrics
    return metrics


def token_differences(actual, expected):
    """Explain top-1 drift only for logits produced by identical input tokens."""
    error_metrics(actual, expected)
    return [dict(row=i, hf_id=expected[i].argmax().item(),
                 compiled_id=actual[i].argmax().item(),
                 hf_top2_margin=expected[i].topk(2).values.diff().abs().item(),
                 hf_selected_token_gap=(expected[i].max() - expected[i, actual[i].argmax()]).item(),
                 compiled_top2_margin=actual[i].topk(2).values.diff().abs().item())
            for i in torch.where(actual.argmax(-1) != expected.argmax(-1))[0].tolist()]


def policy():
    return dict(logit_relative_l2_limit=MAX_RELATIVE_ERROR,
                logit_peak_relative_limit=MAX_RELATIVE_ERROR,
                hf_cache_and_token_differences="reported",
                native_python_aoti="bitwise",
                finite_values_and_cache_state="required")
