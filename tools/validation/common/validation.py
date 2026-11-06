"""Shared validation measurements and report records; callers own acceptance policy."""

import torch

from compiler.common.artifacts import require
from compiler.common.tests.tensors import bits_equal, row_errors
from compiler.common.tests.tensors import cpu_copy as cpu_copy

# The full-model logit/state checks share this default. Kernel-level tests pass
# their independently chosen limits explicitly or use row_errors directly.
DEFAULT_RELATIVE_LIMIT = 0.05


def tensor_metrics(actual, expected, *, rows=None, logits=False):
    """Measure original-tensor bit parity and errors over caller-selected rows."""
    require(actual.shape == expected.shape, "Tensor shapes differ")
    error = row_errors(*(rows if rows is not None else (actual, expected)))
    error["bitwise_equal"] = bits_equal(actual, expected)
    if logits:
        error.update(
            top1_agreement=(actual.argmax(-1) == expected.argmax(-1)).sum().item(),
            rows=actual.shape[0],
        )
    return error


def relative_limits(limit=DEFAULT_RELATIVE_LIMIT):
    return dict(relative_l2=limit, peak_relative=limit)


def within_limits(metrics, limit=DEFAULT_RELATIVE_LIMIT):
    return max(metrics["relative_l2"], metrics["peak_relative"]) <= limit


def record_failure(failures, metrics, *, limit=DEFAULT_RELATIVE_LIMIT, required=True, **context):
    """Record a bounded-error failure only when this comparison is required."""
    failed = required and not within_limits(metrics, limit)
    if failed:
        failures.append(dict(**context, **metrics))
    return failed


def compare_tensors(
    actual, expected, *, limit=DEFAULT_RELATIVE_LIMIT, exact=False, rows=None, logits=False
):
    metrics = tensor_metrics(actual, expected, rows=rows, logits=logits)
    require(metrics["bitwise_equal"] if exact else within_limits(metrics, limit), metrics)
    return metrics


def token_differences(actual, expected):
    """Explain top-1 drift for logits produced with identical input tokens."""
    row_errors(actual, expected)
    require(actual.ndim == 2 and actual.shape[-1] >= 2, "Expected rows with at least two logits")
    return [
        dict(
            row=i,
            reference_id=expected[i].argmax().item(),
            actual_id=actual[i].argmax().item(),
            reference_top2_margin=expected[i].float().topk(2).values.diff().abs().item(),
            reference_selected_token_gap=(
                expected[i].float().max() - expected[i, actual[i].argmax()].float()
            ).item(),
            actual_top2_margin=actual[i].float().topk(2).values.diff().abs().item(),
        )
        for i in torch.where(actual.argmax(-1) != expected.argmax(-1))[0].tolist()
    ]


def raw_bytes(tensor):
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def compare_file(path, tensors):
    with path.open("rb") as stream:
        for tensor in tensors:
            data = raw_bytes(tensor)
            require(stream.read(len(data)) == data, f"Native/Python bitwise mismatch: {path}")
        require(not stream.read(1), f"Extra native output: {path}")


def read_ids(path):
    return [int(word) for word in path.read_text().split()]
