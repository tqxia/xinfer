"""Shared validation measurements and report records; callers own acceptance policy."""

import torch

# The full-model logit/state checks share this default. Kernel-level tests pass
# their independently chosen limits explicitly or use row_errors directly.
DEFAULT_RELATIVE_LIMIT = 0.05


def row_errors(actual, expected):
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    delta = actual - expected
    return dict(max_abs=delta.abs().max().item(),
                relative_l2=(delta.norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-12)).max().item(),
                peak_relative=(delta.abs().amax(dim=-1) / expected.abs().amax(dim=-1).clamp_min(1e-12)).max().item())


def bits_equal(actual, expected):
    return actual.shape == expected.shape and actual.dtype == expected.dtype and torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def tensor_metrics(actual, expected, *, rows=None, logits=False):
    """Measure original-tensor bit parity and errors over caller-selected rows."""
    assert actual.shape == expected.shape
    error = row_errors(*(rows if rows is not None else (actual, expected)))
    error["bitwise_equal"] = bits_equal(actual, expected)
    if logits:
        error.update(top1_agreement=(actual.argmax(-1) == expected.argmax(-1)).sum().item(),
                     rows=actual.shape[0])
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


def compare_tensors(actual, expected, *, limit=DEFAULT_RELATIVE_LIMIT, exact=False,
                    rows=None, logits=False):
    metrics = tensor_metrics(actual, expected, rows=rows, logits=logits)
    assert metrics["bitwise_equal"] if exact else within_limits(metrics, limit), metrics
    return metrics


def token_differences(actual, expected):
    """Explain top-1 drift for logits produced with identical input tokens."""
    row_errors(actual, expected)
    assert actual.ndim == 2 and actual.shape[-1] >= 2
    return [dict(row=i, reference_id=expected[i].argmax().item(),
                 actual_id=actual[i].argmax().item(),
                 reference_top2_margin=expected[i].float().topk(2).values.diff().abs().item(),
                 reference_selected_token_gap=(expected[i].float().max() - expected[i, actual[i].argmax()].float()).item(),
                 actual_top2_margin=actual[i].float().topk(2).values.diff().abs().item())
            for i in torch.where(actual.argmax(-1) != expected.argmax(-1))[0].tolist()]


def cpu_copy(tensors):
    assert all(torch.isfinite(t).all() for t in tensors)
    return tuple(t.cpu().clone() for t in tensors)


def raw_bytes(tensor):
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def compare_file(path, tensors):
    with path.open("rb") as stream:
        for tensor in tensors:
            data = raw_bytes(tensor)
            assert stream.read(len(data)) == data, f"Native/Python bitwise mismatch: {path}"
        assert not stream.read(1), f"Extra native output: {path}"
