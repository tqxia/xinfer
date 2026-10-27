"""Error calculations only; callers choose tensor grouping and acceptance limits."""

import torch


def row_errors(actual, expected):
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    delta = actual - expected
    return dict(max_abs=delta.abs().max().item(),
                relative_l2=(delta.norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-12)).max().item(),
                peak_relative=(delta.abs().amax(dim=-1) / expected.abs().amax(dim=-1).clamp_min(1e-12)).max().item())


def bits_equal(actual, expected):
    return actual.dtype == expected.dtype and torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


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
