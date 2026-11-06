"""Tensor measurements shared by unit tests and compiled validation tools."""

import torch

from compiler.common.artifacts import require


def row_errors(actual, expected):
    require(actual.shape == expected.shape, "Tensor shapes differ")
    require(
        torch.isfinite(actual).all() and torch.isfinite(expected).all(),
        "Nonfinite validation tensors",
    )
    delta = actual - expected
    return dict(
        max_abs=delta.abs().max().item(),
        relative_l2=(delta.norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-12)).max().item(),
        peak_relative=(delta.abs().amax(dim=-1) / expected.abs().amax(dim=-1).clamp_min(1e-12))
        .max()
        .item(),
    )


def bits_equal(actual, expected):
    return (
        actual.shape == expected.shape
        and actual.dtype == expected.dtype
        and torch.equal(
            actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
        )
    )


def cpu_copy(tensors):
    require(all(torch.isfinite(t).all() for t in tensors), "Nonfinite validation tensors")
    return tuple(t.cpu().clone() for t in tensors)
