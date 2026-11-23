"""Numerical, ownership and dynamic-batch AOTI contracts for CUDA projections."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.fx.experimental import _config as symbolic_config

from compiler.common.kernels.projection_gemv import MAX_BATCH_SIZE, projection_gemv
from compiler.common.tests.tensors import bits_equal, cpu_copy, row_errors


DTYPES = (torch.bfloat16, torch.float32)
LIMITS = {torch.bfloat16: 0.01, torch.float32: 0.0001}
ABS_LIMITS = {torch.bfloat16: 0.0, torch.float32: 1e-7}
SHAPES = ((8224, 1024), (1024, 2048), (7168, 1024), (1024, 3584), (5120, 1024))


def inputs(n, k, dtype, *, batch=1, binary=False, rank=2, zero_stride=False):
    generator = torch.Generator().manual_seed(35 + batch)
    x = torch.randn(batch, k, generator=generator).to(dtype)
    weight = (torch.randn(n, k, generator=generator) / k**0.5).to(dtype)
    if binary:
        offsets = torch.arange(batch)[:, None] * 3 + torch.arange(k)[None, :]
        x = ((offsets % 11 - 5) / 16).to(dtype)
        weight = ((torch.arange(n * k).view(n, k) % 9 - 4) / 16).to(dtype)
    if zero_stride and batch > 1:
        x = x[:1].expand(batch, k).contiguous()
    # An independent FP64 CPU sum avoids using the candidate's FP32 reduction.
    expected = torch.mm(x.double(), weight.double().t()).float()
    shape = (batch,) + (1,) * (rank - 2) + (k,)
    if zero_stride:
        strides = (0,) * (rank - 1) + (1,)
        cuda_x = x[:1].cuda().as_strided(shape, strides)
    else:
        strides = (k,) * (rank - 1) + (1,)
        cuda_x = x.cuda().as_strided(shape, strides)
    return cuda_x, weight.cuda(), expected.view(*shape[:-1], n)


class Projection(nn.Module):
    def forward(self, x, weight):
        return projection_gemv(x, weight)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ProjectionCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous = torch.backends.cuda.matmul.allow_tf32
        cls.addClassCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", previous)
        torch.backends.cuda.matmul.allow_tf32 = False

    def assert_output(self, actual, x, weight, expected, saved):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, x.dtype)
        self.assertEqual(actual.device, x.device)
        self.assertTrue(actual.is_contiguous())
        self.assertNotIn(
            actual.untyped_storage().data_ptr(),
            (x.untyped_storage().data_ptr(), weight.untyped_storage().data_ptr()),
        )
        batch = expected.shape[0]
        observed = actual.float().cpu().reshape(batch, -1)
        error = row_errors(observed, expected.reshape(batch, -1))
        self.assertTrue(
            max(error["relative_l2"], error["peak_relative"]) <= LIMITS[x.dtype]
            or error["max_abs"] <= ABS_LIMITS[x.dtype],
            str(error),
        )
        self.assertTrue(all(bits_equal(a, b) for a, b in zip(cpu_copy((x, weight)), saved)))

    @torch.inference_mode()
    def test_actual_projection_geometries_against_independent_control(self):
        for dtype in DTYPES:
            for index, (n, k) in enumerate(SHAPES):
                for batch in (1, 3, MAX_BATCH_SIZE):
                    with self.subTest(dtype=dtype, n=n, k=k, batch=batch):
                        x, weight, expected = inputs(
                            n, k, dtype, batch=batch, rank=3, zero_stride=batch == 1 and index != 2
                        )
                        saved = cpu_copy((x, weight))
                        self.assert_output(projection_gemv(x, weight), x, weight, expected, saved)

    @torch.inference_mode()
    def test_batch_rows_match_singleton_bitwise_and_stay_isolated(self):
        for dtype in DTYPES:
            for n, k in ((8224, 1024), (1024, 3584)):
                x_full, weight, expected = inputs(n, k, dtype, batch=MAX_BATCH_SIZE, rank=3)
                baseline = projection_gemv(x_full, weight)
                self.assert_output(baseline, x_full, weight, expected, cpu_copy((x_full, weight)))
                for row in range(MAX_BATCH_SIZE):
                    single = projection_gemv(x_full[row : row + 1], weight)
                    self.assertTrue(bits_equal(baseline[row : row + 1], single))
                perturbed = x_full.clone()
                perturbed[0].mul_(-1.5).add_(0.25)
                changed = projection_gemv(perturbed, weight)
                self.assertTrue(bits_equal(changed[1:], baseline[1:]))

    @torch.inference_mode()
    def test_masked_tails_and_exact_bfloat16_output_rounding(self):
        for dtype in DTYPES:
            for k in (33, 2049):
                for batch in (1, 5):
                    with self.subTest(dtype=dtype, k=k, batch=batch):
                        x, weight, expected = inputs(
                            7, k, dtype, batch=batch, binary=True, zero_stride=batch == 1
                        )
                        saved = cpu_copy((x, weight))
                        actual = projection_gemv(x, weight)
                        self.assert_output(actual, x, weight, expected, saved)
                        self.assertTrue(bits_equal(actual.cpu(), expected.to(dtype)))
                        x.zero_()
                        saved = cpu_copy((x, weight))
                        self.assert_output(
                            projection_gemv(x, weight),
                            x,
                            weight,
                            torch.zeros_like(expected),
                            saved,
                        )

    @torch.inference_mode()
    def test_offset_and_strided_layouts_repeats_and_nondefault_stream(self):
        for dtype in DTYPES:
            for rank in (2, 3):
                for batch, zero_stride in ((1, False), (1, True), (4, False), (4, True)):
                    with self.subTest(dtype=dtype, rank=rank, batch=batch, zero_stride=zero_stride):
                        x, weight, expected = inputs(
                            17, 2048, dtype, batch=batch, rank=rank, zero_stride=zero_stride
                        )
                        row_stride = 0 if zero_stride else 2049
                        backing = torch.empty(
                            (1 if zero_stride else batch) * 2049 + 1, device="cuda", dtype=dtype
                        )
                        if zero_stride:
                            backing[1:2049].copy_(x[0].reshape(-1))
                        else:
                            for row in range(batch):
                                start = 1 + row * row_stride
                                backing[start : start + 2048].copy_(x[row].reshape(-1))
                        strides = (row_stride,) * (rank - 1) + (1,)
                        x = backing[1:].as_strided(x.shape, strides)
                        saved = cpu_copy((x, weight))
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            actual = projection_gemv(x, weight)
                            repeat = projection_gemv(x, weight)
                        stream.synchronize()
                        self.assert_output(actual, x, weight, expected, saved)
                        self.assertTrue(bits_equal(actual, repeat))
                        self.assertNotEqual(
                            actual.untyped_storage().data_ptr(), repeat.untyped_storage().data_ptr()
                        )

    @torch.inference_mode()
    def test_invalid_batch_and_token_shapes_are_rejected(self):
        weight = torch.empty((16, 32), device="cuda", dtype=torch.bfloat16)
        for shape in ((0, 1, 32), (MAX_BATCH_SIZE + 1, 1, 32), (1, 2, 32), (2, 2, 32)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                projection_gemv(torch.empty(shape, device="cuda", dtype=torch.bfloat16), weight)

    @torch.inference_mode()
    def test_packaging_numerical_controls_and_bfloat16_bitwise(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"AOTI_RUNTIME_CHECK_INPUTS": "1"}),
        ):
            for dtype in DTYPES:
                with self.subTest(dtype=dtype):
                    sample_x, weight, _ = inputs(7, 33, dtype, batch=2, rank=3)
                    batch_dim = torch.export.Dim("batch", min=1, max=MAX_BATCH_SIZE)
                    with symbolic_config.patch(backed_size_oblivious=True):
                        exported = torch.export.export(
                            Projection(),
                            (sample_x, weight),
                            strict=True,
                            dynamic_shapes=({0: batch_dim}, None),
                        )
                        first = next(n for n in exported.graph.nodes if n.op == "placeholder")
                        bounds = exported.range_constraints[first.meta["val"].shape[0].node.expr]
                        self.assertEqual(
                            (int(bounds.lower), int(bounds.upper)), (1, MAX_BATCH_SIZE)
                        )
                        path = Path(directory) / f"projection-{dtype}.pt2"
                        torch._inductor.aoti_compile_and_package(
                            exported,
                            package_path=str(path),
                            inductor_configs={"emulate_precision_casts": True},
                        )
                    model = torch._inductor.aoti_load_package(str(path))
                    try:
                        for batch in range(1, MAX_BATCH_SIZE + 1):
                            with self.subTest(dtype=dtype, batch=batch):
                                x, weight, expected = inputs(7, 33, dtype, batch=batch, rank=3)
                                saved = cpu_copy((x, weight))
                                actual = model(x, weight)
                                self.assert_output(actual, x, weight, expected, saved)
                                self.assertTrue(bits_equal(actual, model(x, weight)))
                                # Captured AOTI options can enable FMA despite the eager
                                # launch flag; FP32 keeps controls, BF16 stays bitwise.
                                if dtype == torch.bfloat16:
                                    self.assertTrue(bits_equal(actual, projection_gemv(x, weight)))
                        for invalid_batch in (0, MAX_BATCH_SIZE + 1):
                            bad_x = torch.empty((invalid_batch, 1, 33), device="cuda", dtype=dtype)
                            with self.assertRaises((RuntimeError, ValueError)):
                                model(bad_x, weight)
                    finally:
                        del model


if __name__ == "__main__":
    unittest.main()
