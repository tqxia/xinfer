"""Numerical, ownership and AOTI contracts for singleton CUDA projections."""

from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from compiler.common.kernels.projection_gemv import projection_gemv
from compiler.common.tests.tensors import bits_equal, cpu_copy, row_errors


DTYPES = (torch.bfloat16, torch.float32)
LIMITS = {torch.bfloat16: 0.01, torch.float32: 0.0001}
ABS_LIMITS = {torch.bfloat16: 0.0, torch.float32: 1e-7}
SHAPES = ((8224, 1024), (1024, 2048), (7168, 1024), (1024, 3584), (5120, 1024))


def inputs(n, k, dtype, *, binary=False, rank=2, zero_stride=False):
    generator = torch.Generator().manual_seed(35)
    x = torch.randn(1, k, generator=generator).to(dtype)
    weight = (torch.randn(n, k, generator=generator) / k**0.5).to(dtype)
    if binary:
        x = ((torch.arange(k) % 11 - 5) / 16).to(dtype).view(1, k)
        weight = ((torch.arange(n * k).view(n, k) % 9 - 4) / 16).to(dtype)
    # An independent FP64 CPU sum avoids using the candidate's FP32 reduction.
    expected = torch.mm(x.double(), weight.double().t()).float()
    shape = (1,) * (rank - 1) + (k,)
    strides = ((0 if zero_stride else k),) * (rank - 1) + (1,)
    return x.cuda().as_strided(shape, strides), weight.cuda(), expected.view(*shape[:-1], n)


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
        observed = actual.float().cpu().reshape(1, -1)
        error = row_errors(observed, expected.reshape(1, -1))
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
                with self.subTest(dtype=dtype, n=n, k=k):
                    x, weight, expected = inputs(n, k, dtype, rank=3, zero_stride=index != 2)
                    saved = cpu_copy((x, weight))
                    self.assert_output(projection_gemv(x, weight), x, weight, expected, saved)

    @torch.inference_mode()
    def test_masked_tails_and_exact_bfloat16_output_rounding(self):
        for dtype in DTYPES:
            for k in (33, 2049):
                with self.subTest(dtype=dtype, k=k):
                    x, weight, expected = inputs(7, k, dtype, binary=True, zero_stride=True)
                    saved = cpu_copy((x, weight))
                    actual = projection_gemv(x, weight)
                    self.assert_output(actual, x, weight, expected, saved)
                    self.assertTrue(bits_equal(actual.cpu(), expected.to(dtype)))
                    x.zero_()
                    saved = cpu_copy((x, weight))
                    self.assert_output(
                        projection_gemv(x, weight), x, weight, torch.zeros_like(expected), saved
                    )

    @torch.inference_mode()
    def test_offset_singleton_layouts_repeats_and_nondefault_stream(self):
        for dtype in DTYPES:
            for rank in (2, 3):
                for zero_stride in (False, True):
                    with self.subTest(dtype=dtype, rank=rank, zero_stride=zero_stride):
                        x, weight, expected = inputs(17, 2048, dtype, rank=rank)
                        backing = torch.empty(2049, device="cuda", dtype=dtype)
                        backing[1:].copy_(x.reshape(-1))
                        strides = ((0 if zero_stride else 2048),) * (rank - 1) + (1,)
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
    def test_packaging_numerical_controls_and_bfloat16_bitwise(self):
        with tempfile.TemporaryDirectory() as directory:
            for dtype in DTYPES:
                with self.subTest(dtype=dtype):
                    x, weight, expected = inputs(7, 33, dtype)
                    saved = cpu_copy((x, weight))
                    exported = torch.export.export(Projection(), (x, weight), strict=True)
                    path = Path(directory) / f"projection-{dtype}.pt2"
                    torch._inductor.aoti_compile_and_package(
                        exported,
                        package_path=str(path),
                        inductor_configs={"emulate_precision_casts": True},
                    )
                    model = torch._inductor.aoti_load_package(str(path))
                    actual = model(x, weight)
                    self.assert_output(actual, x, weight, expected, saved)
                    self.assertTrue(bits_equal(actual, model(x, weight)))
                    # Captured AOTI options can enable FMA despite the eager launch flag.
                    # FP32 keeps numerical controls; the measured BF16 path stays bitwise.
                    if dtype == torch.bfloat16:
                        self.assertTrue(bits_equal(actual, projection_gemv(x, weight)))
                    del model


if __name__ == "__main__":
    unittest.main()
