"""FP8 checkpoint preservation, kernel contracts and dynamic projection packaging."""

import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file
from torch.fx.experimental import _config as symbolic_config

from compiler.common.adapters.finegrained_fp8 import (
    install_fp8,
    load_weights,
    validate_quantization,
)
from compiler.common.kernels.gemm import finegrained_fp8
from compiler.common.kernels.gemm.finegrained_fp8 import (
    KERNEL_REPO,
    KERNEL_REVISION,
    linear,
    load_fp8_kernels,
)
from compiler.common.tests.tensors import bits_equal
from compiler.common.compile import compile_shared
from compiler.common.weights import SharedWeights


class QuantizationConfig(unittest.TestCase):
    def test_supported_config_and_unsupported_quantizers(self):
        quantization = dict(
            quant_method="fp8",
            fmt="e4m3",
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
        validate_quantization(quantization)
        for field, value in (
            ("fmt", "e5m2"),
            ("activation_scheme", "static"),
            ("weight_block_size", [64, 128]),
            ("dequantize", True),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "dynamic E4M3"):
                validate_quantization(dict(quantization, **{field: value}))


class FP8KernelContracts(unittest.TestCase):
    def test_load_fp8_kernels_pins_immutable_revision_and_caches(self):
        fake_kernel = SimpleNamespace(
            matmul_2d=lambda *args, **kwargs: None,
            __file__=f"/cache/kernels/snapshots/{KERNEL_REVISION}/build/torch-cuda/__init__.py",
        )
        get_kernel = Mock(return_value=fake_kernel)
        with (
            patch.object(finegrained_fp8, "_KERNEL", None),
            patch.dict(sys.modules, {"kernels": SimpleNamespace(get_kernel=get_kernel)}),
        ):
            self.assertIs(load_fp8_kernels(), fake_kernel)
            self.assertIs(load_fp8_kernels(), fake_kernel)
        get_kernel.assert_called_once_with(KERNEL_REPO, revision=KERNEL_REVISION, backend="cuda")

    def test_load_fp8_kernels_rejects_missing_entrypoint_and_mismatched_snapshot(self):
        cases = (
            (SimpleNamespace(), "missing its projection operators"),
            (
                SimpleNamespace(
                    matmul_2d=lambda *args, **kwargs: None,
                    __file__="/cache/kernels/snapshots/0000000000000000000000000000000000000000/build/torch-cuda/__init__.py",
                ),
                f"Expected FP8 kernel revision {KERNEL_REVISION}",
            ),
        )
        for fake_kernel, message in cases:
            get_kernel = Mock(return_value=fake_kernel)
            with (
                self.subTest(message=message),
                patch.object(finegrained_fp8, "_KERNEL", None),
                patch.dict(sys.modules, {"kernels": SimpleNamespace(get_kernel=get_kernel)}),
                self.assertRaisesRegex(ValueError, message),
            ):
                load_fp8_kernels()

    def test_linear_validates_weight_scale_and_input_contracts_without_cuda(self):
        valid_w = torch.zeros((256, 128), dtype=torch.float8_e4m3fn)
        valid_s = torch.ones((2, 1), dtype=torch.bfloat16)
        valid_x = torch.zeros((1, 4, 128), dtype=torch.bfloat16)

        bad_cases = (
            (
                valid_x,
                torch.zeros((256, 128), dtype=torch.bfloat16),
                valid_s,
                "contiguous E4M3",
            ),
            (
                valid_x,
                torch.zeros((1, 256, 128), dtype=torch.float8_e4m3fn),
                valid_s,
                "contiguous E4M3",
            ),
            (
                valid_x,
                torch.zeros((128, 256), dtype=torch.float8_e4m3fn).t(),
                valid_s,
                "contiguous E4M3",
            ),
            (
                valid_x,
                torch.zeros((192, 128), dtype=torch.float8_e4m3fn),
                valid_s,
                "multiples of 128",
            ),
            (
                valid_x,
                valid_w,
                torch.ones((1, 1), dtype=torch.bfloat16),
                "128x128 blocks",
            ),
            (
                valid_x,
                valid_w,
                torch.ones((2, 1), dtype=torch.float16),
                "BF16/FP32 tensors",
            ),
            (
                valid_x,
                valid_w,
                valid_s,
                "contiguous CUDA BF16",
            ),
            (
                torch.zeros((1, 4, 256), dtype=torch.bfloat16),
                valid_w,
                valid_s,
                "contiguous CUDA BF16",
            ),
        )
        for x, weight, scale, message in bad_cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                linear(x, weight, scale)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class FP8Projection(unittest.TestCase):
    @torch.inference_mode()
    def test_hf_adapter_and_shared_aoti_preserve_quantized_weights(self):
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(38)
        kernel = load_fp8_kernels()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            weight = torch.randn(256, 128).to(torch.float8_e4m3fn)
            scale = torch.tensor([[0.01], [0.02]], dtype=torch.bfloat16)
            save_file({"0.weight": weight, "0.weight_scale_inv": scale}, root / "text.safetensors")
            with torch.device("meta"):
                model = torch.nn.Sequential(torch.nn.Linear(128, 256, bias=False))
            load_weights(model, root / "text.safetensors")
            self.assertEqual(os.environ.get("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR"), "1")
            self.assertTrue(bits_equal(model[0].weight.cpu(), weight))
            self.assertTrue(bits_equal(model[0].weight_scale_inv.cpu(), scale))
            samples = {
                (batch, tokens): torch.randn(
                    batch, tokens, 128, device="cuda", dtype=torch.bfloat16
                )
                for batch in (1, 2, 8)
                for tokens in (1, 2, 97, 98, 512)
            }
            expected = {key: model(x).cpu() for key, x in samples.items()}
            install_fp8(model)
            for key, x in samples.items():
                self.assertTrue(bits_equal(model(x).cpu(), expected[key]), key)

            # Verify local Triton kernel also matches pinned upstream matmul_2d with FP32 scales.
            fp32_scale = scale.to(device="cuda", dtype=torch.float32)
            fp8_weight = weight.to("cuda")
            for key in ((1, 1), (2, 97), (8, 512)):
                x = samples[key]
                ref = kernel.matmul_2d(x, fp8_weight, fp32_scale, [128, 128], output_dtype=x.dtype)
                self.assertTrue(bits_equal(linear(x, fp8_weight, fp32_scale).cpu(), ref.cpu()), key)

            def entry(kind, staging):
                batch = torch.export.Dim("batch", min=1, max=8)
                shape = {0: batch}
                if kind == "prefill":
                    shape[1] = torch.export.Dim("tokens", min=2, max=512)
                sample = samples[2, 2 if kind == "prefill" else 1]
                return torch.export.export(model, (sample,), dynamic_shapes=(shape,), strict=True)

            with symbolic_config.patch(backed_size_oblivious=True):
                contract = compile_shared(
                    root / "bundle",
                    entry,
                    identity=dict(model="fp8-projection"),
                    inductor_configs={"emulate_precision_casts": True},
                )
            pool = SharedWeights(root / "bundle", contract)
            self.assertIn("float8_e4m3fn", {s["dtype"] for s in pool.manifest["storages"].values()})
            for key, x in samples.items():
                function = pool.load("decode" if key[1] == 1 else "prefill")
                self.assertTrue(bits_equal(function(x).cpu(), expected[key]), key)
