"""FP8 checkpoint preservation and dynamic projection packaging contracts."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from safetensors.torch import save_file
from torch.fx.experimental import _config as symbolic_config

from compiler.common.kernels.finegrained_fp8 import load_fp8_kernels
from compiler.common.tests.tensors import bits_equal
from compiler.common.weights import SharedWeights, compile_shared
from compiler.qwen35.config import configure
from compiler.qwen35.fp8_adapter import install_fp8, load_weights, validate_quantization
from compiler.qwen35.model import _text_config


class QuantizationConfig(unittest.TestCase):
    def test_untied_fp8_config_and_unsupported_quantizers(self):
        quantization = dict(
            quant_method="fp8",
            fmt="e4m3",
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps(
                    dict(
                        text_config=dict(tie_word_embeddings=False),
                        quantization_config=quantization,
                    )
                )
            )
            self.assertFalse(_text_config(path).tie_word_embeddings)
        for field, value in (
            ("fmt", "e5m2"),
            ("activation_scheme", "static"),
            ("weight_block_size", [64, 128]),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "dynamic E4M3"):
                validate_quantization(dict(quantization, **{field: value}))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class FP8Projection(unittest.TestCase):
    @torch.inference_mode()
    def test_hf_adapter_and_shared_aoti_preserve_quantized_weights(self):
        configure()
        torch.manual_seed(38)
        load_fp8_kernels()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            weight = torch.randn(256, 128).to(torch.float8_e4m3fn)
            scale = torch.tensor([[0.01], [0.02]], dtype=torch.bfloat16)
            save_file({"0.weight": weight, "0.weight_scale_inv": scale}, root / "text.safetensors")
            with torch.device("meta"):
                model = torch.nn.Sequential(torch.nn.Linear(128, 256, bias=False))
            load_weights(model, root / "text.safetensors")
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

            def entry(kind, staging):
                batch = torch.export.Dim("batch", min=1, max=8)
                shape = {0: batch}
                if kind == "prefill":
                    shape[1] = torch.export.Dim("tokens", min=2, max=512)
                sample = samples[2, 2 if kind == "prefill" else 1]
                return torch.export.export(
                    model, (sample,), dynamic_shapes=(shape,), strict=True
                ), None

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
