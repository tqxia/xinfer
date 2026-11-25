"""Bundled native operators load through ordinary Python AOTI packages."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch, sentinel
import zipfile

import torch

from compiler.common.kernels.flash_attention.attention import (
    attention_c_shims,
    attention_library,
    batched_attention_metadata,
    load_attention_ops,
)
from compiler.common.package import _linking_environment, compile_package


WORKER = textwrap.dedent(
    """
    import gc
    import json
    import sys
    from pathlib import Path

    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def inputs(tokens, prefix):
        torch.manual_seed(23 + tokens + prefix)
        options = dict(device='cuda', dtype=torch.bfloat16)
        offsets = torch.tensor([[0, tokens], [0, prefix]], dtype=torch.int32)
        return (
            torch.randn(1, 16, tokens, 128, **options),
            torch.randn(1, 8, 1024, 128, **options),
            torch.randn(1, 8, 1024, 128, **options),
            offsets,
            offsets.cuda(),
            torch.zeros((1, 1), device='cuda', dtype=torch.int32),
        )

    def bits_equal(actual, expected):
        return torch.equal(
            actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
        )

    def check_math(actual, q, k, v, tokens, prefix):
        positions = torch.arange(prefix - tokens, prefix, device='cuda')
        mask = torch.arange(prefix, device='cuda')[None, :] <= positions[:, None]
        with sdpa_kernel(SDPBackend.MATH):
            expected = F.scaled_dot_product_attention(
                q.float(),
                k[:, :, :prefix].float(),
                v[:, :, :prefix].float(),
                attn_mask=mask,
                enable_gqa=True,
                scale=128**-0.5,
            )
        error = (actual.float() - expected).norm() / expected.norm()
        if error.item() >= 0.01:
            raise AssertionError(f'Attention error exceeds bound: {error.item()}')

    directory, build_library, order = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(',')
    checks, library = [], None
    for cycle in range(3):
        models = {
            kind: torch._inductor.aoti_load_package(str(directory / f'{kind}.pt2'))
            for kind in order
        }
        for kind, model in models.items():
            for prefix, prefill_tokens in ((32, 2), (512, 512), (1024, 16)):
                tokens = 1 if kind == 'decode' else prefill_tokens
                args = inputs(tokens, prefix)
                q, k, v, offsets, cuda_offsets, blocks = args
                saved = [tensor.clone() for tensor in args]
                with torch.inference_mode():
                    output = model(*args)
                    expected = torch.ops.xinfer.batched_flash_attention(*args, 128**-0.5)
                if not bits_equal(output, expected):
                    raise AssertionError(f'Bitwise attention mismatch: {cycle}/{kind}/{prefix}')
                check_math(output, q, k, v, tokens, prefix)
                if not bool(torch.isfinite(output).all()):
                    raise AssertionError('Nonfinite attention output')
                if not all(bits_equal(a, b) for a, b in zip(args, saved)):
                    raise AssertionError('Read-only attention input changed')
                if output.untyped_storage().data_ptr() in {
                    tensor.untyped_storage().data_ptr() for tensor in (q, k, v)
                }:
                    raise AssertionError('Attention output aliases input storage')
                checks.append([cycle, kind, prefix])
        paths = {
            line.split(maxsplit=5)[5].removesuffix(' (deleted)')
            for line in Path('/proc/self/maps').read_text().splitlines()
            if 'libxinfer_attention_ops.so' in line
        }
        if len(paths) != 1:
            raise AssertionError(f'Expected one mapped attention library: {paths}')
        library = paths.pop()
        if 'aotinductor_' not in library or Path(library) == build_library:
            raise AssertionError(f'Attention library was not extracted from the package: {library}')
        del models, model
        gc.collect()
    append = {
        kind: torch._inductor.aoti_load_package(str(directory / f'{kind}.pt2'))
        for kind in ('append', 'append_decode')
    }
    for tokens, position in ((2, 0), (16, 17), (1, 511)):
        prefix = position + tokens
        q, k, v, offsets, cuda_offsets, blocks = inputs(tokens, prefix)
        new_k, new_v = k[:, :, :tokens].clone().mul_(2), v[:, :, :tokens].clone().mul_(3)
        positions = torch.arange(position, prefix, device='cuda')
        readonly = (q, new_k, new_v, positions, offsets, cuda_offsets, blocks)
        saved = [tensor.clone() for tensor in readonly]
        expected_k, expected_v = k.clone(), v.clone()
        expected_k.index_copy_(2, positions, new_k)
        expected_v.index_copy_(2, positions, new_v)
        pointers = (k.data_ptr(), v.data_ptr())
        with torch.inference_mode():
            output, returned_k, returned_v = append[
                'append_decode' if tokens == 1 else 'append'
            ](q, k, v, new_k, new_v, positions, offsets, cuda_offsets, blocks)
            expected = torch.ops.xinfer.batched_flash_attention(
                q, expected_k, expected_v, offsets, cuda_offsets, blocks, 128**-0.5
            )
        if (returned_k.data_ptr(), returned_v.data_ptr()) != pointers:
            raise AssertionError('Returned KV buffers lost caller-owned storage')
        if (k.data_ptr(), v.data_ptr()) != pointers:
            raise AssertionError('Incoming KV pointers changed')
        if not all(bits_equal(a, b) for a, b in zip(readonly, saved)):
            raise AssertionError('Read-only append input changed')
        for actual, expected_cache in ((k, expected_k), (v, expected_v)):
            if not bits_equal(actual, expected_cache):
                raise AssertionError('KV values differ, including untouched positions')
        if not bits_equal(output, expected):
            raise AssertionError('Appended attention differs from eager operator')
    invalid = torch._inductor.aoti_load_package(str(directory / 'invalid_scale.pt2'))
    try:
        with torch.inference_mode():
            invalid(*inputs(1, 32))
    except RuntimeError as error:
        if 'Attention scale must be finite and positive' not in str(error):
            raise
    else:
        raise AssertionError('Native C-shim error did not propagate')
    print(json.dumps(dict(
        order=order, checks=checks, library=library, mutation_checks=3,
        native_error_propagated=True,
    )), flush=True)
    """
)


class PackagePreflight(unittest.TestCase):
    def test_invalid_libraries_fail_before_compilation_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            invalid = directory / "attention.so"
            invalid.touch()
            libraries = [directory / part / "libattention.so" for part in ("one", "two")]
            for library in libraries:
                library.parent.mkdir()
                library.touch()
            destination = directory / "model.pt2"
            for selected, message in (
                ([directory / "libmissing.so"], "Native operator library is missing"),
                ([invalid], "Expected a native library named"),
                (libraries, "Duplicate native library name"),
            ):
                with (
                    self.subTest(libraries=selected),
                    patch.object(torch._inductor, "aot_compile") as compile,
                    self.assertRaisesRegex(ValueError, message),
                ):
                    compile_package(
                        sentinel.exported,
                        destination,
                        inductor_configs={},
                        native_libraries=selected,
                    )
                compile.assert_not_called()
                self.assertFalse(destination.exists())

    def test_link_environment_restores_existing_values_after_failure(self):
        original = {"LIBRARY_PATH": "/existing/lib", "LD_RUN_PATH": "/existing/run"}
        with patch.dict(os.environ, original):
            with self.assertRaisesRegex(RuntimeError, "compile failed"):
                with _linking_environment([Path("/build/libattention.so")]):
                    self.assertEqual(os.environ["LIBRARY_PATH"], "/build:/existing/lib")
                    self.assertEqual(os.environ["LD_RUN_PATH"], "$ORIGIN")
                    raise RuntimeError("compile failed")
            for name, value in original.items():
                self.assertEqual(os.environ[name], value)

    def test_link_environment_restores_absent_values(self):
        with patch.dict(os.environ):
            os.environ.pop("LIBRARY_PATH", None)
            os.environ.pop("LD_RUN_PATH", None)
            with _linking_environment([Path("/build/libattention.so")]):
                self.assertEqual(os.environ["LIBRARY_PATH"], "/build")
                self.assertEqual(os.environ["LD_RUN_PATH"], "$ORIGIN")
            self.assertNotIn("LIBRARY_PATH", os.environ)
            self.assertNotIn("LD_RUN_PATH", os.environ)


@unittest.skipUnless(
    sys.platform == "linux" and torch.cuda.is_available(), "CUDA on Linux is required"
)
class BundledAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_attention_ops(model="qwen3")
        cls.library = attention_library()
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.directory = Path(temporary.name)

        class Attention(torch.nn.Module):
            def __init__(self, scale=128**-0.5):
                super().__init__()
                self.scale = scale

            def forward(self, q, k, v, offsets, cuda_offsets, blocks):
                return torch.ops.xinfer.batched_flash_attention(
                    q, k, v, offsets, cuda_offsets, blocks, self.scale
                )

        class Append(Attention):
            def forward(self, q, k, v, new_k, new_v, positions, offsets, cuda_offsets, blocks):
                k.index_copy_(2, positions, new_k)
                v.index_copy_(2, positions, new_v)
                return super().forward(q, k, v, offsets, cuda_offsets, blocks), k, v

        for kind, tokens in (
            ("prefill", 16),
            ("decode", 1),
            ("append", 16),
            ("append_decode", 1),
            ("invalid_scale", 1),
        ):
            options = dict(device="cuda", dtype=torch.bfloat16)
            offsets, cuda_offsets, blocks = batched_attention_metadata(
                torch.tensor([32], dtype=torch.int32),
                torch.tensor([tokens], dtype=torch.int32),
                torch.device("cuda"),
            )
            inputs = (
                torch.randn(1, 16, tokens, 128, **options),
                torch.randn(1, 8, 1024, 128, **options),
                torch.randn(1, 8, 1024, 128, **options),
                offsets,
                cuda_offsets,
                blocks,
            )
            token_shape = {2: torch.export.Dim("tokens", min=2, max=512)} if tokens > 1 else {}
            dynamic = (token_shape, {}, {}, {}, {}, {})
            module = Attention(-1 if kind == "invalid_scale" else 128**-0.5)
            if kind.startswith("append"):
                q, k, v, offsets, cuda_offsets, blocks = inputs
                inputs = (
                    q,
                    k,
                    v,
                    k[:, :, :tokens].clone(),
                    v[:, :, :tokens].clone(),
                    torch.arange(32 - tokens, 32, device="cuda"),
                    offsets,
                    cuda_offsets,
                    blocks,
                )
                dynamic = (
                    token_shape,
                    {},
                    {},
                    token_shape,
                    token_shape,
                    {0: token_shape[2]} if tokens > 1 else {},
                    {},
                    {},
                    {},
                )
                module = Append()
            with torch.inference_mode():
                exported = torch.export.export(module, inputs, dynamic_shapes=dynamic)
                compile_package(
                    exported,
                    cls.directory / f"{kind}.pt2",
                    inductor_configs={},
                    native_libraries=[cls.library],
                    custom_op_shims=attention_c_shims(),
                )

    def test_packages_have_direct_shim_linkage_and_identical_payloads(self):
        for kind in ("prefill", "decode", "append", "append_decode", "invalid_scale"):
            with (
                self.subTest(kind=kind),
                zipfile.ZipFile(self.directory / f"{kind}.pt2") as archive,
            ):
                names = archive.namelist()
                libraries = [name for name in names if name.endswith("/" + self.library.name)]
                self.assertEqual(len(libraries), 1)
                self.assertEqual(archive.read(libraries[0]), self.library.read_bytes())
                sources = b"\n".join(
                    archive.read(name) for name in names if name.endswith(".wrapper.cpp")
                ).decode()
                call = "AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_batched_flash_attention("
                self.assertIn(call, sources)
                self.assertNotIn("aoti_torch_proxy_executor_call_function", sources)
                self.assertLess(sources.index("AOTICudaStreamGuard"), sources.index(call))
                wrappers = [name for name in names if name.endswith(".wrapper.so")]
                self.assertEqual(len(wrappers), 1)
                with tempfile.TemporaryDirectory() as temporary:
                    wrapper = Path(temporary) / "wrapper.so"
                    wrapper.write_bytes(archive.read(wrappers[0]))
                    dynamic = subprocess.check_output(["readelf", "-d", str(wrapper)], text=True)
                self.assertIn(f"Shared library: [{self.library.name}]", dynamic)
                self.assertRegex(dynamic, r"\((?:RUNPATH|RPATH)\).*\[\$ORIGIN\]")

    def test_fresh_processes_load_both_orders_and_reuse_one_library(self):
        env = dict(os.environ, XINFER_ATTENTION_OPS_LIBRARY="/nonexistent/attention.so")
        env.pop("PYTHONPATH", None)
        for order in (("prefill", "decode"), ("decode", "prefill")):
            with self.subTest(order=order):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        WORKER,
                        str(self.directory),
                        str(self.library),
                        ",".join(order),
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=1200,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                report = json.loads(result.stdout.splitlines()[-1])
                self.assertEqual(report["order"], list(order))
                self.assertEqual(len(report["checks"]), 18)
                self.assertEqual(report["mutation_checks"], 3)
                self.assertTrue(report["native_error_propagated"])


if __name__ == "__main__":
    unittest.main()
