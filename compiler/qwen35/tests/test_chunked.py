"""Transform-tile launch geometry and partial-chunk coverage regressions."""

import unittest
from unittest.mock import patch

import torch

from compiler.qwen35.kernels import chunked


class TransformTileContract(unittest.TestCase):
    def test_invalid_tiles_fail_before_workspace_allocation(self):
        q = torch.empty(1, 2, 2, 128)
        other = torch.empty(0)
        for tile in (0, -16, 8, 24, 48, 256, True, 32.0):
            with self.subTest(tile=tile), patch.object(chunked, "TRANSFORM_TILE", tile), \
                    patch.object(chunked.torch, "empty") as allocate:
                with self.assertRaisesRegex(ValueError, "TRANSFORM_TILE must"):
                    chunked.launch_chunked(q, q, q, other, other, other, other, other)
                allocate.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TransformTiles(unittest.TestCase):
    @torch.inference_mode()
    def test_all_columns_and_padded_rows_are_written(self):
        c, heads, dim = chunked.CHUNK_SIZE, 2, 128
        for tokens in (c, c + 1):
            chunks = (tokens + c - 1) // c
            key = torch.arange(tokens * heads * dim, device="cuda", dtype=torch.float32).reshape(1, tokens, heads, dim)
            value = key + 17
            beta = torch.ones(1, tokens, heads, device="cuda")
            inverse = torch.eye(c, device="cuda").expand(chunks, heads, c, c).contiguous()
            prefix = torch.zeros(chunks, heads, c, device="cuda")
            expected = []
            for data in (value, key):
                padded = torch.zeros(chunks * c, heads, dim, device="cuda")
                padded[:tokens].copy_(data[0])
                expected.append(padded.reshape(chunks, c, heads, dim).transpose(1, 2))
            for tile in (16, 32, 64, 128):
                with self.subTest(tokens=tokens, tile=tile):
                    shape = (chunks, heads, c, dim)
                    u, w = (torch.full(shape, float("nan"), device="cuda") for _ in range(2))
                    chunked._transforms[(chunks, heads, dim // tile)](
                        key, value, beta, inverse, prefix, u, w, tokens,
                        heads, dim, c, tile, **chunked.OPTIONS)
                    torch.testing.assert_close(u, expected[0], rtol=0, atol=0)
                    torch.testing.assert_close(w, expected[1], rtol=0, atol=0)

    @torch.inference_mode()
    def test_launcher_forwards_nondefault_tile(self):
        torch.manual_seed(35)
        tokens, heads, dim = chunked.CHUNK_SIZE + 1, 2, 128
        q, k, v = (torch.randn(1, tokens, heads, dim, device="cuda") for _ in range(3))
        q = torch.nn.functional.normalize(q, dim=-1) * dim**-0.5
        k = torch.nn.functional.normalize(k, dim=-1)
        g = -torch.rand(1, tokens, heads, device="cuda") * 0.01
        beta = torch.rand_like(g)
        state = torch.randn(1, heads, dim, dim, device="cuda") * 0.1
        results = []
        for tile in (32, 64):
            output, final = torch.empty_like(v), torch.empty_like(state)
            with patch.object(chunked, "TRANSFORM_TILE", tile):
                chunked.launch_chunked(q, k, v, g, beta, state, output, final)
            results.append((output, final))
        for expected, actual in zip(*results):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
