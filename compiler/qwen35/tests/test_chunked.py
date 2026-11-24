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
            with (
                self.subTest(tile=tile),
                patch.object(chunked, "TRANSFORM_TILE", tile),
                patch.object(chunked.torch, "empty") as allocate,
            ):
                with self.assertRaisesRegex(ValueError, "TRANSFORM_TILE must"):
                    chunked.launch_chunked(q, q, q, other, other, other, other, other, other)
                allocate.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TransformTiles(unittest.TestCase):
    @torch.inference_mode()
    def test_all_columns_and_padded_rows_are_written(self):
        c, heads, dim = chunked.CHUNK_SIZE, 2, 128
        for tokens in (c, c + 1):
            chunks = (tokens + c - 1) // c
            key = torch.arange(tokens * heads * dim, device="cuda", dtype=torch.float32).reshape(
                1, tokens, heads, dim
            )
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
                        key,
                        value,
                        beta,
                        inverse,
                        prefix,
                        u,
                        w,
                        torch.tensor([tokens], device="cuda", dtype=torch.int32),
                        tokens,
                        heads,
                        dim,
                        c,
                        tile,
                        **chunked.OPTIONS,
                    )
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
                chunked.launch_chunked(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    state,
                    output,
                    final,
                    torch.tensor([tokens], device="cuda", dtype=torch.int32),
                )
            results.append((output, final))
        for expected, actual in zip(*results):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    @torch.inference_mode()
    def test_block_solve_matches_triangular_inverse_and_clamps_lengths(self):
        from compiler.qwen35.kernels.recurrent import fused

        torch.manual_seed(42)
        c, heads, dim = chunked.CHUNK_SIZE, 2, 128
        for length in (0, 1, 17, 64, 98):
            tokens = max(length, 64) if length <= 64 else 100
            chunks = (tokens + c - 1) // c
            k = torch.nn.functional.normalize(
                torch.randn(1, tokens, heads, dim, device="cuda"), dim=-1
            )
            g = -torch.rand(1, tokens, heads, device="cuda") * 0.2
            beta = torch.rand(1, tokens, heads, device="cuda") * 0.9
            inverse = torch.empty((1, chunks, heads, c, c), device="cuda", dtype=torch.float32)
            prefix = torch.empty((1, chunks, heads, c), device="cuda", dtype=torch.float32)
            chunked._solve[(chunks, heads)](
                k,
                g,
                beta,
                inverse,
                prefix,
                torch.tensor([length], device="cuda", dtype=torch.int32),
                tokens,
                heads,
                dim,
                c,
                **chunked.OPTIONS,
            )
            eye = torch.eye(c, device="cuda", dtype=torch.float32)
            lower = torch.tril(torch.ones((c, c), device="cuda", dtype=torch.bool), diagonal=-1)
            for chunk_idx in range(chunks):
                start = chunk_idx * c
                end = min(start + c, length)
                for head in range(heads):
                    valid = max(end - start, 0)
                    g_chunk = torch.zeros(c, device="cuda")
                    k_chunk = torch.zeros((c, dim), device="cuda")
                    b_chunk = torch.zeros(c, device="cuda")
                    if valid:
                        g_chunk[:valid] = g[0, start:end, head]
                        k_chunk[:valid] = k[0, start:end, head]
                        b_chunk[:valid] = beta[0, start:end, head]
                    p_ref = torch.cumsum(g_chunk, 0)
                    torch.testing.assert_close(
                        prefix[0, chunk_idx, head], p_ref, rtol=1e-5, atol=1e-6
                    )
                    decay = torch.exp(
                        torch.where(lower, p_ref[:, None] - p_ref[None, :], -float("inf"))
                    )
                    m = eye + ((k_chunk * b_chunk[:, None]) @ k_chunk.T) * decay
                    prod = inverse[0, chunk_idx, head] @ m
                    torch.testing.assert_close(prod, eye, rtol=1e-4, atol=1e-5)

        # Out-of-range query_lengths clamp safely to [0, tokens] in both branches.
        for tokens in (4, 100):
            batch, hk, hv = 2, 16, 16
            channels = (2 * hk + hv) * dim
            qkv = torch.randn(batch, tokens, channels, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(batch, tokens, hv, device="cuda", dtype=torch.bfloat16)
            a = torch.randn(batch, tokens, hv, device="cuda", dtype=torch.bfloat16)
            conv = torch.randn(batch, channels, 4, device="cuda", dtype=torch.bfloat16)
            state = torch.randn(batch, hv, dim, dim, device="cuda", dtype=torch.float32)
            weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
            a_log = torch.randn(hv, device="cuda", dtype=torch.float32) * 0.1
            dt_bias = torch.randn(hv, device="cuda", dtype=torch.bfloat16) * 0.1
            clamped_out = fused(
                qkv,
                b,
                a,
                conv,
                state,
                weight,
                a_log,
                dt_bias,
                torch.tensor([0, tokens], device="cuda", dtype=torch.int32),
            )
            oob_out = fused(
                qkv,
                b,
                a,
                conv,
                state,
                weight,
                a_log,
                dt_bias,
                torch.tensor([-3, tokens + 7], device="cuda", dtype=torch.int32),
            )
            for actual, expected in zip(oob_out, clamped_out):
                self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
