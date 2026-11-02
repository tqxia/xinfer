"""CUDA regression checks for shared grouped-query prefix Flash attention."""

import unittest

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from compiler.common.attention import cache_extent, load_attention_ops


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class PrefixFlash(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_attention_ops()
        cls.op = torch.ops.xinfer.flash_attention.default

    def inputs(self, heads, kv_heads, dim, tokens, feature_stride=1):
        options = dict(device="cuda", dtype=torch.bfloat16)
        q = torch.randn(1, heads, tokens, dim * feature_stride, **options)[..., ::feature_stride]
        k = torch.randn(1, kv_heads, 1024, dim * feature_stride, **options)[..., ::feature_stride]
        v = torch.randn(1, kv_heads, 1024, dim * feature_stride, **options)[..., ::feature_stride]
        return q, k, v

    @torch.inference_mode()
    def test_grouped_prefix_and_read_only_contract(self):
        torch.manual_seed(35)
        for heads, kv_heads, dim in ((16, 8, 128), (8, 2, 256)):
            for tokens, length in ((1, 1), (1, 129), (2, 3), (16, 129), (512, 1024)):
                with self.subTest(heads=heads, tokens=tokens, length=length):
                    q, k, v = self.inputs(heads, kv_heads, dim, tokens)
                    before = [t.clone() for t in (q, k, v)]
                    extent = cache_extent(length)
                    actual = self.op(q, k, v, extent, dim**-0.5)
                    native = torch.ops.aten._scaled_dot_product_flash_attention.default(
                        q, k[:, :, :length], v[:, :, :length], 0., True, scale=dim**-0.5)[0]
                    self.assertTrue(torch.equal(actual, native))
                    self.assertEqual(actual.stride(), (heads * tokens * dim, tokens * dim, dim, 1))
                    self.assertNotIn(actual.data_ptr(), [t.data_ptr() for t in (q, k, v)])
                    for tensor, original in zip((q, k, v), before):
                        self.assertTrue(torch.equal(tensor, original))
                    positions = torch.arange(length - tokens, length, device="cuda")
                    mask = torch.arange(length, device="cuda")[None, :] <= positions[:, None]
                    with sdpa_kernel(SDPBackend.MATH):
                        expected = F.scaled_dot_product_attention(
                            q.float(), k[:, :, :length].float(), v[:, :, :length].float(),
                            attn_mask=mask, enable_gqa=True, scale=dim**-0.5)
                    error = (actual.float() - expected).norm() / expected.norm()
                    self.assertLess(error.item(), 0.01)
                    k[:, :, length:].fill_(100)
                    v[:, :, length:].fill_(-100)
                    extent.fill_(255)
                    self.assertTrue(torch.equal(actual, self.op(q, k, v, extent, dim**-0.5)))

    @torch.inference_mode()
    def test_appended_chunks_use_lower_right_causality(self):
        for heads, kv_heads, dim in ((16, 8, 128), (8, 2, 256)):
            with self.subTest(heads=heads):
                q, k, v = self.inputs(heads, kv_heads, dim, 2)
                q.zero_()
                k.zero_()
                v.copy_(torch.arange(1024, device="cuda", dtype=torch.bfloat16)[None, None, :, None])
                output = self.op(q, k, v, cache_extent(5), dim**-0.5)
                expected = torch.tensor([1.5, 2.0], device="cuda", dtype=torch.bfloat16)
                self.assertTrue(torch.equal(output, expected[None, None, :, None].expand_as(output)))

    @torch.inference_mode()
    def test_feature_strides_and_fake_contract(self):
        for heads, kv_heads, dim in ((16, 8, 128), (8, 2, 256)):
            with self.subTest(heads=heads):
                q, k, v = self.inputs(heads, kv_heads, dim, 2, feature_stride=2)
                extent = cache_extent(3)
                output = self.op(q, k, v, extent, dim**-0.5)
                expected = torch.ops.aten._scaled_dot_product_flash_attention.default(
                    q.contiguous(), k[:, :, :3].contiguous(), v[:, :, :3].contiguous(),
                    0., True, scale=dim**-0.5)[0]
                self.assertTrue(torch.equal(output, expected))
                checks = torch.library.opcheck(self.op, (q, k, v, extent, dim**-0.5))
                self.assertTrue(all(value == "SUCCESS" for value in checks.values()))

    @torch.inference_mode()
    def test_invalid_extent_and_head_dimensions(self):
        q, k, v = self.inputs(8, 2, 256, 2)
        for length in (0, 1, 1025):
            with self.subTest(length=length), self.assertRaisesRegex(RuntimeError, "Invalid valid-prefix length"):
                self.op(q, k, v, cache_extent(length), 256**-0.5)
        with self.assertRaisesRegex(RuntimeError, "Unsupported query/KV head dimensions"):
            self.op(q[:, :4], k, v, cache_extent(3), 256**-0.5)


if __name__ == "__main__":
    unittest.main()
