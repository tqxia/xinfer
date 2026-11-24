"""CUDA regression checks for shared grouped-query prefix Flash attention."""

import unittest

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from compiler.common.attention import (
    cache_extent,
    load_attention_ops,
    batched_attention_metadata,
)
from compiler.common.kernels.kv_cache import append_kv_batched
from compiler.common.tests.tensors import row_errors


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class PrefixFlash(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_attention_ops(model="qwen3")
        cls.op = torch.ops.xinfer.flash_attention.default

    def inputs(self, heads, kv_heads, dim, tokens, feature_stride=1, capacity=1024, batch_size=1):
        options = dict(device="cuda", dtype=torch.bfloat16)
        q = torch.randn(batch_size, heads, tokens, dim * feature_stride, **options)[
            ..., ::feature_stride
        ]
        k = torch.randn(batch_size, kv_heads, capacity, dim * feature_stride, **options)[
            ..., ::feature_stride
        ]
        v = torch.randn(batch_size, kv_heads, capacity, dim * feature_stride, **options)[
            ..., ::feature_stride
        ]
        return q, k, v

    @torch.inference_mode()
    def test_batched_attention_and_masked_writes(self):
        torch.manual_seed(18)
        for batch in range(1, 9):
            for width, start, capacity, ragged in (
                (1, 17, 1024, False),
                (1, 17, 1024, True),
                (2, 511, 513, True),
                (17, 7, 1024, True),
                (512, 1, 1024, True),
                (512, 2048, 4096, True),
            ):
                counts = [
                    width if not ragged or r % 3 == 0 else (0 if r % 3 == 1 else 1)
                    for r in range(batch)
                ]
                starts = [start - r if start >= batch else start for r in range(batch)]
                lengths = [s + c for s, c in zip(starts, counts)]
                with self.subTest(batch=batch, counts=counts, capacity=capacity):
                    q, k, v = self.inputs(16, 8, 128, width, capacity=capacity, batch_size=batch)
                    key = torch.randn(batch, width, 8, 128, device="cuda", dtype=q.dtype).transpose(
                        1, 2
                    )
                    value = torch.randn_like(key)
                    positions = torch.stack(
                        [torch.arange(s, s + width, device="cuda") for s in starts]
                    )
                    host, cu, blocks = batched_attention_metadata(
                        torch.tensor(lengths, dtype=torch.int32),
                        torch.tensor(counts, dtype=torch.int32),
                        q.device,
                    )
                    # Inductor may use zero stride for the singleton page dimension.
                    if not ragged:
                        blocks = blocks.as_strided((batch, 1), (1, 0))
                    expected_k, expected_v = k.clone(), v.clone()
                    for row, (count, offset) in enumerate(zip(counts, starts)):
                        expected_k[row, :, offset : offset + count] = key[row, :, :count]
                        expected_v[row, :, offset : offset + count] = value[row, :, :count]
                        # Even invalid padded positions must never be dereferenced.
                        positions[row, count:] = capacity + 100
                    before_inputs = [t.clone() for t in (key, value, positions)]
                    append_kv_batched(key, value, positions, cu[0], k, v)
                    self.assertTrue(torch.equal(k, expected_k) and torch.equal(v, expected_v))
                    self.assertTrue(
                        all(
                            torch.equal(a, b)
                            for a, b in zip((key, value, positions), before_inputs)
                        )
                    )
                    op = torch.ops.xinfer.batched_flash_attention.default
                    actual = op(q, k, v, host, cu, blocks, 128**-0.5)
                    self.assertEqual(actual.stride(), torch.empty(q.shape).stride())
                    self.assertNotIn(actual.data_ptr(), [t.data_ptr() for t in (q, k, v)])
                    for row, (count, length) in enumerate(zip(counts, lengths)):
                        self.assertEqual(torch.count_nonzero(actual[row, :, count:]).item(), 0)
                        if count:
                            mask = (
                                torch.arange(length, device="cuda")[None, :]
                                <= torch.arange(length - count, length, device="cuda")[:, None]
                            )
                            with sdpa_kernel(SDPBackend.MATH):
                                expected = F.scaled_dot_product_attention(
                                    q[row : row + 1, :, :count].float(),
                                    k[row : row + 1, :, :length].float(),
                                    v[row : row + 1, :, :length].float(),
                                    attn_mask=mask,
                                    enable_gqa=True,
                                    scale=128**-0.5,
                                )
                            error = row_errors(
                                actual[row : row + 1, :, :count].float().flatten(1),
                                expected.flatten(1),
                            )
                            self.assertLess(error["relative_l2"], 0.01)
                            self.assertLessEqual(error["peak_relative"], 0.05)
                        q[row, :, count:] = float("nan")
                        k[row, :, length:] = float("nan")
                        v[row, :, length:] = float("nan")
                    self.assertTrue(torch.equal(actual, op(q, k, v, host, cu, blocks, 128**-0.5)))
                    if batch > 1:
                        k[0].fill_(5)
                        v[0].fill_(7)
                        self.assertTrue(
                            torch.equal(actual[1:], op(q, k, v, host, cu, blocks, 128**-0.5)[1:])
                        )
                    torch.library.opcheck(
                        op,
                        (q, k, v, host, cu, blocks, 128**-0.5),
                        test_utils=("test_schema", "test_faketensor"),
                    )

    @torch.inference_mode()
    def test_appended_prefill_batches(self):
        # Shapes from the Qwen3 prefill backend divergence; retain full-capacity
        # strides and use an independent math reference for the attention op.
        torch.manual_seed(18)
        for batch_size in (1, 2):
            for length in (1024, 2560):
                with self.subTest(batch_size=batch_size, length=length):
                    q, k, v = self.inputs(16, 8, 128, 512, capacity=4096, batch_size=batch_size)
                    extent = cache_extent(length)
                    actual = self.op(q, k, v, extent, 128**-0.5)
                    positions = torch.arange(length - 512, length, device="cuda")
                    mask = torch.arange(length, device="cuda")[None, :] <= positions[:, None]
                    with sdpa_kernel(SDPBackend.MATH):
                        reference = F.scaled_dot_product_attention(
                            q.float(),
                            k[:, :, :length].float(),
                            v[:, :, :length].float(),
                            attn_mask=mask,
                            enable_gqa=True,
                            scale=128**-0.5,
                        )
                    error = row_errors(actual.float().flatten(1), reference.flatten(1))
                    self.assertLess(error["relative_l2"], 0.01)
                    self.assertLessEqual(error["peak_relative"], 0.05)
                    # Future cache slots and another request's KV must not
                    # affect the valid output for this row.
                    k[:, :, length:].fill_(100)
                    v[:, :, length:].fill_(-100)
                    self.assertTrue(torch.equal(actual, self.op(q, k, v, extent, 128**-0.5)))
                    if batch_size == 2:
                        k[0].fill_(5)
                        v[0].fill_(7)
                        self.assertTrue(
                            torch.equal(actual[1], self.op(q, k, v, extent, 128**-0.5)[1])
                        )

    @torch.inference_mode()
    def test_two_independent_rows(self):
        torch.manual_seed(13)
        for tokens, length in ((1, 32), (2, 3), (16, 32), (512, 512)):
            triples = [self.inputs(16, 8, 128, tokens) for _ in range(2)]
            q, k, v = [torch.cat([triple[i] for triple in triples]) for i in range(3)]
            extent = cache_extent(length)
            out = self.op(q, k, v, extent, 128**-0.5)
            expected = torch.cat([self.op(*triple, extent, 128**-0.5) for triple in triples])
            # Batch size can change Flash's reduction schedule. Use the same
            # numerical bounds as the FP32 reference checks below.
            error = row_errors(out.float().flatten(1), expected.float().flatten(1))
            self.assertLess(error["relative_l2"], 0.01)
            self.assertLessEqual(error["peak_relative"], 0.05)
            positions = torch.arange(length - tokens, length, device="cuda")
            mask = torch.arange(length, device="cuda")[None, :] <= positions[:, None]
            with sdpa_kernel(SDPBackend.MATH):
                reference = F.scaled_dot_product_attention(
                    q.float(),
                    k[:, :, :length].float(),
                    v[:, :, :length].float(),
                    attn_mask=mask,
                    enable_gqa=True,
                    scale=128**-0.5,
                )
            error = row_errors(out.float().flatten(1), reference.flatten(1))
            self.assertLess(error["relative_l2"], 0.01)
            self.assertLessEqual(error["peak_relative"], 0.05)
            native = torch.ops.aten._scaled_dot_product_flash_attention.default(
                q, k[:, :, :length], v[:, :, :length], 0.0, True, scale=128**-0.5
            )[0]
            self.assertTrue(torch.equal(out, native))
            self.assertEqual(out.shape, q.shape)
            k[0].fill_(5)
            v[0].fill_(7)
            self.assertTrue(torch.equal(out[1], self.op(q, k, v, extent, 128**-0.5)[1]))
            torch.library.opcheck(self.op, (q, k, v, extent, 128**-0.5))

    @torch.inference_mode()
    def test_grouped_prefix_and_read_only_contract(self):
        torch.manual_seed(35)
        for heads, kv_heads, dim in ((16, 8, 128), (8, 2, 256)):
            for tokens, length in (
                (1, 1),
                (1, 2),
                (1, 32),
                (1, 129),
                (1, 1024),
                (2, 2),
                (2, 3),
                (16, 32),
                (16, 129),
                (510, 1022),
                (512, 512),
                (512, 1024),
            ):
                with self.subTest(heads=heads, tokens=tokens, length=length):
                    q, k, v = self.inputs(heads, kv_heads, dim, tokens)
                    before = [t.clone() for t in (q, k, v)]
                    extent = cache_extent(length)
                    actual = self.op(q, k, v, extent, dim**-0.5)
                    native = torch.ops.aten._scaled_dot_product_flash_attention.default(
                        q, k[:, :, :length], v[:, :, :length], 0.0, True, scale=dim**-0.5
                    )[0]
                    self.assertTrue(torch.equal(actual, native))
                    self.assertEqual(actual.stride(), (heads * tokens * dim, tokens * dim, dim, 1))
                    self.assertNotIn(actual.data_ptr(), [t.data_ptr() for t in (q, k, v)])
                    for tensor, original in zip((q, k, v), before):
                        self.assertTrue(torch.equal(tensor, original))
                    positions = torch.arange(length - tokens, length, device="cuda")
                    mask = torch.arange(length, device="cuda")[None, :] <= positions[:, None]
                    with sdpa_kernel(SDPBackend.MATH):
                        expected = F.scaled_dot_product_attention(
                            q.float(),
                            k[:, :, :length].float(),
                            v[:, :, :length].float(),
                            attn_mask=mask,
                            enable_gqa=True,
                            scale=dim**-0.5,
                        )
                    error = row_errors(actual.float().flatten(1), expected.flatten(1))
                    self.assertLess(error["relative_l2"], 0.01)
                    self.assertLessEqual(error["peak_relative"], 0.05)
                    k[:, :, length:].fill_(100)
                    v[:, :, length:].fill_(-100)
                    extent.fill_(255)
                    self.assertTrue(torch.equal(actual, self.op(q, k, v, extent, dim**-0.5)))
                    checks = torch.library.opcheck(self.op, (q, k, v, extent, dim**-0.5))
                    self.assertTrue(all(value == "SUCCESS" for value in checks.values()))

    @torch.inference_mode()
    def test_long_prefix_and_tensor_derived_capacity(self):
        # Keep Q short: this checks appended causality at 8K/16K without an
        # N-by-N reference allocation. Existing cases cover 1K bundles.
        for heads, kv_heads, dim in ((16, 8, 128), (16, 4, 256)):
            for length in (8192, 8193, 16384):
                with self.subTest(heads=heads, length=length):
                    q, k, v = self.inputs(heads, kv_heads, dim, 2, capacity=16384)
                    extent = cache_extent(length)
                    actual = self.op(q, k, v, extent, dim**-0.5)
                    positions = torch.arange(length - q.shape[2], length, device="cuda")
                    mask = torch.arange(length, device="cuda")[None, :] <= positions[:, None]
                    with sdpa_kernel(SDPBackend.MATH):
                        expected = F.scaled_dot_product_attention(
                            q.float(),
                            k[:, :, :length].float(),
                            v[:, :, :length].float(),
                            attn_mask=mask,
                            enable_gqa=True,
                            scale=dim**-0.5,
                        )
                    error = row_errors(actual.float().flatten(1), expected.flatten(1))
                    self.assertLess(error["relative_l2"], 0.01)
                    self.assertLessEqual(error["peak_relative"], 0.05)
                    k[:, :, length:].fill_(100)
                    v[:, :, length:].fill_(-100)
                    self.assertTrue(torch.equal(actual, self.op(q, k, v, extent, dim**-0.5)))
                    torch.library.opcheck(
                        self.op,
                        (q, k, v, extent, dim**-0.5),
                        test_utils=("test_schema", "test_faketensor"),
                    )
                    with self.assertRaisesRegex(RuntimeError, "Invalid valid-prefix length"):
                        self.op(q, k, v, cache_extent(16385), dim**-0.5)

    @torch.inference_mode()
    def test_appended_chunks_use_lower_right_causality(self):
        for heads, kv_heads, dim in ((16, 8, 128), (8, 2, 256)):
            with self.subTest(heads=heads):
                q, k, v = self.inputs(heads, kv_heads, dim, 2)
                q.zero_()
                k.zero_()
                v.copy_(
                    torch.arange(1024, device="cuda", dtype=torch.bfloat16)[None, None, :, None]
                )
                output = self.op(q, k, v, cache_extent(5), dim**-0.5)
                expected = torch.tensor([1.5, 2.0], device="cuda", dtype=torch.bfloat16)
                self.assertTrue(
                    torch.equal(output, expected[None, None, :, None].expand_as(output))
                )

    @torch.inference_mode()
    def test_feature_strides_and_fake_contract(self):
        for heads, kv_heads, dim in ((16, 8, 128), (8, 2, 256)):
            with self.subTest(heads=heads):
                q, k, v = self.inputs(heads, kv_heads, dim, 2, feature_stride=2)
                extent = cache_extent(3)
                output = self.op(q, k, v, extent, dim**-0.5)
                expected = torch.ops.aten._scaled_dot_product_flash_attention.default(
                    q.contiguous(),
                    k[:, :, :3].contiguous(),
                    v[:, :, :3].contiguous(),
                    0.0,
                    True,
                    scale=dim**-0.5,
                )[0]
                self.assertTrue(torch.equal(output, expected))
                checks = torch.library.opcheck(self.op, (q, k, v, extent, dim**-0.5))
                self.assertTrue(all(value == "SUCCESS" for value in checks.values()))

    @torch.inference_mode()
    def test_invalid_extent_and_head_dimensions(self):
        for heads, kv_heads, dim in ((16, 8, 128), (8, 2, 256)):
            q, k, v = self.inputs(heads, kv_heads, dim, 2)
            for length in (0, 1, 1025):
                with (
                    self.subTest(heads=heads, length=length),
                    self.assertRaisesRegex(RuntimeError, "Invalid valid-prefix length"),
                ):
                    self.op(q, k, v, cache_extent(length), dim**-0.5)
        q, k, v = self.inputs(8, 2, 256, 2)
        # A positive GQA multiple is valid even outside the original profiles.
        self.assertEqual(self.op(q[:, :4], k, v, cache_extent(3), 256**-0.5).shape, q[:, :4].shape)
        with self.assertRaisesRegex(RuntimeError, "Query heads must be a positive multiple"):
            self.op(q[:, :3], k, v, cache_extent(3), 256**-0.5)
        for dim in (7, 127, 264):
            with self.subTest(dim=dim):
                invalid = self.inputs(8, 2, dim, 2)
                with self.assertRaisesRegex(RuntimeError, "Flash head dimension"):
                    self.op(*invalid, cache_extent(3), dim**-0.5)

    def test_batched_validation_guards_and_nonunit_strides(self):
        op = torch.ops.xinfer.batched_flash_attention.default
        with torch.inference_mode():
            for batch, width, capacity in ((1, 1, 513), (2, 1, 1024), (2, 4, 1024)):
                with self.subTest(batch=batch, width=width, capacity=capacity):
                    q, k, v = self.inputs(
                        16, 8, 128, width, feature_stride=2, capacity=capacity, batch_size=batch
                    )
                    lengths = torch.tensor([width + 4 + r for r in range(batch)], dtype=torch.int32)
                    counts = torch.full((batch,), width, dtype=torch.int32)
                    host, cu, blocks = batched_attention_metadata(lengths, counts, q.device)
                    actual = op(q, k, v, host, cu, blocks, 128**-0.5)
                    reference = op(
                        q.contiguous(), k.contiguous(), v.contiguous(), host, cu, blocks, 128**-0.5
                    )
                    self.assertTrue(torch.equal(actual, reference))
                    self.assertEqual(actual.stride(), torch.empty(q.shape).stride())

            q, k, v = self.inputs(8, 2, 128, 2, capacity=1024, batch_size=2)
            lengths = torch.tensor([4, 5], dtype=torch.int32)
            counts = torch.tensor([2, 2], dtype=torch.int32)
            host, cu, blocks = batched_attention_metadata(lengths, counts, q.device)
            with self.assertRaisesRegex(RuntimeError, "Query heads must be a positive multiple"):
                op(q[:, :3], k, v, host, cu, blocks, 128**-0.5)
            for bad_dim in (7, 127, 264):
                bad_q, bad_k, bad_v = self.inputs(8, 2, bad_dim, 2, capacity=1024, batch_size=2)
                with self.assertRaisesRegex(RuntimeError, "Flash head dimension"):
                    op(bad_q, bad_k, bad_v, host, cu, blocks, 128**-0.5)
            for bad_scale in (0.0, -1.0, float("nan"), float("inf")):
                with self.assertRaisesRegex(RuntimeError, "Attention scale must be finite"):
                    op(q, k, v, host, cu, blocks, bad_scale)
            wide_q, _, _ = self.inputs(8, 2, 128, 513, capacity=1024, batch_size=2)
            with self.assertRaisesRegex(RuntimeError, "Unsupported Q shape"):
                op(wide_q, k, v, host, cu, blocks, 128**-0.5)

        grad_q = q.clone().requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "inference only"):
            op(grad_q, k, v, host, cu, blocks, 128**-0.5)


if __name__ == "__main__":
    unittest.main()
