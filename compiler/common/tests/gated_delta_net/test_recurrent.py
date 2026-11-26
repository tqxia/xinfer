"""Native DeltaNet kernel behavior, numerical controls and strict export contracts."""

import unittest

import torch

from compiler.common.tests.tensors import bits_equal, cpu_copy
from compiler.common.kernels.gated_delta_net.deltanet import (
    batched_deltanet,
    cumulative_query_lengths,
    load_deltanet_ops,
)
from compiler.common.tests.gated_delta_net.fixtures import (
    BOUNDARY_LENGTH,
    CHUNK_SIZE,
    PARTITIONS,
    Fused,
    compare,
    dispatch_cases,
    inputs_for,
    make_cases,
    reference,
    run_cases,
)


def inputs(value_heads, device="meta"):
    channels = (32 + value_heads) * 128
    return (
        torch.empty(1, 1, channels, device=device),
        torch.empty(1, 1, value_heads, device=device),
        torch.empty(1, 1, value_heads, device=device),
        torch.empty(1, channels, 4, device=device),
        torch.empty(1, value_heads, 128, 128, device=device),
        torch.empty(channels, 4, device=device),
        torch.empty(value_heads, device=device),
        torch.empty(value_heads, device=device),
    )


class RecurrentContract(unittest.TestCase):
    def test_kernel_rejects_inconsistent_shapes_before_cuda(self):
        for value_heads in (16, 32, 48):
            sample = inputs(value_heads)
            invalid = {
                "key_heads": (0, sample[0][..., : (16 + value_heads) * 128]),
                "channels": (0, sample[0][..., :-1]),
                "gates": (1, sample[1][..., :-1]),
                "conv": (3, sample[3][..., :3]),
                "state": (4, sample[4][:, :-1]),
                "decay": (6, sample[6][:-1]),
            }
            for name, (index, value) in invalid.items():
                broken = list(sample)
                broken[index] = value
                with self.subTest(value_heads=value_heads, field=name):
                    with self.assertRaisesRegex(ValueError, "Expected"):
                        batched_deltanet(*broken)
            with self.subTest(value_heads=value_heads, field="device"):
                with self.assertRaisesRegex(ValueError, "Expected CUDA"):
                    batched_deltanet(*sample)
        with self.assertRaisesRegex(ValueError, "Hk=16"):
            batched_deltanet(*inputs(24))

    def test_kernel_validates_decay_parameter_dtypes_before_cuda(self):
        fp32 = inputs(16)
        for index, name in ((6, "A_log"), (7, "dt_bias")):
            broken = list(fp32)
            broken[index] = broken[index].bfloat16()
            with self.subTest(qkv="fp32", field=name):
                with self.assertRaisesRegex(ValueError, "Expected matching BF16 or FP32"):
                    batched_deltanet(*broken)
        bf16 = tuple(tensor if idx == 4 else tensor.bfloat16() for idx, tensor in enumerate(fp32))
        for index, name in ((6, "A_log"), (7, "dt_bias")):
            broken = list(bf16)
            broken[index] = broken[index].half()
            with self.subTest(qkv="bf16_half", field=name):
                with self.assertRaisesRegex(ValueError, "Expected matching BF16 or FP32"):
                    batched_deltanet(*broken)
            mixed = list(bf16)
            mixed[index] = mixed[index].float()
            with self.subTest(qkv="bf16_fp32_decay", field=name):
                with self.assertRaisesRegex(ValueError, "Expected CUDA"):
                    batched_deltanet(*mixed)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class RecurrentKernels:
    value_heads = 16

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        load_deltanet_ops()
        torch.manual_seed(35)
        cls.cases = dispatch_cases(make_cases(cls.value_heads))
        with torch.inference_mode():
            cls.results = {"native": run_cases(batched_deltanet, cls.cases, inplace=True)}

    @classmethod
    def tearDownClass(cls):
        del cls.results, cls.cases

    @classmethod
    def engine(cls, name):
        if name not in cls.results:
            call = {
                "hf": reference,
                "hf_chunk64": lambda *x: reference(*x, chunked=True),
            }[name]
            with torch.inference_mode():
                cls.results[name] = run_cases(call, cls.cases, inplace=False)
        return cls.results[name]

    def compare(self, label, actual, expected, exact_tensors=()):
        report = dict(comparisons=[], failures=[])
        compare(report, label, actual, expected, exact_tensors)
        self.assertFalse(
            bool(report["failures"]),
            f"{len(report['failures'])} failures: {report['failures'][:8]}",
        )

    def sample(self, dtype, tokens):
        name = f"{dtype}-t{tokens}-state"
        return inputs_for(next(case for case in self.cases if case["name"] == name))

    def test_hf_recurrent_and_chunked_references(self):
        self.assertEqual(CHUNK_SIZE, 64, "The HF chunk-64 control must stay independent")
        for engine in ("hf", "hf_chunk64"):
            with self.subTest(reference=engine):
                self.compare(f"native_vs_{engine}", self.engine("native"), self.engine(engine))

    def test_partition_invariance(self):
        for engine in ("native", "hf", "hf_chunk64"):
            values = self.engine(engine)
            for dtype in ("bf16", "fp32"):
                with self.subTest(engine=engine, dtype=dtype):
                    parts = [values[f"{dtype}-split", i] for i in range(len(PARTITIONS))]
                    key = (f"{dtype}-partition", 0)
                    self.compare(
                        f"{engine}_partition",
                        {key: (torch.cat([part[0] for part in parts], dim=1), *parts[-1][1:])},
                        {key: values[f"{dtype}-whole", 0]},
                        (1,),
                    )

    def test_reset_repeat_bitwise(self):
        for engine in ("native", "hf", "hf_chunk64"):
            values = self.engine(engine)
            for dtype in ("bf16", "fp32"):
                for original, repeat in (
                    ("carry", "repeat"),
                    ("split", "split-repeat"),
                    ("cond-alternate", "cond-alternate-repeat"),
                ):
                    case = next(c for c in self.cases if c["name"] == f"{dtype}-{original}")
                    for step in range(len(case["steps"])):
                        with self.subTest(engine=engine, case=case["name"], step=step):
                            self.assertTrue(
                                all(
                                    bits_equal(a, b)
                                    for a, b in zip(
                                        values[f"{dtype}-{original}", step],
                                        values[f"{dtype}-{repeat}", step],
                                    )
                                )
                            )

    @torch.inference_mode()
    def test_nondefault_stream_and_noncontiguous_inputs(self):
        for dtype in ("bf16", "fp32"):
            for tokens in (BOUNDARY_LENGTH - 1, BOUNDARY_LENGTH + 1):
                with self.subTest(dtype=dtype, tokens=tokens):
                    sample = self.sample(dtype, tokens)
                    ref_inputs = tuple(x.clone() for x in sample)
                    ref_core = batched_deltanet(*ref_inputs)
                    expected = cpu_copy((ref_core, ref_inputs[3], ref_inputs[4]))

                    stream_inputs = tuple(x.clone() for x in sample)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        actual_core = batched_deltanet(*stream_inputs)
                    stream.synchronize()
                    self.assertTrue(
                        all(
                            bits_equal(a, b)
                            for a, b in zip(
                                cpu_copy((actual_core, stream_inputs[3], stream_inputs[4])),
                                expected,
                            )
                        )
                    )

                    strided = [x.clone() for x in sample]
                    backing = sample[0].new_empty((*sample[0].shape[:-1], sample[0].shape[-1] * 2))
                    backing[..., ::2].copy_(sample[0])
                    strided[0], strided[5] = backing[..., ::2], sample[5].t().contiguous().t()
                    self.assertFalse(strided[0].is_contiguous())
                    self.assertFalse(strided[5].is_contiguous())
                    strided_core = batched_deltanet(*strided)
                    self.assertTrue(
                        all(
                            bits_equal(a, b)
                            for a, b in zip(
                                cpu_copy((strided_core, strided[3], strided[4])), expected
                            )
                        )
                    )

    @torch.inference_mode()
    def test_rejected_inputs(self):
        for dtype in ("bf16", "fp32"):
            sample = self.sample(dtype, 32)
            invalid = dict(
                empty=[*(x[:, :0] for x in sample[:3]), *sample[3:]],
                too_long=[*(x.repeat(1, 17, 1)[:, :513] for x in sample[:3]), *sample[3:]],
                gates=[sample[0], sample[1][..., :1], *sample[2:]],
                conv_width=[*sample[:3], sample[3][..., :3], *sample[4:]],
                state_dtype=[*sample[:4], sample[4].bfloat16(), *sample[5:]],
                weight_dtype=[*sample[:5], sample[5].half(), *sample[6:]],
                noncontiguous_conv=[
                    *sample[:3],
                    sample[3].transpose(1, 2).transpose(1, 2),
                    *sample[4:],
                ],
                noncontiguous_state=[
                    *sample[:4],
                    sample[4].transpose(2, 3).contiguous().transpose(2, 3),
                    *sample[5:],
                ],
                cpu=[x.cpu() for x in sample],
            )
            # Ensure noncontiguous_conv actually has non-contiguous strides.
            conv_buf = sample[3].new_empty((1, sample[3].shape[1], sample[3].shape[2] * 2))
            invalid["noncontiguous_conv"] = [*sample[:3], conv_buf[..., ::2], *sample[4:]]
            self.assertFalse(invalid["noncontiguous_conv"][3].is_contiguous())
            self.assertFalse(invalid["noncontiguous_state"][4].is_contiguous())
            for name, inputs in invalid.items():
                with self.subTest(dtype=dtype, invalid=name):
                    with self.assertRaises((ValueError, RuntimeError)):
                        batched_deltanet(*inputs)

    @torch.inference_mode()
    def test_operator_schema_and_fake_tensors(self):
        for dtype in ("bf16", "fp32"):
            sample = self.sample(dtype, 32)
            cu_query, cuda_cu_query = cumulative_query_lengths(None, 1, 32, sample[0].device)
            with self.subTest(dtype=dtype):
                checks = torch.library.opcheck(
                    torch.ops.xinfer.batched_deltanet.default,
                    (*sample, cu_query, cuda_cu_query),
                    test_utils=("test_schema", "test_faketensor"),
                )
                self.assertTrue(all(value == "SUCCESS" for value in checks.values()))

    def test_strict_export_uses_single_native_op_without_cond(self):
        for dtype in ("bf16", "fp32"):
            for mode, tokens in (("decode", 1), ("prefill", 32)):
                with self.subTest(dtype=dtype, mode=mode):
                    dynamic = None
                    if mode == "prefill":
                        length = torch.export.Dim("tokens", min=2, max=512)
                        dynamic = tuple([{1: length}] * 3 + [None] * 5)
                    exported = torch.export.export(
                        Fused(), self.sample(dtype, tokens), dynamic_shapes=dynamic, strict=True
                    )
                    exported.validate()
                    self.assertEqual(
                        set(
                            exported.run_decompositions(
                                {}
                            ).graph_signature.user_inputs_to_mutate.values()
                        ),
                        {"conv", "state"},
                    )
                    nodes = [
                        node
                        for module in exported.graph_module.modules()
                        if isinstance(module, torch.fx.GraphModule)
                        for node in module.graph.nodes
                    ]
                    self.assertEqual(
                        sum(node.target == torch.ops.higher_order.cond for node in nodes), 0
                    )
                    self.assertEqual(
                        sum(
                            node.target == torch.ops.xinfer.batched_deltanet.default
                            for node in nodes
                        ),
                        1,
                    )


class RecurrentKernels08B(RecurrentKernels, unittest.TestCase):
    pass


class RecurrentKernels4B(RecurrentKernels, unittest.TestCase):
    value_heads = 32


class RecurrentKernels27B(RecurrentKernels, unittest.TestCase):
    value_heads = 48


if __name__ == "__main__":
    unittest.main()
