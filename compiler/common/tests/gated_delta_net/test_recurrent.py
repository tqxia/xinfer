"""Fused/conditional DeltaNet behavior, numerical controls and export contracts."""

import unittest

import torch

from compiler.common.tests.tensors import bits_equal, cpu_copy
from compiler.common.kernels.gated_delta_net.chunked import CHUNK_SIZE
from compiler.common.kernels.gated_delta_net.recurrent import SHORT_THRESHOLD, fused
from compiler.common.tests.gated_delta_net.fixtures import (
    PARTITIONS,
    Fused,
    compare,
    dispatch_cases,
    inputs_for,
    make_cases,
    reference,
    run_cases,
    selected,
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
                        fused(*broken)
            with self.subTest(value_heads=value_heads, field="device"):
                with self.assertRaisesRegex(ValueError, "Expected CUDA"):
                    fused(*sample)
        with self.assertRaisesRegex(ValueError, "Hk=16"):
            fused(*inputs(24))

    def test_kernel_validates_decay_parameter_dtypes_before_cuda(self):
        fp32 = inputs(16)
        for index, name in ((6, "A_log"), (7, "dt_bias")):
            broken = list(fp32)
            broken[index] = broken[index].bfloat16()
            with self.subTest(qkv="fp32", field=name):
                with self.assertRaisesRegex(ValueError, "Expected matching BF16 or FP32"):
                    fused(*broken)
        bf16 = tuple(tensor if idx == 4 else tensor.bfloat16() for idx, tensor in enumerate(fp32))
        for index, name in ((6, "A_log"), (7, "dt_bias")):
            broken = list(bf16)
            broken[index] = broken[index].half()
            with self.subTest(qkv="bf16_half", field=name):
                with self.assertRaisesRegex(ValueError, "Expected matching BF16 or FP32"):
                    fused(*broken)
            mixed = list(bf16)
            mixed[index] = mixed[index].float()
            with self.subTest(qkv="bf16_fp32_decay", field=name):
                with self.assertRaisesRegex(ValueError, "Expected CUDA"):
                    fused(*mixed)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class RecurrentKernels:
    value_heads = 16

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(35)
        cls.cases = dispatch_cases(make_cases(cls.value_heads))
        with torch.inference_mode():
            cls.results = {"dispatch": run_cases(fused, cls.cases)}

    @classmethod
    def tearDownClass(cls):
        del cls.results, cls.cases

    @classmethod
    def engine(cls, name):
        if name not in cls.results:
            call = {
                "branch": selected,
                "hf": reference,
                "hf_chunk64": lambda *x: reference(*x, chunked=True),
            }[name]
            with torch.inference_mode():
                cls.results[name] = run_cases(call, cls.cases)
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

    def test_dispatch_matches_independent_branch_selection(self):
        self.compare(
            "dispatch_vs_branch", self.engine("dispatch"), self.engine("branch"), (0, 1, 2)
        )

    def test_hf_recurrent_and_chunked_references(self):
        self.assertEqual(CHUNK_SIZE, 64, "The HF chunk-64 control must stay independent")
        for engine in ("hf", "hf_chunk64"):
            with self.subTest(reference=engine):
                self.compare(f"dispatch_vs_{engine}", self.engine("dispatch"), self.engine(engine))

    def test_partition_invariance(self):
        for engine in ("dispatch", "branch", "hf", "hf_chunk64"):
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
        for engine in ("dispatch", "branch", "hf", "hf_chunk64"):
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
            for tokens in (SHORT_THRESHOLD - 1, SHORT_THRESHOLD + 1):
                with self.subTest(dtype=dtype, tokens=tokens):
                    sample = self.sample(dtype, tokens)
                    expected = cpu_copy(fused(*sample))
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        actual = fused(*(x.clone() for x in sample))
                    stream.synchronize()
                    self.assertTrue(
                        all(bits_equal(a, b) for a, b in zip(cpu_copy(actual), expected))
                    )
                    strided = list(sample)
                    backing = sample[0].new_empty((*sample[0].shape[:-1], sample[0].shape[-1] * 2))
                    backing[..., ::2].copy_(sample[0])
                    strided[0], strided[5] = backing[..., ::2], sample[5].t().contiguous().t()
                    self.assertFalse(strided[0].is_contiguous())
                    self.assertFalse(strided[5].is_contiguous())
                    self.assertTrue(
                        all(bits_equal(a, b) for a, b in zip(cpu_copy(fused(*strided)), expected))
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
                cpu=[x.cpu() for x in sample],
            )
            for name, inputs in invalid.items():
                with self.subTest(dtype=dtype, invalid=name):
                    with self.assertRaises((ValueError, RuntimeError)):
                        fused(*inputs)

    @torch.inference_mode()
    def test_operator_schema_and_fake_tensors(self):
        for dtype in ("bf16", "fp32"):
            sample = self.sample(dtype, 32)
            for op in (
                torch.ops.xinfer.delta_cond_short.default,
                torch.ops.xinfer.delta_cond_long.default,
            ):
                with self.subTest(dtype=dtype, operator=str(op)):
                    checks = torch.library.opcheck(
                        op,
                        (*sample, torch.tensor([32], device="cuda", dtype=torch.int32)),
                        test_utils=("test_schema", "test_faketensor"),
                    )
                    self.assertTrue(all(value == "SUCCESS" for value in checks.values()))

    def test_strict_export_preserves_symbolic_dispatch(self):
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
                    self.assertFalse(exported.graph_signature.user_inputs_to_mutate)
                    nodes = [
                        node
                        for module in exported.graph_module.modules()
                        if isinstance(module, torch.fx.GraphModule)
                        for node in module.graph.nodes
                    ]
                    cond = [node for node in nodes if node.target == torch.ops.higher_order.cond]
                    self.assertEqual(len(cond), int(mode == "prefill"))
                    if cond:
                        predicate = cond[0].args[0].meta["val"]
                        self.assertIsInstance(predicate, torch.SymBool)
                        self.assertNotIn(str(predicate), ("True", "False"))
                    self.assertEqual(
                        sum(
                            node.target == torch.ops.xinfer.delta_cond_short.default
                            for node in nodes
                        ),
                        1,
                    )
                    self.assertEqual(
                        sum(
                            node.target == torch.ops.xinfer.delta_cond_long.default
                            for node in nodes
                        ),
                        int(mode == "prefill"),
                    )


class RecurrentKernels08B(RecurrentKernels, unittest.TestCase):
    pass


class RecurrentKernels4B(RecurrentKernels, unittest.TestCase):
    value_heads = 32


class RecurrentKernels27B(RecurrentKernels, unittest.TestCase):
    value_heads = 48


if __name__ == "__main__":
    unittest.main()
