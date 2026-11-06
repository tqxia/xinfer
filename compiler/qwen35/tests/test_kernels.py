"""Fused/conditional DeltaNet behavior, numerical controls and export contracts."""

import unittest

import torch

from compiler.common.tests.tensors import bits_equal, cpu_copy
from compiler.qwen35.config import configure
from compiler.qwen35.kernels.chunked import CHUNK_SIZE
from compiler.qwen35.kernels.recurrent import SHORT_THRESHOLD, fused
from compiler.qwen35.tests.delta import (
    ABS_LIMITS,
    LIMITS,
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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class RecurrentKernels:
    value_heads = 16

    @classmethod
    def setUpClass(cls):
        configure()
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
                "fp32": reference,
            }[name]
            with torch.inference_mode():
                cls.results[name] = run_cases(call, cls.cases, fp32=name == "fp32")
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

    def test_fp32_reference_controls(self):
        # Keep the original BF16-to-FP32 acceptance policy for both implementations.
        # A pre-existing numerical gap remains a failure, not an expected failure.
        for engine in ("dispatch", "hf"):
            with self.subTest(engine=engine, limits=LIMITS, absolute_limits=ABS_LIMITS):
                self.compare(f"{engine}_vs_fp32", self.engine(engine), self.engine("fp32"))

    def test_partition_invariance(self):
        for engine in ("dispatch", "branch", "hf", "hf_chunk64", "fp32"):
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
        for engine in ("dispatch", "branch", "hf", "hf_chunk64", "fp32"):
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
                        op, sample, test_utils=("test_schema", "test_faketensor")
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


if __name__ == "__main__":
    unittest.main()
