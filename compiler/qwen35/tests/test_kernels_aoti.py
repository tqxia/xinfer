"""DeltaNet export, AOTI parity and compiled CUDA-stream contracts."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from compiler.common.tests.tensors import bits_equal, cpu_copy
from compiler.qwen35.config import configure
from compiler.qwen35.kernels.recurrent import SHORT_THRESHOLD, fused
from compiler.qwen35.tests.delta import (
    Fused,
    dispatch_cases,
    inputs_for,
    make_cases,
    run_cases,
)


def compiled_cases(cases):
    """Keep the compiled boundary and state matrix separate from numerical cases."""
    names = {
        f"{dtype}-{name}"
        for dtype in ("bf16", "fp32")
        for name in (
            "t1-zero",
            "t1-state",
            "t2-zero",
            "t2-state",
            f"t{SHORT_THRESHOLD - 1}-state",
            f"t{SHORT_THRESHOLD}-state",
            f"t{SHORT_THRESHOLD + 1}-zero",
            f"t{SHORT_THRESHOLD + 1}-state",
            "t512-state",
            "cond-alternate",
            "cond-alternate-repeat",
        )
    }
    selected = [case for case in dispatch_cases(cases) if case["name"] in names]
    if {case["name"] for case in selected} != names:
        raise AssertionError("Missing compiled fixture")
    return selected


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class AOTIKernels:
    value_heads = 16

    @classmethod
    def setUpClass(cls):
        guards = patch.dict(os.environ, {"AOTI_RUNTIME_CHECK_INPUTS": "1"})
        guards.start()
        cls.addClassCleanup(guards.stop)
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.packages, cls.graphs, cls.results = {}, {}, {}
        cls.cases = []
        # Release package loaders before deleting their temporary archives.
        cls.addClassCleanup(cls.clear_runtime)
        configure()
        torch.manual_seed(35)
        original = make_cases(cls.value_heads)
        cls.cases = compiled_cases(original)
        with torch.inference_mode():
            for dtype in ("bf16", "fp32"):
                for mode, tokens in (("decode", 1), ("prefill", 32)):
                    case = next(c for c in original if c["name"] == f"{dtype}-t{tokens}-state")
                    sample = inputs_for(case)
                    dynamic = None
                    if mode == "prefill":
                        length = torch.export.Dim("tokens", min=2, max=512)
                        dynamic = tuple([{1: length}] * 3 + [None] * 5)
                    exported = torch.export.export(
                        Fused(), sample, dynamic_shapes=dynamic, strict=True
                    )
                    decomposed = exported.run_decompositions({}, decompose_custom_triton_ops=True)
                    cls.graphs[dtype, mode] = (exported, decomposed)
                    path = Path(temporary.name) / f"{dtype}-{mode}.pt2"
                    torch._inductor.aoti_compile_and_package(
                        exported,
                        package_path=str(path),
                        inductor_configs={"emulate_precision_casts": True},
                    )
                    cls.packages[dtype, mode] = torch._inductor.aoti_load_package(str(path))

    @classmethod
    def clear_runtime(cls):
        cls.packages.clear()
        cls.graphs.clear()
        cls.results.clear()
        cls.cases.clear()

    @classmethod
    def aoti(cls, *inputs):
        return cls.packages[
            "bf16" if inputs[0].dtype == torch.bfloat16 else "fp32",
            "decode" if inputs[0].shape[1] == 1 else "prefill",
        ](*inputs)

    @classmethod
    def engine(cls, name):
        if name not in cls.results:
            with torch.inference_mode():
                cls.results[name] = run_cases(fused if name == "eager" else cls.aoti, cls.cases)
        return cls.results[name]

    def assert_bits(self, actual, expected):
        self.assertEqual(len(actual), 3)
        self.assertEqual(len(expected), 3)
        for name, a, b in zip(("core", "conv", "recurrent"), actual, expected):
            with self.subTest(tensor=name):
                self.assertTrue(bits_equal(a, b))

    def test_export_and_decomposition_preserve_symbolic_dispatch(self):
        self.assertEqual(len(self.graphs), 4)
        for (dtype, mode), graphs in self.graphs.items():
            for stage, exported in zip(("export", "decomposed"), graphs):
                with self.subTest(dtype=dtype, mode=mode, stage=stage):
                    exported.validate()
                    self.assertFalse(exported.graph_signature.user_inputs_to_mutate)
                    nodes = [
                        node
                        for module in exported.graph_module.modules()
                        if isinstance(module, torch.fx.GraphModule)
                        for node in module.graph.nodes
                    ]
                    conditions = [
                        node for node in nodes if node.target == torch.ops.higher_order.cond
                    ]
                    self.assertEqual(len(conditions), int(mode == "prefill"))
                    if conditions:
                        predicate = conditions[0].args[0]
                        self.assertIsInstance(predicate, torch.fx.Node)
                        value = predicate.meta.get("val")
                        self.assertIsInstance(value, torch.SymBool)
                        self.assertNotIn(str(value), ("True", "False"))

    def test_python_aoti_matches_eager_bitwise(self):
        actual, expected = self.engine("aoti"), self.engine("eager")
        self.assertEqual(actual.keys(), expected.keys())
        self.assertEqual(len(actual), sum(len(case["steps"]) for case in self.cases))
        for key, tensors in actual.items():
            with self.subTest(case=key[0], step=key[1]):
                self.assert_bits(tensors, expected[key])

    def test_reset_repeat_bitwise(self):
        results = self.engine("aoti")
        for dtype in ("bf16", "fp32"):
            original = f"{dtype}-cond-alternate"
            repeated = f"{original}-repeat"
            case = next(case for case in self.cases if case["name"] == original)
            self.assertGreater(len(case["steps"]), 0)
            for step in range(len(case["steps"])):
                with self.subTest(dtype=dtype, step=step):
                    self.assertIn((original, step), results)
                    self.assertIn((repeated, step), results)
                    self.assert_bits(results[original, step], results[repeated, step])

    @torch.inference_mode()
    def test_nondefault_stream(self):
        for dtype in ("bf16", "fp32"):
            for tokens in (1, SHORT_THRESHOLD, SHORT_THRESHOLD + 1):
                with self.subTest(dtype=dtype, tokens=tokens):
                    name = f"{dtype}-t{tokens}-state"
                    sample = inputs_for(next(case for case in self.cases if case["name"] == name))
                    expected = cpu_copy(self.aoti(*sample))
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        actual = self.aoti(*sample)
                    stream.synchronize()
                    self.assert_bits(cpu_copy(actual), expected)


class AOTIKernels08B(AOTIKernels, unittest.TestCase):
    pass


class AOTIKernels4B(AOTIKernels, unittest.TestCase):
    value_heads = 32


if __name__ == "__main__":
    unittest.main()
