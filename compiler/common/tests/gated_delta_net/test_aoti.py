"""Native DeltaNet export, AOTI C-shim parity and in-place CUDA-stream contracts."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.fx.experimental import _config as symbolic_config

from compiler.common.compile import compile_package
from compiler.common.tests.tensors import bits_equal, cpu_copy
from compiler.common.kernels.gated_delta_net.deltanet import (
    batched_deltanet,
    deltanet_c_shims,
    deltanet_library,
    load_deltanet_ops,
)
from compiler.common.tests.gated_delta_net.fixtures import (
    BOUNDARY_LENGTH,
    Fused,
    dispatch_cases,
    inputs_for,
    make_cases,
    run_cases,
)
from compiler.common.tests.gated_delta_net.test_batch import batch_inputs


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
            f"t{BOUNDARY_LENGTH - 1}-state",
            f"t{BOUNDARY_LENGTH}-state",
            f"t{BOUNDARY_LENGTH + 1}-zero",
            f"t{BOUNDARY_LENGTH + 1}-state",
            "t512-state",
            "cond-alternate",
            "cond-alternate-repeat",
        )
    }
    selected = [case for case in dispatch_cases(cases) if case["name"] in names]
    if {case["name"] for case in selected} != names:
        raise AssertionError("Missing compiled fixture")
    return selected


def package_deltanet(exported, path):
    compile_package(
        exported,
        path,
        inductor_configs={"emulate_precision_casts": True},
        native_libraries=[deltanet_library()],
        custom_op_shims=deltanet_c_shims(),
    )
    return torch._inductor.aoti_load_package(str(path))


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
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        load_deltanet_ops()
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
                    decomposed = exported.run_decompositions({})
                    cls.graphs[dtype, mode] = (exported, decomposed)
                    path = Path(temporary.name) / f"{dtype}-{mode}.pt2"
                    cls.packages[dtype, mode] = package_deltanet(exported, path)

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
                cls.results[name] = run_cases(
                    batched_deltanet if name == "eager" else cls.aoti, cls.cases, inplace=True
                )
        return cls.results[name]

    def assert_bits(self, actual, expected):
        self.assertEqual(len(actual), 3)
        self.assertEqual(len(expected), 3)
        for name, a, b in zip(("core", "conv", "recurrent"), actual, expected):
            with self.subTest(tensor=name):
                self.assertTrue(bits_equal(a, b))

    def test_export_and_decomposition_preserve_native_inplace_op_without_cond(self):
        self.assertEqual(len(self.graphs), 4)
        for (dtype, mode), graphs in self.graphs.items():
            for stage, exported in zip(("export", "decomposed"), graphs):
                with self.subTest(dtype=dtype, mode=mode, stage=stage):
                    exported.validate()
                    nodes = [
                        node
                        for module in exported.graph_module.modules()
                        if isinstance(module, torch.fx.GraphModule)
                        for node in module.graph.nodes
                    ]
                    self.assertEqual(
                        sum(node.target == torch.ops.higher_order.cond for node in nodes), 0
                    )
                    if stage == "export":
                        self.assertEqual(
                            sum(
                                node.target == torch.ops.xinfer.batched_deltanet.default
                                for node in nodes
                            ),
                            1,
                        )
                    else:
                        self.assertEqual(
                            set(exported.graph_signature.user_inputs_to_mutate.values()),
                            {"conv", "state"},
                        )
                        self.assertEqual(
                            sum(
                                node.target == torch.ops.higher_order.auto_functionalized_v2
                                and node.args
                                and node.args[0] == torch.ops.xinfer.batched_deltanet.default
                                for node in nodes
                            ),
                            1,
                        )

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
                    self.assert_bits(results[original, step], results[repeated, step])

    @torch.inference_mode()
    def test_nondefault_stream_matches_eager_bitwise(self):
        for dtype in ("bf16", "fp32"):
            for tokens in (1, BOUNDARY_LENGTH - 1, BOUNDARY_LENGTH + 1):
                with self.subTest(dtype=dtype, tokens=tokens):
                    case = next(c for c in self.cases if c["name"] == f"{dtype}-t{tokens}-state")
                    eager_inputs = inputs_for(case)
                    aoti_inputs = tuple(x.clone() for x in eager_inputs)
                    eager_core = batched_deltanet(*eager_inputs)
                    expected = cpu_copy((eager_core, eager_inputs[3], eager_inputs[4]))
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        actual_core = self.aoti(*aoti_inputs)
                    stream.synchronize()
                    self.assert_bits(
                        cpu_copy((actual_core, aoti_inputs[3], aoti_inputs[4])), expected
                    )


class AOTIKernels08B(AOTIKernels, unittest.TestCase):
    pass


class AOTIKernels4B(AOTIKernels, unittest.TestCase):
    value_heads = 32


class AOTIKernels27B(AOTIKernels, unittest.TestCase):
    value_heads = 48


def export_batched(dtype, mode, *, decay_dtype=None):
    sample = list(batch_inputs(2, 32 if mode == "prefill" else 1, dtype))
    if decay_dtype is not None:
        sample[6] = sample[6].to(decay_dtype)
    sample = tuple(sample)
    batch = torch.export.Dim("batch", min=1, max=8)
    token_shape = {0: batch}
    if mode == "prefill":
        token_shape[1] = torch.export.Dim("tokens", min=2, max=512)
    shapes = (*[token_shape] * 3, {0: batch}, {0: batch}, None, None, None, {0: batch})
    # The same setting used by Qwen3 keeps singleton B inside AOTI's guards.
    with symbolic_config.patch(backed_size_oblivious=True):
        exported = torch.export.export(Fused(), sample, dynamic_shapes=shapes, strict=True)
    exported.validate()
    return exported


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class BatchedAOTI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        load_deltanet_ops()
        guards = patch.dict(os.environ, {"AOTI_RUNTIME_CHECK_INPUTS": "1"})
        guards.start()
        cls.addClassCleanup(guards.stop)
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.directory = Path(temporary.name)
        cls.packages, cls.graphs = {}, {}
        cls.addClassCleanup(cls.packages.clear)
        with torch.inference_mode(), symbolic_config.patch(backed_size_oblivious=True):
            for dtype in (torch.bfloat16, torch.float32):
                for mode in ("prefill", "decode"):
                    exported = export_batched(dtype, mode)
                    cls.graphs[dtype, mode] = exported
                    path = cls.directory / f"{dtype}-{mode}.pt2"
                    cls.packages[dtype, mode] = package_deltanet(exported, path)

    @classmethod
    def call(cls, *inputs):
        mode = "decode" if inputs[0].shape[1] == 1 else "prefill"
        return cls.packages[inputs[0].dtype, mode](*inputs)

    def test_batch_range_and_single_native_op_without_cond(self):
        for (dtype, mode), exported in self.graphs.items():
            with self.subTest(dtype=dtype, mode=mode):
                symbol = exported.graph_module.graph.nodes
                qkv = next(n for n in symbol if n.op == "placeholder")
                value_range = exported.range_constraints[qkv.meta["val"].shape[0].node.expr]
                self.assertEqual((int(value_range.lower), int(value_range.upper)), (1, 8))
                self.assertEqual(
                    sum(n.target == torch.ops.higher_order.cond for n in exported.graph.nodes), 0
                )
                self.assertEqual(
                    sum(
                        n.target == torch.ops.xinfer.batched_deltanet.default
                        for n in exported.graph.nodes
                    ),
                    1,
                )
                self.assertEqual(
                    set(
                        exported.run_decompositions(
                            {}
                        ).graph_signature.user_inputs_to_mutate.values()
                    ),
                    {"conv", "state"},
                )

    @torch.inference_mode()
    def test_all_batches_and_prefix_views_match_eager_bitwise(self):
        for dtype in (torch.bfloat16, torch.float32):
            for batch in range(1, 9):
                for tokens in (1, 2, 97, 98, 512):
                    with self.subTest(dtype=dtype, batch=batch, tokens=tokens):
                        full = batch_inputs(8, tokens, dtype)
                        counts = batch_inputs(batch, tokens, dtype)[-1]
                        aoti_conv, aoti_state = full[3].clone(), full[4].clone()
                        eager_conv, eager_state = full[3].clone(), full[4].clone()
                        tail_conv = full[3][batch:].clone()
                        tail_state = full[4][batch:].clone()
                        conv_ptr, state_ptr = aoti_conv.data_ptr(), aoti_state.data_ptr()

                        qkv, b, a = (x[:batch].contiguous() for x in full[:3])
                        read_only_before = cpu_copy((qkv, b, a, full[5], full[6], full[7], counts))

                        actual_core = self.call(
                            qkv,
                            b,
                            a,
                            aoti_conv[:batch],
                            aoti_state[:batch],
                            full[5],
                            full[6],
                            full[7],
                            counts,
                        )
                        expected_core = batched_deltanet(
                            qkv,
                            b,
                            a,
                            eager_conv[:batch],
                            eager_state[:batch],
                            full[5],
                            full[6],
                            full[7],
                            query_lengths=counts,
                        )
                        self.assertEqual(aoti_conv.data_ptr(), conv_ptr)
                        self.assertEqual(aoti_state.data_ptr(), state_ptr)
                        self.assertTrue(bits_equal(actual_core, expected_core))
                        self.assertTrue(bits_equal(aoti_conv[:batch], eager_conv[:batch]))
                        self.assertTrue(bits_equal(aoti_state[:batch], eager_state[:batch]))
                        self.assertTrue(bits_equal(aoti_conv[batch:], tail_conv))
                        self.assertTrue(bits_equal(aoti_state[batch:], tail_state))
                        self.assertTrue(
                            all(
                                bits_equal(x, y)
                                for x, y in zip(
                                    cpu_copy((qkv, b, a, full[5], full[6], full[7], counts)),
                                    read_only_before,
                                )
                            )
                        )
                        self.assertFalse(
                            any(
                                torch._C._is_alias_of(actual_core, t)
                                for t in (
                                    qkv,
                                    b,
                                    a,
                                    aoti_conv,
                                    aoti_state,
                                    full[5],
                                    full[6],
                                    full[7],
                                )
                            )
                        )

    @torch.inference_mode()
    def test_carried_state_and_nondefault_stream(self):
        for dtype in (torch.bfloat16, torch.float32):
            eager_conv = eager_state = aoti_conv = aoti_state = None
            conv_ptr = state_ptr = None
            for tokens in (98, 1, 97, 129, 1):
                inputs = list(batch_inputs(8, tokens, dtype))
                if eager_conv is None:
                    eager_conv, eager_state = inputs[3].clone(), inputs[4].clone()
                    aoti_conv, aoti_state = inputs[3].clone(), inputs[4].clone()
                    conv_ptr, state_ptr = aoti_conv.data_ptr(), aoti_state.data_ptr()
                expected_core = batched_deltanet(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    eager_conv,
                    eager_state,
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    query_lengths=inputs[8],
                )
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    actual_core = self.call(
                        inputs[0],
                        inputs[1],
                        inputs[2],
                        aoti_conv,
                        aoti_state,
                        inputs[5],
                        inputs[6],
                        inputs[7],
                        inputs[8],
                    )
                stream.synchronize()
                with self.subTest(dtype=dtype, tokens=tokens):
                    self.assertEqual(aoti_conv.data_ptr(), conv_ptr)
                    self.assertEqual(aoti_state.data_ptr(), state_ptr)
                    self.assertTrue(bits_equal(actual_core, expected_core))
                    self.assertTrue(bits_equal(aoti_conv, eager_conv))
                    self.assertTrue(bits_equal(aoti_state, eager_state))

    @torch.inference_mode()
    def test_out_of_range_batch_is_rejected(self):
        for batch in (0, 9):
            inputs = list(batch_inputs(8, 1, torch.bfloat16))
            for index in (0, 1, 2, 3, 4, 8):
                source = inputs[index]
                inputs[index] = source[:0] if batch == 0 else torch.cat((source, source[:1]))
            with self.subTest(batch=batch), self.assertRaises((RuntimeError, ValueError)):
                self.call(*inputs)

    @torch.inference_mode()
    def test_bf16_with_fp32_decay_parameter_matches_eager_bitwise(self):
        with tempfile.TemporaryDirectory() as temporary:
            packages = {}
            with symbolic_config.patch(backed_size_oblivious=True):
                for mode in ("prefill", "decode"):
                    exported = export_batched(torch.bfloat16, mode, decay_dtype=torch.float32)
                    path = Path(temporary) / f"mixed-{mode}.pt2"
                    packages[mode] = package_deltanet(exported, path)
            try:
                for batch in (1, 8):
                    for tokens in (1, 97, 98):
                        with self.subTest(batch=batch, tokens=tokens):
                            inputs = list(batch_inputs(batch, tokens, torch.bfloat16))
                            inputs[6] = torch.linspace(
                                -2.5, 0.5, 16, device="cuda", dtype=torch.float32
                            )
                            aoti_inputs = [x.clone() for x in inputs]
                            eager_inputs = [x.clone() for x in inputs]
                            actual_core = packages["decode" if tokens == 1 else "prefill"](
                                *aoti_inputs
                            )
                            expected_core = batched_deltanet(
                                *eager_inputs[:8], query_lengths=eager_inputs[8]
                            )
                            self.assertTrue(bits_equal(actual_core, expected_core))
                            self.assertTrue(bits_equal(aoti_inputs[3], eager_inputs[3]))
                            self.assertTrue(bits_equal(aoti_inputs[4], eager_inputs[4]))
            finally:
                packages.clear()


if __name__ == "__main__":
    unittest.main()
