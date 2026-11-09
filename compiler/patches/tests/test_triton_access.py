"""CPU contracts for the pinned Triton read-dependency compatibility context."""

import hashlib
import inspect
import unittest
from unittest.mock import patch

from torch._higher_order_ops import triton_kernel_wrap as implementation

from compiler.patches import triton_access


Param, Intermediate, Op = implementation.Param, implementation.Intermediate, implementation.Op


def operation(index, name, args, **kwargs):
    result = Intermediate(index)
    return result, [Op(name, None, args, result, **kwargs)]


def analyze(functions, name="kernel", num_args=3):
    implementation.analyze_kernel_access.reset()
    implementation.get_tma_stores.reset()
    accesses = implementation.analyze_kernel_access(
        functions,
        name,
        num_args,
        tuple(f"arg{i}" for i in range(num_args)),
        frozenset(range(num_args)),
    )
    return (
        accesses.read_writes.reads,
        accesses.read_writes.writes,
        accesses.read_writes.index_exprs,
        accesses.can_fuse_epilogue,
    )


def diamond(depth, stores=1):
    ops, previous = {}, Param(0)
    for index in range(depth):
        result, operations = operation(index, "arith.addi", [previous, previous])
        ops[result], previous = operations, result
    result, operations = operation(depth, "tt.load", [previous])
    ops[result] = operations
    for index in range(stores):
        result, operations = operation(-index - 1, "tt.store", [Param(1), result])
        ops[result] = operations
    return {"kernel": ops}


class TritonReadAccess(unittest.TestCase):
    def tearDown(self):
        implementation.analyze_kernel_access.reset()
        implementation.get_tma_stores.reset()

    def assert_unchanged(self, functions):
        expected = analyze(functions)
        with triton_access.memoize_triton_reads():
            self.assertEqual(analyze(functions), expected)
        return expected

    def test_shared_read_dag_and_duplicate_stores_preserve_exact_contract(self):
        for stores in (1, 2):
            with self.subTest(stores=stores):
                result = self.assert_unchanged(diamond(12, stores))
                self.assertEqual({dep.name for dep in result[0]}, {"arg0"})
                self.assertEqual({dep.name for dep in result[1]}, {"arg1"})
                self.assertEqual(result[3], stores == 1)

    def test_duplicate_write_paths_are_not_deduplicated(self):
        ops = dict(
            [
                operation(0, "arith.addptr", [Param(1), Param(1)]),
                operation(-1, "tt.store", [Intermediate(0), Param(0)]),
            ]
        )
        result = self.assert_unchanged({"kernel": ops})
        self.assertFalse(result[3], "Two paths to one written parameter still count twice")

    def test_load_derived_store_pointer_does_not_mutate_loaded_input(self):
        ops = dict(
            [
                operation(0, "tt.load", [Param(0)]),
                operation(1, "arith.addptr", [Param(1), Intermediate(0)]),
                operation(-1, "tt.store", [Intermediate(1), Param(2)]),
            ]
        )
        result = self.assert_unchanged({"kernel": ops})
        self.assertEqual({dep.name for dep in result[0]}, {"arg0"})
        self.assertEqual({dep.name for dep in result[1]}, {"arg1"})
        self.assertTrue(result[3])

    def test_nested_function_uses_original_module_analyzer_and_cache(self):
        inner = dict(
            [
                operation(0, "tt.load", [Param(0)]),
                operation(-1, "tt.store", [Param(1), Intermediate(0)]),
            ]
        )
        result = Intermediate(0)
        outer = {result: [Op("tt.call", "inner", [Param(0), Param(1)], result)]}
        result = self.assert_unchanged({"kernel": outer, "inner": inner})
        self.assertEqual({dep.name for dep in result[0]}, {"arg0"})
        self.assertEqual({dep.name for dep in result[1]}, {"arg1"})
        self.assertTrue(result[3])

    def test_tma_descriptor_metadata_is_not_analyzed_as_memory(self):
        ops = dict(
            [
                operation(0, "tt.make_tensor_descriptor", [Param(0), Param(2)]),
                operation(1, "tt.descriptor_load", [Intermediate(0)]),
                operation(2, "tt.make_tensor_descriptor", [Param(1), Param(2)]),
                operation(-1, "tt.descriptor_store", [Intermediate(2), Intermediate(1)]),
            ]
        )
        result = self.assert_unchanged({"kernel": ops})
        self.assertEqual({dep.name for dep in result[0]}, {"arg0"})
        self.assertEqual({dep.name for dep in result[1]}, {"arg1"})

    def test_unknown_impure_assembly_is_still_rejected(self):
        functions = {"kernel": dict([operation(0, "tt.elementwise_inline_asm", [Param(0)])])}
        for patched in (False, True):
            with self.subTest(patched=patched):
                if patched:
                    with triton_access.memoize_triton_reads():
                        with self.assertRaisesRegex(RuntimeError, "do not know how to analyze"):
                            analyze(functions)
                else:
                    with self.assertRaisesRegex(RuntimeError, "do not know how to analyze"):
                        analyze(functions)

    def test_deep_shared_dag_has_linear_intermediate_visits(self):
        class CountedOps(dict):
            visits = 0

            def __getitem__(self, key):
                self.visits += 1
                if self.visits > 4 * depth:
                    raise AssertionError("Read DAG traversal revisited too many intermediates")
                return super().__getitem__(key)

        depth = 256
        ops = CountedOps(diamond(depth)["kernel"])
        with triton_access.memoize_triton_reads():
            result = analyze({"kernel": ops})
        self.assertLessEqual(ops.visits, depth + 1)
        self.assertEqual({dep.name for dep in result[0]}, {"arg0"})
        self.assertTrue(result[3])

    def test_context_resets_caches_and_restores_function_on_exception(self):
        analyzer = implementation.analyze_kernel_access
        original = analyzer.fn
        analyze(diamond(2))
        self.assertTrue(analyzer.cache)
        with self.assertRaisesRegex(RuntimeError, "compile failed"):
            with triton_access.memoize_triton_reads():
                self.assertIsNot(analyzer.fn, original)
                self.assertIs(analyzer.fn.__globals__, vars(implementation))
                self.assertEqual(analyzer.cache, {})
                self.assertEqual(implementation.get_tma_stores.cache, {})
                analyze(diamond(2))
                raise RuntimeError("compile failed")
        self.assertIs(analyzer.fn, original)
        self.assertEqual(analyzer.cache, {})
        self.assertEqual(implementation.get_tma_stores.cache, {})

    def test_unsupported_version_and_source_fail_before_mutation(self):
        analyzer = implementation.analyze_kernel_access
        original = analyzer.fn
        for mock, message in (
            (patch("torch.__version__", "2.14.0"), "requires torch==2.13.0"),
            (patch("inspect.getsource", return_value="unsupported source"), "analyzer source"),
        ):
            with self.subTest(message=message), mock:
                analyzer.cache["preserved"] = True
                with self.assertRaisesRegex(ValueError, message):
                    with triton_access.memoize_triton_reads():
                        self.fail("Unsupported analyzer entered the context")
                self.assertIs(analyzer.fn, original)
                self.assertEqual(analyzer.cache, {"preserved": True})
                analyzer.reset()

    def test_ast_shape_guard_rejects_an_unexpected_nested_function(self):
        analyzer = implementation.analyze_kernel_access
        original = analyzer.fn
        source = inspect.getsource(original).replace(
            "def _find_arg_access_count(", "def _unexpected_nested_function("
        )
        with (
            patch("inspect.getsource", return_value=source),
            patch.object(
                triton_access, "ANALYZER_SHA256", hashlib.sha256(source.encode()).hexdigest()
            ),
            self.assertRaisesRegex(ValueError, "analyzer structure"),
        ):
            with triton_access.memoize_triton_reads():
                self.fail("Unexpected analyzer structure entered the context")
        self.assertIs(analyzer.fn, original)


if __name__ == "__main__":
    unittest.main()
