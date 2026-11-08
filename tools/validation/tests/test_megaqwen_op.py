"""CPU checks for prototype fixtures and strict cross-path comparison failures."""

import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import torch

from compiler.common import attention
from compiler.qwen3.tests.test_megaqwen import metadata_inputs
from tools.validation import megaqwen_op
from tools.validation.megaqwen_op import (
    cache_file_digests,
    caller_storage,
    cases_for_capacity,
    check_caller_storage,
    compare_paths,
    inspect_generated_copies,
    inspect_megaqwen_dispatch,
    require_no_unexpected_copies,
)


class PrototypeContract(unittest.TestCase):
    def test_weight_order_and_pool_views_are_preserved(self):
        names = [name for name, _ in megaqwen_op.WEIGHT_SPECS]
        self.assertEqual(len(names), 311)
        self.assertEqual(len(set(names)), 311)
        self.assertEqual(
            names[:3],
            ["model.model.embed_tokens.weight", "model.model.norm.weight", "model.lm_head.weight"],
        )
        self.assertEqual(names[3], "model.model.layers.0.input_layernorm.weight")
        self.assertEqual(names[-1], "model.model.layers.27.mlp.down_proj.weight")
        weights = metadata_inputs()[1]
        pool = SimpleNamespace(
            manifest={"sources": {name: f"view_{i}" for i, name in enumerate(names)}},
            tensors={f"view_{i}": tensor for i, tensor in enumerate(weights)},
        )
        self.assertTrue(all(a is b for a, b in zip(megaqwen_op.canonical_weights(pool), weights)))
        pool.tensors["view_3"].pointer += 1
        with self.assertRaisesRegex(ValueError, "Unaligned"):
            megaqwen_op.canonical_weights(pool)

    def test_explicit_prototype_has_no_owned_parameters_or_buffers(self):
        step = megaqwen_op.MegaQwenStep()
        self.assertEqual(dict(step.named_buffers()), {})
        self.assertEqual(dict(step.named_parameters()), {})

    def test_replay_uses_the_operator_for_every_singleton_including_position_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            kv = [torch.zeros(1, 1, 6, 1, dtype=torch.bfloat16) for _ in range(2)]
            calls = []

            def scores():
                return torch.zeros(1, 151936, dtype=torch.bfloat16)

            def prefill(ids, positions, cache, extent):
                calls.append(("prefill", positions.tolist()))
                for tensor in cache:
                    tensor[:, :, positions] = 1
                return scores()

            def operator(ids, weights, cache, cos, sin, workspace, extent):
                position = extent.shape[0] - 2
                calls.append(("operator", position))
                for tensor in cache:
                    tensor[:, :, position] = 1
                return scores()

            pool = SimpleNamespace(load=Mock(return_value=prefill))
            cases = [{"name": "mixed", "steps": [[13], [17], [1, 2], [3]]}]
            rows = megaqwen_op.run_python_path(
                "python_op",
                operator,
                pool,
                [],
                kv,
                torch.empty(1),
                torch.empty(1),
                [],
                cases,
                Path(temporary),
                512,
            )
            pool.load.assert_called_once_with("prefill")
            self.assertEqual(
                calls, [("operator", 0), ("operator", 1), ("prefill", [2, 3]), ("operator", 4)]
            )
            self.assertEqual(rows[0]["selected_decode_calls"], 3)

    def test_export_packages_the_build_library_and_its_direct_c_shim(self):
        exported = SimpleNamespace(
            validate=Mock(),
            state_dict={},
            constants={},
            graph_module=SimpleNamespace(code="MegaQwen export graph"),
            graph_signature="fixture signature",
            graph=SimpleNamespace(
                nodes=[SimpleNamespace(op="call_function", target=sentinel.decode)]
            ),
        )
        inputs = (
            None,
            [None],
            [SimpleNamespace(shape=[1, 8, 1024, 128])],
            None,
            None,
            [None],
            None,
        )
        source = "AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_megaqwen_decode(ids, &output));"

        def compile_export(module, destination, **options):
            with zipfile.ZipFile(destination, "w") as archive:
                archive.writestr("model.wrapper.cpp", source)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with (
                patch.object(megaqwen_op, "MegaQwenStep"),
                patch.object(torch.export, "export", return_value=exported),
                patch.object(
                    torch.ops.xinfer,
                    "megaqwen_decode",
                    SimpleNamespace(default=sentinel.decode),
                    create=True,
                ),
                patch.object(megaqwen_op, "megaqwen_library", return_value=sentinel.library),
                patch.object(megaqwen_op, "megaqwen_c_shims", return_value=sentinel.shims),
                patch.object(megaqwen_op, "compile_package", side_effect=compile_export) as compile,
                patch.object(torch._inductor, "aoti_load_package") as load,
            ):
                load.return_value.get_constant_fqns.return_value = []
                compiled, report = megaqwen_op.export_package(inputs, output)
            compile.assert_called_once_with(
                exported,
                output / "megaqwen_decode.pt2",
                inductor_configs={},
                native_libraries=(sentinel.library,),
                custom_op_shims=sentinel.shims,
            )
            self.assertIs(compiled, load.return_value)
            self.assertEqual(len(report["megaqwen_c_shim_calls"]), 1)
            self.assertEqual(report["proxy_executor_calls"], [])

    def test_direct_c_shim_inspection_rejects_proxy_and_missing_decode(self):
        direct = "AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_megaqwen_decode(ids, &output));"
        proxy = "AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_proxy_executor_call_function(executor, 0));"
        self.assertEqual(
            len(inspect_megaqwen_dispatch({"model.wrapper.cpp": direct})["megaqwen_c_shim_calls"]),
            1,
        )
        for source in ("", proxy, direct + proxy, direct + direct):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "one direct"):
                inspect_megaqwen_dispatch({"model.wrapper.cpp": source})

    def test_prototype_requires_only_its_megaqwen_operator_preload(self):
        pair = {"prefill": {}, "decode": {}}
        args = SimpleNamespace(packages=Path("packages"), op_library=Path("mega.so"))
        with (
            patch.object(megaqwen_op, "read_package_metadata", return_value=pair),
            patch.object(megaqwen_op, "validate_model_metadata"),
            patch.object(megaqwen_op, "validate_shared_weights"),
            patch.object(megaqwen_op, "load_megaqwen_ops") as load_mega,
            patch.object(attention, "load_attention_ops") as load_attention,
            patch.object(
                torch,
                "set_num_threads",
                side_effect=RuntimeError("stop before device allocation"),
            ),
            self.assertRaisesRegex(RuntimeError, "stop before device allocation"),
        ):
            megaqwen_op.verify(args, {})
        load_mega.assert_called_once_with(args.op_library)
        load_attention.assert_not_called()

    @staticmethod
    def alignment_source(mask="(16 -1)"):
        return f"""
auto arg7_1 = std::move(inputs[7]);
if ((reinterpret_cast<std::uintptr_t>(arg7_1.data_ptr()) & {mask}) != 0) {{
    AOTI_TORCH_WARN("copy_(text) is only a warning; not an executed copy call");
    AtenTensorHandle arg7_1_aligned;
    aoti_torch_clone_preserve_strides(arg7_1, &arg7_1_aligned);
    arg7_1 = std::move(RAIIAtenTensorHandle(arg7_1_aligned));
}}
"""

    def test_alignment_fallbacks_are_reported_separately(self):
        for mask in ("(16 -1)", "15"):
            with self.subTest(mask=mask):
                result = inspect_generated_copies({"wrapper.cpp": self.alignment_source(mask)})
                require_no_unexpected_copies(result)
                self.assertEqual(result["unexpected_clone_copy_calls"], [])
                fallback = result["guarded_alignment_fallback_copies"]
                self.assertEqual(fallback["count"], 1)
                self.assertEqual(fallback["calls"][0]["input_index"], 7)
                self.assertEqual(fallback["calls"][0]["alignment_bytes"], 16)
                self.assertIn("fallback is not observed", result["zero_copy_scope"])

    def test_unconditional_clone_and_copy_variants_are_rejected(self):
        for call in (
            "aoti_torch_clone_preserve_strides(a, &b)",
            "aoti_torch_clone(a, &b)",
            "aoti_torch_copy_(a, b)",
            "aoti_torch_copy_strided(a, b)",
            "at::clone(a)",
            "at::copy_(a, b)",
            "tensor.clone()",
            "std::copy(a, b, c)",
            "cudaMemcpyAsync(a, b, size, kind, stream)",
        ):
            with self.subTest(call=call):
                source = self.alignment_source() + call + ";"
                result = inspect_generated_copies({"wrapper.cpp": source})
                self.assertEqual(result["guarded_alignment_fallback_copies"]["count"], 1)
                self.assertEqual(len(result["unexpected_clone_copy_calls"]), 1)
                with self.assertRaisesRegex(ValueError, "unclassified or unconditional"):
                    require_no_unexpected_copies(result)

    def test_unrecognized_guards_cannot_hide_clone_calls(self):
        for source in (
            self.alignment_source("31"),
            self.alignment_source().replace("auto arg7_1", "auto other"),
            self.alignment_source().replace("AtenTensorHandle", "other(); AtenTensorHandle", 1),
        ):
            with self.subTest(source=source):
                result = inspect_generated_copies({"wrapper.cpp": source})
                self.assertEqual(result["guarded_alignment_fallback_copies"]["count"], 0)
                with self.assertRaisesRegex(ValueError, "unclassified or unconditional"):
                    require_no_unexpected_copies(result)

    def test_caller_storage_replacement_and_output_alias_are_rejected(self):
        tensors = [torch.empty(1) for _ in range(8)]
        ids, weight, kv, cos, sin, workspace, extent, output = tensors
        inputs = (ids, [weight], [kv], cos, sin, [workspace], extent)
        addresses = caller_storage(inputs)
        check_caller_storage(inputs, addresses, output)
        with self.assertRaisesRegex(ValueError, "Logits alias"):
            check_caller_storage(inputs, addresses, kv)
        replacement = torch.empty_like(kv)
        kv.set_(replacement)
        with self.assertRaisesRegex(ValueError, "replaced caller-owned storage"):
            check_caller_storage(inputs, addresses, output)

    def test_fixtures_cover_thresholds_chunk_tail_and_full_capacity(self):
        for capacity in (512, 1024, 16384, 40960):
            cases = cases_for_capacity(capacity)
            self.assertEqual(len({case["name"] for case in cases}), len(cases))
            self.assertTrue(all(1 <= len(step) <= 512 for case in cases for step in case["steps"]))
            self.assertTrue(all(sum(map(len, case["steps"])) <= capacity for case in cases))
            self.assertEqual(sum(map(len, cases[-1]["steps"])), capacity)
            self.assertEqual(list(map(len, cases[-1]["steps"][-2:])), [1, 1])
            split = next(case for case in cases if case["name"] == "split_256")
            self.assertEqual(list(map(len, split["steps"])), [255, 1, 1, 2, 1])
            self.assertTrue(split["nondefault_stream"])
            poison = next(case for case in cases if "same_as" in case)
            clean = next(case for case in cases if case["name"] == poison["same_as"])
            self.assertEqual(poison["steps"], clean["steps"])
            self.assertEqual(poison["poison_kv"], 3)
            if capacity >= 1026:
                prompt = next(case for case in cases if case["name"] == "prompt_1025")
                self.assertEqual(list(map(len, prompt["steps"])), [512, 512, 1, 1])

    def test_wrong_cache_file_size_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache.bf16"
            path.write_bytes(b"short")
            with self.assertRaisesRegex(ValueError, "cache file size"):
                cache_file_digests(path, 129)

    def test_logits_and_cache_mismatches_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            left, right = Path(temporary) / "left.f32", Path(temporary) / "right.f32"
            left.write_bytes(b"same logits")
            right.write_bytes(left.read_bytes())
            a = {
                "name": "case",
                "position": 129,
                "rows": 3,
                "logits_file": left,
                "valid_kv_sha256": ["a"] * 56,
            }
            b = dict(a, logits_file=right)
            compare_paths([a], [b])
            for changed in (dict(b, position=130), dict(b, valid_kv_sha256=["b"] * 56)):
                with self.assertRaises(ValueError):
                    compare_paths([a], [changed])
            right.write_bytes(b"different logits")
            with self.assertRaisesRegex(ValueError, "logits differ bitwise"):
                compare_paths([a], [b])


if __name__ == "__main__":
    unittest.main()
