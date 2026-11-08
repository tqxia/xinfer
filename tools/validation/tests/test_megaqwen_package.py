"""CPU checks for shipping-package routing, storage contracts and trace parity."""

import copy
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from compiler.common import attention
from compiler.qwen3 import megaqwen
from tools.validation.megaqwen_op import cache_file_digests
from tools.validation.qwen3 import megaqwen_package as package_contract
from tools.validation.qwen3.megaqwen_package import (
    check_native,
    expected_counts,
    inspect_package,
)


class PackageContract(unittest.TestCase):
    def test_package_validation_leaves_both_operator_libraries_to_aoti(self):
        args = SimpleNamespace(packages=Path("packages"))
        for backend in ("aoti", "megaqwen"):
            extra = {} if backend == "aoti" else {"decode_backend": "megaqwen"}
            pair = {"decode": {"model_specific": extra, "state": []}}
            with (
                self.subTest(backend=backend),
                patch.object(package_contract, "read_package_metadata", return_value=pair),
                patch.object(megaqwen, "load_megaqwen_ops") as load_mega,
                patch.object(attention, "load_attention_ops") as load_attention,
                patch.object(package_contract.torch, "set_num_threads"),
                patch.object(
                    package_contract,
                    "load_aoti_runtime",
                    side_effect=RuntimeError("stop before device allocation"),
                ),
                patch(
                    "subprocess.check_output", side_effect=AssertionError("No native DSO lookup")
                ),
                patch.dict(
                    os.environ,
                    XINFER_MEGAQWEN_LIBRARY="unrelated-mega.so",
                    XINFER_ATTENTION_OPS_LIBRARY="unrelated-attention.so",
                ),
                self.assertRaisesRegex(RuntimeError, "stop before device allocation"),
            ):
                package_contract.verify(args, {})
            load_mega.assert_not_called()
            load_attention.assert_not_called()

    def test_native_parser_rejects_missing_storage_checks_and_corrupted_traces(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python_file = root / "python.f32"
            native_file = root / "case.f32"
            python_file.write_bytes(b"same float32 fixture")
            native_file.write_bytes(python_file.read_bytes())
            cache = root / "case.kv.bf16"
            cache.write_bytes(bytes(56 * 8 * 128 * 2))
            case = {"name": "case", "steps": [[13]], "dump_cache": True}
            expected = {
                "name": "case",
                **expected_counts([1]),
                "logits_file": str(python_file),
                "valid_kv_sha256": cache_file_digests(cache, 1),
            }
            native = {
                "passed": True,
                "no_python_library": True,
                "async_destruction_lifetime": True,
                "cases": [
                    {
                        "name": "case",
                        **expected_counts([1]),
                        "passed": True,
                        "caller_owned_storage_preserved": True,
                    }
                ],
            }
            check_native([case], [expected], native, root)
            for mutate in (
                lambda report: report["cases"][0].__setitem__(
                    "caller_owned_storage_preserved", False
                ),
                lambda report: report["cases"][0].__setitem__("decode_calls", 0),
                lambda report: report.__setitem__("async_destruction_lifetime", False),
            ):
                changed = copy.deepcopy(native)
                mutate(changed)
                with self.assertRaises(ValueError):
                    check_native([case], [expected], changed, root)
            native_file.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "logits differ bitwise"):
                check_native([case], [expected], native, root)
            native_file.write_bytes(python_file.read_bytes())
            cache.write_bytes(bytes([1]) * cache.stat().st_size)
            with self.assertRaisesRegex(ValueError, "KV differs bitwise"):
                check_native([case], [expected], native, root)

    def test_generic_counts_include_initial_singleton_and_prompt_tail(self):
        self.assertEqual(
            expected_counts([1, 1, 1]),
            {"position": 3, "rows": 3, "prefill_calls": 0, "decode_calls": 3},
        )
        self.assertEqual(expected_counts([512, 512, 1, 1])["decode_calls"], 2)

    def test_unclassified_copies_block_mega_export_but_default_is_diagnostic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "decode.pt2"
            with zipfile.ZipFile(package, "w") as archive:
                archive.writestr("wrapper.cpp", "aoti_torch_clone_preserve_strides(a, &b);")
            default = root / "default"
            default.mkdir()
            result = inspect_package(package, default, "aoti")
            self.assertEqual(len(result["unexpected_clone_copy_calls"]), 1)
            candidate = root / "candidate"
            candidate.mkdir()
            with self.assertRaisesRegex(ValueError, "unclassified or unconditional"):
                inspect_package(package, candidate, "megaqwen")


if __name__ == "__main__":
    unittest.main()
