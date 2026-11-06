"""CPU-only checks that the hybrid validator cannot accept silent AOTI fallback."""

import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.validation.common.native import check_dispatch, dispatch_counts, native_environment
from tools.validation.hybrid import _suite, extra_cases


class DispatchTests(unittest.TestCase):
    def test_one_token_prompt_is_aoti_then_native(self):
        self.assertEqual(
            dispatch_counts([1, 1, 1], "megaqwen"),
            dict(position=3, prefill_calls=0, decode_calls=3, megaqwen_calls=2),
        )
        self.assertEqual(dispatch_counts([1], "megaqwen")["megaqwen_calls"], 0)

    def test_return_to_prefill_does_not_disable_later_native_decode(self):
        self.assertEqual(
            dispatch_counts([2, 1, 13, 1], "megaqwen"),
            dict(position=17, prefill_calls=2, decode_calls=2, megaqwen_calls=2),
        )
        self.assertEqual(dispatch_counts([2, 1, 13, 1], "aoti")["megaqwen_calls"], 0)

    def test_long_prompt_chunk_tail_uses_the_selected_decode_backend(self):
        self.assertEqual(
            dispatch_counts([512, 512, 1, 1], "megaqwen"),
            dict(position=1026, prefill_calls=2, decode_calls=2, megaqwen_calls=2),
        )
        self.assertEqual(dispatch_counts([4, 1], "megaqwen", max_prefill=4)["position"], 5)
        with self.assertRaises(ValueError):
            dispatch_counts([5, 1], "megaqwen", max_prefill=4)

    def test_silent_fallback_and_missing_counters_fail(self):
        row = dispatch_counts([25, 1], "aoti")
        with self.assertRaisesRegex(ValueError, "megaqwen_calls"):
            check_dispatch(row, [25, 1], "megaqwen")
        row.pop("megaqwen_calls")
        with self.assertRaisesRegex(ValueError, "megaqwen_calls"):
            check_dispatch(row, [25, 1], "aoti")

    def test_invalid_partitions_and_backend_fail(self):
        for lengths in ([], [0], [513], [True], [1.0]):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                dispatch_counts(lengths, "megaqwen")
        with self.assertRaises(ValueError):
            dispatch_counts([1], "unknown")


class FixtureTests(unittest.TestCase):
    def test_hybrid_suite_inherits_live_references_and_long_context_hooks(self):
        runtime = object()
        metadata = {"capacity": 40960}
        main_cases = [
            dict(name="capital_teacher", fixture="capital", steps=[[42, 13], [0]]),
            dict(name="long_prompt_513", steps=[[42] * 512, [13], [0]], long_context=True),
        ]
        chats = [dict(name="long_prompt_513_generate", prompt=[42] * 513, limit=3)]

        class MainSuite:
            def __init__(self, model_dir, pair):
                self.model_dir, self.pair = model_dir, pair
                self.capacity = pair["capacity"]

            def fixtures(self, actual_runtime):
                self.runtime = actual_runtime
                return main_cases, chats

            def capture_states(self, case, step):
                return case.get("long_context") and step == 2

        with patch.dict(
            sys.modules, {"tools.validation.qwen3.model": SimpleNamespace(Suite=MainSuite)}
        ):
            suite = _suite("checkpoint", metadata)
        cases, actual_chats = suite.fixtures(runtime)
        self.assertIs(suite.runtime, runtime)
        self.assertEqual(suite.capacity, 40960)
        self.assertEqual(cases[:2], main_cases)
        self.assertIs(actual_chats, chats)
        self.assertTrue(suite.capture_states(cases[1], 2))
        self.assertEqual([len(tokens) for tokens in cases[1]["steps"]], [512, 1, 1])
        self.assertEqual(len(cases), len(main_cases) + 7)

    def test_added_fixtures_cover_handoff_and_poison(self):
        original = [dict(name="retained", steps=[[42]])]
        result = extra_cases(original, [42, 13])
        self.assertEqual(original, [dict(name="retained", steps=[[42]])])
        self.assertEqual(len(result), 8)
        prompt, handoff, poison = result[1:4]
        self.assertEqual([len(t) for t in prompt["steps"]], [25, 1, 1, 1])
        self.assertEqual([len(t) for t in handoff["steps"]], [2, 1, 13, 1, 2, 1])
        self.assertEqual(poison["steps"], handoff["steps"])
        self.assertEqual(poison["same_as"], handoff["name"])
        self.assertTrue(all(case["dump_cache"] for case in result[1:]))
        self.assertEqual(len({case["name"] for case in result}), len(result))

    def test_split_transitions_return_to_aoti_and_preserve_poison_pairs(self):
        cases = extra_cases([], [42, 13])
        for boundary in (128, 256):
            with self.subTest(boundary=boundary):
                clean = next(case for case in cases if case["name"] == f"hybrid_split_{boundary}")
                poison = next(case for case in cases if case.get("same_as") == clean["name"])
                lengths = [len(tokens) for tokens in clean["steps"]]
                self.assertEqual(lengths, [boundary - 1, 1, 1, 2, 1])
                self.assertEqual(poison["steps"], clean["steps"])
                self.assertEqual(poison["poison_kv"], 3)
                self.assertEqual(dispatch_counts(lengths, "megaqwen")["megaqwen_calls"], 3)

    def test_fixture_collisions_and_empty_seed_fail(self):
        for name in (
            "hybrid_prompt_25",
            "hybrid_back_to_aoti",
            "hybrid_back_to_aoti_poison",
            "hybrid_split_128",
            "hybrid_split_128_poison",
            "hybrid_split_256",
            "hybrid_split_256_poison",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                extra_cases([dict(name=name)], [42])
        with self.assertRaises(ValueError):
            extra_cases([], [])

    def test_native_environment_retains_device_selection_but_excludes_python(self):
        environment = dict(
            CUDA_VISIBLE_DEVICES="GPU-test",
            MEGAQWEN_NUM_BLOCKS="92",
            PATH="/custom/bin",
            PYTHONHOME="/python",
            PYTHONPATH="/modules",
            LD_LIBRARY_PATH="/unsafe/lib",
            VIRTUAL_ENV="/venv",
        )
        with patch.dict(os.environ, environment, clear=True):
            actual = native_environment()
        self.assertEqual(actual["CUDA_VISIBLE_DEVICES"], "GPU-test")
        self.assertEqual(actual["MEGAQWEN_NUM_BLOCKS"], "92")
        self.assertEqual(actual["PYTHONHOME"], "/nonexistent-python-home")
        self.assertEqual(actual["PATH"], "/usr/bin:/bin")
        self.assertNotIn("PYTHONPATH", actual)
        self.assertNotIn("LD_LIBRARY_PATH", actual)
        self.assertNotIn("VIRTUAL_ENV", actual)


if __name__ == "__main__":
    unittest.main()
