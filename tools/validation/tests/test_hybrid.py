"""CPU-only checks that the hybrid validator cannot accept silent AOTI fallback."""

import os
import unittest
from unittest.mock import patch

from tools.validation.hybrid import check_dispatch, dispatch_counts, extra_cases, native_environment


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
    def test_added_fixtures_cover_handoff_and_poison(self):
        original = [dict(name="retained", steps=[[42]])]
        result = extra_cases(original, [42, 13])
        self.assertEqual(original, [dict(name="retained", steps=[[42]])])
        self.assertEqual(len(result), 4)
        prompt, handoff, poison = result[1:]
        self.assertEqual([len(t) for t in prompt["steps"]], [25, 1, 1, 1])
        self.assertEqual([len(t) for t in handoff["steps"]], [2, 1, 13, 1, 2, 1])
        self.assertEqual(poison["steps"], handoff["steps"])
        self.assertEqual(poison["same_as"], handoff["name"])
        self.assertTrue(all(case["dump_cache"] for case in result[1:]))
        self.assertEqual(len({case["name"] for case in result}), len(result))

    def test_fixture_collisions_and_empty_seed_fail(self):
        for name in ("hybrid_prompt_25", "hybrid_back_to_aoti", "hybrid_back_to_aoti_poison"):
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
