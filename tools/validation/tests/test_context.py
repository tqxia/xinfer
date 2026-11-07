"""Long requests retain matched partitions and bounded state checkpoints."""

import unittest

from tools.validation.common.context import capture_long_state, long_context_cases, state_shape


class ContextFixtures(unittest.TestCase):
    def test_long_matrix_preserves_tokens_and_per_call_bounds(self):
        cases = long_context_cases([3, 4, 5], 16384, 512)
        by_name = {case["name"]: case for case in cases}
        for length in (1, 511, 512, 513, 8191, 8192):
            case = by_name[f"long_prompt_{length}"]
            chunks = case["steps"][:-2]
            self.assertEqual(sum(map(len, chunks)), length)
            tokens = [token for chunk in case["steps"] for token in chunk]
            self.assertEqual(tokens, ([3, 4, 5] * ((length + 2) // 3 + 1))[: length + 2])
            self.assertTrue(all(1 <= len(chunk) <= 512 for chunk in chunks))
            checkpoints = [
                step for step in range(len(case["steps"])) if capture_long_state(case, step)
            ]
            self.assertEqual(checkpoints, [len(chunks) - 1, len(case["steps"]) - 1])
        self.assertEqual(list(map(len, by_name["long_prompt_513"]["steps"])), [512, 1, 1, 1])
        end = by_name["long_context_end"]
        self.assertTrue(end["check_overflow"])
        self.assertEqual(sum(map(len, end["steps"])), 16384)
        self.assertEqual(len(end["steps"][-1]), 1)
        self.assertTrue(all(len(chunk) <= 512 for case in cases for chunk in case["steps"]))
        self.assertEqual(by_name["long_prompt_513_poison"]["same_as"], "long_prompt_513")
        self.assertEqual(by_name["long_prompt_513_repeat"]["same_as"], "long_prompt_513")

    def test_retained_small_profile_has_no_unsupported_long_cases(self):
        self.assertEqual(long_context_cases([3], 1024, 512), [])

    def test_only_kv_state_uses_populated_prefix(self):
        key = dict(shape=[1, 4, 16384, 256], update="inplace")
        recurrent = dict(shape=[1, 32, 128, 128], update="replace")
        self.assertEqual(state_shape(key, 513, True), [1, 4, 513, 256])
        self.assertEqual(state_shape(key, 513, False), key["shape"])
        self.assertEqual(state_shape(recurrent, 513, True), recurrent["shape"])
        self.assertEqual(key["shape"], [1, 4, 16384, 256])


if __name__ == "__main__":
    unittest.main()
