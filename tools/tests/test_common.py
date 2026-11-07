"""Shared prompt partitioning preserves every token and singleton remainder."""

import unittest

from tools.common import prompt_chunks


class PromptChunks(unittest.TestCase):
    def test_request_boundaries_and_decode_remainder(self):
        for length, sizes in (
            (1, [1]),
            (511, [511]),
            (512, [512]),
            (513, [512, 1]),
            (8191, [512] * 15 + [511]),
            (8192, [512] * 16),
            (8193, [512] * 16 + [1]),
        ):
            with self.subTest(length=length):
                tokens = list(range(length))
                chunks = prompt_chunks(tokens, 512)
                self.assertEqual([len(chunk) for chunk in chunks], sizes)
                self.assertEqual([token for chunk in chunks for token in chunk], tokens)
                self.assertEqual(tokens, list(range(length)))

    def test_empty_prompt_and_invalid_call_bound(self):
        for tokens, bound in (([], 512), ([1], 0), ([1], 1), ([1], True), ([1], 2.5)):
            with self.subTest(tokens=tokens, bound=bound), self.assertRaises(ValueError):
                prompt_chunks(tokens, bound)


if __name__ == "__main__":
    unittest.main()
