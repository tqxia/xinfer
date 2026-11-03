"""Keep metadata hashes stable and input validation active under Python -O."""

import hashlib
import subprocess
import sys
import unittest

from compiler.common.artifacts import ROOT, canonical_json, json_sha256, validate_model_type


class ArtifactHelpers(unittest.TestCase):
    def test_model_tag_requires_a_matching_known_pair(self):
        for expected in ("qwen3", "qwen35"):
            validate_model_type({}, {}, expected)
            validate_model_type({"model_type": expected}, {"model_type": expected}, expected)
            with self.assertRaisesRegex(ValueError, "Package model types differ"):
                validate_model_type({"model_type": expected}, {}, expected)
            with self.assertRaisesRegex(ValueError, "Package model types differ"):
                validate_model_type({}, {"model_type": expected}, expected)
            for wrong in (None, "unknown", "qwen35" if expected == "qwen3" else "qwen3"):
                with self.subTest(expected=expected, wrong=wrong), \
                        self.assertRaisesRegex(ValueError, "Unsupported model type"):
                    validate_model_type({"model_type": wrong}, {"model_type": wrong}, expected)

    def test_metadata_hash_preserves_canonical_encoding(self):
        metadata = {"z": [None, True, -0.0, 1.25], "name": "紫霄",
                    "a": {"right": 2, "left": 1}}
        expected = ('{"a":{"left":1,"right":2},"name":"\\u7d2b\\u9704",'
                    '"z":[null,true,-0.0,1.25]}')
        self.assertEqual(canonical_json(metadata), expected)
        self.assertEqual(json_sha256(metadata), hashlib.sha256(expected.encode("utf-8")).hexdigest())
        reordered = {"a": {"left": 1, "right": 2}, "name": "紫霄", "z": metadata["z"]}
        self.assertEqual(json_sha256(metadata), json_sha256(reordered))

    def test_input_validation_survives_optimized_python(self):
        code = ("from compiler.common.artifacts import require\n"
                "assert False, 'this assertion is removed under -O'\n"
                "require(False, 'invalid archive checksum')\n")
        result = subprocess.run([sys.executable, "-O", "-c", code], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ValueError: invalid archive checksum", result.stderr)


if __name__ == "__main__":
    unittest.main()
