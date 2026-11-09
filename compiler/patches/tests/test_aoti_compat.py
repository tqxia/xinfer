"""CPU checks for scoped include precedence, source guards and restoration."""

from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch._inductor.cpp_builder import BuildOptionsBase, CppBuilder, CppTorchDeviceOptions

from compiler.patches import aoti_compat


class AOTIHeaderOverlay(unittest.TestCase):
    def test_overlay_precedes_torch_only_for_aoti_and_cleans_up_after_failure(self):
        header = Path(torch.__file__).resolve().parent / "include" / aoti_compat.HEADER
        before = header.read_bytes()
        original = CppTorchDeviceOptions.get_include_dirs
        aot = SimpleNamespace(_aot_mode=True, _include_dirs=["torch-include", "cuda-include"])
        jit = SimpleNamespace(_aot_mode=False, _include_dirs=["torch-include"])
        with self.assertRaisesRegex(RuntimeError, "compile failed"):
            with aoti_compat.owned_constant_handles() as overlay:
                self.assertEqual(
                    CppTorchDeviceOptions.get_include_dirs(aot), [str(overlay), *aot._include_dirs]
                )
                self.assertEqual(CppTorchDeviceOptions.get_include_dirs(jit), jit._include_dirs)
                self.assertEqual(aot._include_dirs, ["torch-include", "cuda-include"])
                fixed = (overlay / aoti_compat.HEADER).read_text()
                self.assertEqual(
                    fixed, before.decode().replace(aoti_compat.ORIGINAL, aoti_compat.FIXED)
                )
                self.assertNotIn(aoti_compat.ORIGINAL, fixed)
                raise RuntimeError("compile failed")
        self.assertIs(CppTorchDeviceOptions.get_include_dirs, original)
        self.assertNotIn("get_include_dirs", CppTorchDeviceOptions.__dict__)
        self.assertFalse(overlay.exists())
        self.assertEqual(header.read_bytes(), before)

    def test_unsupported_build_and_header_fail_before_global_mutation(self):
        original = CppTorchDeviceOptions.get_include_dirs
        for guard, message in (
            (patch("torch.__version__", "2.14.0"), "pinned Torch"),
            (patch("torch.version.git_version", "unknown"), "pinned Torch"),
            (patch.object(aoti_compat, "HEADER_SHA256", "0" * 64), "ownership header"),
        ):
            with self.subTest(message=message), guard, self.assertRaisesRegex(ValueError, message):
                with aoti_compat.owned_constant_handles():
                    self.fail("Unsupported build entered the overlay context")
            self.assertIs(CppTorchDeviceOptions.get_include_dirs, original)
        self.assertIs(original, BuildOptionsBase.get_include_dirs)

    def test_overlay_changes_the_actual_aoti_cpp_cache_key_command(self):
        compiler = shutil.which("c++")
        self.assertIsNotNone(compiler)
        commands = []
        for _ in range(2):
            with aoti_compat.owned_constant_handles() as overlay:
                options = CppTorchDeviceOptions(
                    device_type="cpu", aot_mode=True, compile_only=True, compiler=compiler
                )
                command = CppBuilder(
                    name="cache-key", sources="fixture.cpp", BuildOption=options
                ).get_command_line()
                commands.append(command)
                self.assertIn(f"-I{overlay}", command)
                torch_include = Path(torch.__file__).resolve().parent / "include"
                self.assertLess(command.index(f"-I{overlay}"), command.index(f"-I{torch_include}"))
        self.assertNotEqual(commands[0], commands[1])


if __name__ == "__main__":
    unittest.main()
