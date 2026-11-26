"""Native CMake build wrapper resolves Torch, CUDA runtime and offline flags."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ENVIRONMENTS = Path(__file__).resolve().parents[1]

PYTHON_STUB = """#!/usr/bin/env bash
exec "$STUB_REAL_PYTHON" "$STUB_PY_DRIVER" "$@"
"""

PY_DRIVER = """import os
import sys

args = sys.argv[1:]
if len(args) >= 2 and args[0] == "-c":
    code = args[1]
    if "cmake_prefix_path" in code:
        print(os.environ["STUB_TORCH_PREFIX"])
    elif "get_device_capability" in code:
        print(os.environ.get("STUB_LIVE_SM", ""))
    elif "libcudart.so.12" in code:
        print(os.environ["STUB_CUDART"])
    else:
        raise SystemExit(f"Unexpected python -c: {code}")
else:
    raise SystemExit(f"Unexpected python args: {args}")
"""

CMAKE_STUB = """#!/usr/bin/env bash
exec "$STUB_REAL_PYTHON" -c '
import json, os, sys
from pathlib import Path
with Path(os.environ["STUB_CMAKE_RECORD"]).open("a") as handle:
    handle.write(json.dumps(dict(cwd=os.getcwd(), args=sys.argv[1:])) + "\\n")
' "$@"
"""


class NativeBuildWrapper(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "project with spaces"
        self.scripts = self.root / "environments"
        self.scripts.mkdir(parents=True)
        shutil.copy2(ENVIRONMENTS / "build.sh", self.scripts / "build.sh")
        self.outside = self.base / "caller"
        self.outside.mkdir()
        self.record = self.base / "cmake.jsonl"
        self.driver = self.base / "py_driver.py"
        self.driver.write_text(PY_DRIVER)
        self.venv_bin = self.root / "work/l20/venv/bin"
        self.venv_bin.mkdir(parents=True)
        for name, body in (("python", PYTHON_STUB), ("cmake", CMAKE_STUB)):
            path = self.venv_bin / name
            path.write_text(body)
            path.chmod(0o755)
        self.gb10_bin = self.root / "work/gb10/venv/bin"
        self.gb10_bin.mkdir(parents=True)
        for name, body in (("python", PYTHON_STUB), ("cmake", CMAKE_STUB)):
            path = self.gb10_bin / name
            path.write_text(body)
            path.chmod(0o755)
        self.env = dict(
            os.environ,
            STUB_REAL_PYTHON=sys.executable,
            STUB_PY_DRIVER=str(self.driver),
            STUB_CMAKE_RECORD=str(self.record),
            STUB_TORCH_PREFIX="/opt/torch/share/cmake",
            STUB_CUDART="/opt/torch/nvidia/cuda_runtime/lib/libcudart.so.12",
        )
        self.env.pop("TORCH_CUDA_ARCH_LIST", None)
        self.env.pop("FETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON", None)

    def run_build(self, *arguments, env=None):
        return subprocess.run(
            ["bash", str(self.scripts / "build.sh"), *arguments],
            cwd=self.outside,
            env=self.env | (env or {}),
            capture_output=True,
            text=True,
        )

    def calls(self):
        if not self.record.exists():
            return []
        return [json.loads(line) for line in self.record.read_text().splitlines()]

    def test_l20_build_passes_cuda_runtime_and_default_options(self):
        result = self.run_build("--target", "l20", "--build", "build/custom")
        self.assertEqual(result.returncode, 0, result.stderr)
        configure, build = self.calls()
        self.assertEqual(Path(configure["cwd"]).resolve(), self.root.resolve())
        self.assertIn("-G", configure["args"])
        self.assertIn("Ninja", configure["args"])
        self.assertIn("-DCMAKE_PREFIX_PATH=/opt/torch/share/cmake", configure["args"])
        self.assertIn("-DTORCH_CUDA_ARCH_LIST=8.9", configure["args"])
        self.assertIn(
            "-DCUDA_cudart_LIBRARY=/opt/torch/nvidia/cuda_runtime/lib/libcudart.so.12",
            configure["args"],
        )
        self.assertIn(
            "-DCUDA_CUDART_LIBRARY=/opt/torch/nvidia/cuda_runtime/lib/libcudart.so.12",
            configure["args"],
        )
        self.assertIn("-DXINFER_BUILD_VALIDATION=ON", configure["args"])
        self.assertIn("-DXINFER_BUILD_BENCHMARK=ON", configure["args"])
        self.assertEqual(build["args"], ["--build", str(self.root / "build/custom"), "-j", "2"])

    def test_gb10_omits_l20_cudart_and_forwards_offline_and_custom_flags(self):
        result = self.run_build(
            "--target",
            "gb10",
            "--jobs",
            "4",
            "--no-validation",
            "--no-benchmark",
            "--targets",
            "xinfer_attention_ops,xinfer_aoti",
            env={"FETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON": "/opt/deps/nlohmann_json"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        configure, build = self.calls()
        self.assertIn("-DTORCH_CUDA_ARCH_LIST=12.1", configure["args"])
        self.assertFalse(any("CUDA_cudart_LIBRARY" in arg for arg in configure["args"]))
        self.assertIn(
            "-DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=/opt/deps/nlohmann_json",
            configure["args"],
        )
        self.assertIn("-DFETCHCONTENT_FULLY_DISCONNECTED=ON", configure["args"])
        self.assertIn("-DXINFER_BUILD_VALIDATION=OFF", configure["args"])
        self.assertIn("-DXINFER_BUILD_BENCHMARK=OFF", configure["args"])
        self.assertEqual(
            build["args"],
            [
                "--build",
                str(self.root / "build"),
                "-j",
                "4",
                "--target",
                "xinfer_attention_ops",
                "xinfer_aoti",
            ],
        )

    def test_usage_errors_fail_before_invoking_cmake(self):
        help_result = self.run_build("--help")
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("environments/build.sh", help_result.stdout)
        for args in (
            ("--target", "invalid"),
            ("--jobs", "0"),
            ("--unknown",),
            ("--target", "l20", "--python", "missing/python"),
        ):
            with self.subTest(args=args):
                result = self.run_build(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
