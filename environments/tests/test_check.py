"""Environment contracts can run without a Torch installation or GPU."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from environments import check as environment


PROFILE = dict(
    system="Linux",
    machine="x86_64",
    python="3.12",
    torch_runtime="2.7.0+cu126",
    cuda="12.6",
    triton="3.3.0",
    sm=[8, 9],
)
REQUIREMENTS = {"transformers": "5.20.2"}


def observed():
    return dict(
        platform=dict(system="Linux", machine="x86_64"),
        python=dict(version="3.12.13", executable="/env/bin/python"),
        torch=dict(
            runtime_version="2.7.0+cu126",
            distribution_version="2.7.0",
            cuda="12.6",
            torch_dir="/env/torch/share/cmake/Torch",
        ),
        triton="3.3.0",
        packages=dict(REQUIREMENTS),
        gpu=dict(name="NVIDIA L20", capability=[8, 9]),
    )


class EnvironmentContract(unittest.TestCase):
    def test_native_only_collection_does_not_inspect_model_dependencies(self):
        with (
            patch.object(
                environment,
                "model_requirements",
                side_effect=AssertionError("Unexpected model requirements"),
            ),
            patch.object(
                environment.importlib.util, "find_spec", wraps=environment.importlib.util.find_spec
            ) as modules,
            patch.object(environment, "native_tool", return_value={}),
        ):
            actual = environment.collect(None)
        self.assertEqual(actual["packages"], {})
        self.assertNotIn("optional_backends", actual)
        self.assertFalse(
            any(call.args[0] in ("fla", "causal_conv1d") for call in modules.call_args_list)
        )

    def test_qwen35_detects_forbidden_backends_without_importing_them(self):
        find_spec = environment.importlib.util.find_spec

        def available(name, *args, **kwargs):
            return (
                object() if name in ("fla", "causal_conv1d") else find_spec(name, *args, **kwargs)
            )

        with (
            patch.object(environment.importlib.util, "find_spec", side_effect=available),
            patch.object(environment, "native_tool", return_value={}),
        ):
            collected = environment.collect("qwen35")
        self.assertEqual(collected["optional_backends"], dict(fla=True, causal_conv1d=True))
        actual = observed()
        actual["optional_backends"] = collected["optional_backends"]
        self.assertEqual(
            environment.validate(actual, PROFILE, REQUIREMENTS),
            [
                "Qwen3.5 requires an environment without fla",
                "Qwen3.5 requires an environment without causal_conv1d",
            ],
        )

    def test_matching_stack_and_distribution_version_without_cuda_suffix(self):
        self.assertEqual(environment.validate(observed(), PROFILE, REQUIREMENTS), [])

    def test_same_release_different_cuda_and_missing_dependency_are_rejected(self):
        actual = observed()
        actual["torch"].update(runtime_version="2.7.0+cu128", cuda="12.8")
        actual["packages"]["transformers"] = None
        issues = environment.validate(actual, PROFILE, REQUIREMENTS)
        self.assertEqual(len(issues), 3)
        self.assertTrue(any("Torch runtime" in s for s in issues))
        self.assertTrue(any("Torch CUDA build" in s for s in issues))
        self.assertTrue(any("transformers" in s for s in issues))

    def test_gpu_mismatch_and_no_cuda(self):
        actual = observed()
        actual["gpu"]["capability"] = [12, 1]
        self.assertTrue(
            any("GPU capability" in s for s in environment.validate(actual, PROFILE, REQUIREMENTS))
        )
        actual["gpu"] = dict(error="CUDA device 0 is unavailable")
        self.assertEqual(
            environment.validate(actual, PROFILE, REQUIREMENTS), ["CUDA device 0 is unavailable"]
        )

    def test_normalized_cpu_names_and_gb10_profile(self):
        actual = observed()
        actual["platform"]["machine"] = "AMD64"
        self.assertEqual(environment.validate(actual, PROFILE, REQUIREMENTS), [])
        profile = dict(
            PROFILE,
            machine="aarch64",
            torch_runtime="2.13.0+cu130",
            cuda="13.0",
            triton="3.7.1",
            sm=[12, 1],
        )
        actual["platform"]["machine"] = "arm64"
        actual["torch"].update(runtime_version="2.13.0+cu130", cuda="13.0")
        actual.update(triton="3.7.1", gpu=dict(name="GB10", capability=[12, 1]))
        self.assertEqual(environment.validate(actual, profile, REQUIREMENTS), [])

    def test_build_torch_dir_resolves_symlinks_and_rejects_a_different_stack(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "torch/share/cmake/Torch"
            native.mkdir(parents=True)
            alias = root / "linked-torch"
            alias.symlink_to(root / "torch", target_is_directory=True)
            cache = root / "build"
            cache.mkdir()
            (cache / "CMakeCache.txt").write_text(f"Torch_DIR:PATH={alias}/share/cmake/Torch\n")
            actual = observed()
            actual["torch"]["torch_dir"] = str(native)
            actual["build"] = environment.configured_build(cache)
            self.assertEqual(environment.validate(actual, PROFILE, REQUIREMENTS), [])
            actual["build"]["torch_dir"] = str(root / "other/torch")
            self.assertTrue(
                any(
                    "Build Torch_DIR" in s
                    for s in environment.validate(actual, PROFILE, REQUIREMENTS)
                )
            )

    def test_missing_build_configuration_has_an_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            actual = observed()
            actual["build"] = environment.configured_build(Path(tmp))
            self.assertTrue(
                any("configure" in s for s in environment.validate(actual, PROFILE, REQUIREMENTS))
            )

    def test_failed_cli_still_writes_machine_readable_report(self):
        actual = observed()
        actual["torch"] = dict(error="ModuleNotFoundError: No module named torch")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "nested/report.json"
            with (
                patch.object(environment, "collect", return_value=copy.deepcopy(actual)),
                redirect_stdout(io.StringIO()) as stdout,
            ):
                code = environment.main(
                    ["--target", "l20", "--model", "qwen35", "--output", str(output)]
                )
            report = json.loads(output.read_text())
            self.assertEqual(code, 1)
            self.assertFalse(report["passed"])
            self.assertTrue(any("Torch cannot be imported" in s for s in report["issues"]))
            self.assertEqual(json.loads(stdout.getvalue()), report)


if __name__ == "__main__":
    unittest.main()
