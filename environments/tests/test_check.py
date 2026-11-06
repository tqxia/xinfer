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


PROFILE = dict(system="Linux", machine="x86_64", sm=[8, 9])
NATIVE_REQUIREMENTS = dict(torch="2.13.0+cu126", triton="3.7.1", cmake="4.3.4", ninja="1.13.2")
MODEL_REQUIREMENTS = {"transformers": "5.20.2"}
REQUIREMENTS = MODEL_REQUIREMENTS | NATIVE_REQUIREMENTS


def observed():
    return dict(
        platform=dict(system="Linux", machine="x86_64"),
        python=dict(version="3.12.13", executable="/env/bin/python"),
        torch=dict(
            runtime_version="2.13.0+cu126",
            distribution_version="2.13.0",
            cuda="12.6",
            torch_dir="/env/torch/share/cmake/Torch",
        ),
        packages=REQUIREMENTS | {"torch": "2.13.0"},
        gpu=dict(name="NVIDIA L20", capability=[8, 9]),
    )


class EnvironmentContract(unittest.TestCase):
    def test_requirement_reader_reads_flat_pins_and_ignores_pip_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qwen35_requirements.txt"
            path.write_text(
                "# Complete model environment\n"
                "--extra-index-url https://download.pytorch.org/whl/cu130\n\n"
                "torch==2.13.0+cu130\n"
                "triton==3.7.1 # Compiler\n"
                "cmake==4.3.4\n"
                "ninja==1.13.2\n"
                "transformers==5.20.2\n"
            )
            self.assertEqual(
                environment.read_requirements(path),
                REQUIREMENTS | {"torch": "2.13.0+cu130"},
            )

    def test_each_device_model_file_contains_complete_dependencies(self):
        packages = {
            "torch",
            "triton",
            "cmake",
            "ninja",
            "transformers",
            "tokenizers",
            "safetensors",
            "huggingface-hub",
        }
        for target in ("l20", "gb10"):
            for model in ("qwen3", "qwen35"):
                with self.subTest(target=target, model=model):
                    requirements = environment.model_requirements(target, model)
                    self.assertTrue(packages <= requirements.keys())

    def test_collection_records_all_requested_native_and_model_packages(self):
        with (
            patch.object(environment, "package_version", side_effect=REQUIREMENTS.get) as versions,
            patch.object(
                environment.importlib.util, "find_spec", wraps=environment.importlib.util.find_spec
            ) as modules,
            patch.object(environment, "native_tool", return_value={}),
        ):
            actual = environment.collect("qwen3", REQUIREMENTS)
        self.assertEqual(actual["packages"], REQUIREMENTS)
        self.assertTrue(set(REQUIREMENTS) <= {call.args[0] for call in versions.call_args_list})
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
            collected = environment.collect("qwen35", REQUIREMENTS)
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
        actual["torch"].update(runtime_version="2.13.0+cu128", cuda="12.8")
        actual["packages"]["transformers"] = None
        issues = environment.validate(actual, PROFILE, REQUIREMENTS)
        self.assertEqual(len(issues), 3)
        self.assertTrue(any("Torch runtime" in s for s in issues))
        self.assertTrue(any("Torch CUDA build" in s for s in issues))
        self.assertTrue(any("transformers" in s for s in issues))

    def test_all_native_package_pins_are_checked(self):
        actual = observed()
        actual["packages"].update(triton="3.7.0", cmake=None, ninja="1.12.0")
        issues = environment.validate(actual, PROFILE, REQUIREMENTS)
        self.assertEqual(len(issues), 3)
        for name in ("triton", "cmake", "ninja"):
            self.assertTrue(any(issue.startswith(name + ":") for issue in issues))

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

    def test_normalized_cpu_names_and_gb10_requirements(self):
        actual = observed()
        actual["platform"]["machine"] = "AMD64"
        self.assertEqual(environment.validate(actual, PROFILE, REQUIREMENTS), [])
        profile = dict(PROFILE, machine="aarch64", sm=[12, 1])
        requirements = REQUIREMENTS | {"torch": "2.13.0+cu130"}
        actual["platform"]["machine"] = "arm64"
        actual["torch"].update(runtime_version="2.13.0+cu130", cuda="13.0")
        actual["gpu"] = dict(name="GB10", capability=[12, 1])
        self.assertEqual(environment.validate(actual, profile, requirements), [])

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

    def write_requirements(self, root, model="qwen35"):
        device = root / "environments/l20"
        device.mkdir(parents=True, exist_ok=True)
        (device / "profile.json").write_text(json.dumps(PROFILE))
        requirements = device / f"{model}_requirements.txt"
        requirements.write_text(
            "\n".join(f"{name}=={version}" for name, version in REQUIREMENTS.items())
        )
        return requirements

    def test_cli_reads_pins_through_the_same_model_entrypoint_as_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requirements = self.write_requirements(root)
            with (
                patch.object(environment, "ROOT", root),
                patch.object(environment, "collect", return_value=observed()) as collect,
                redirect_stdout(io.StringIO()) as stdout,
            ):
                self.assertEqual(environment.main(["--target", "l20", "--model", "qwen35"]), 0)
                first = json.loads(stdout.getvalue())
                self.assertEqual(first["requirements"], REQUIREMENTS)
                self.assertEqual(first["expected"], PROFILE)
                collect.assert_called_once_with("qwen35", REQUIREMENTS, None)
                requirements.write_text(
                    requirements.read_text().replace("cmake==4.3.4", "cmake==4.4.0")
                )
                stdout.seek(0)
                stdout.truncate(0)
                self.assertEqual(environment.main(["--target", "l20", "--model", "qwen35"]), 1)
                second = json.loads(stdout.getvalue())
                self.assertEqual(second["requirements"]["cmake"], "4.4.0")
                self.assertEqual(len(second["issues"]), 1)
                self.assertTrue(second["issues"][0].startswith("cmake:"))

    def test_models_on_one_device_can_select_different_torch_and_cuda_builds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qwen3 = self.write_requirements(root, "qwen3")
            self.write_requirements(root, "qwen35")
            qwen3.write_text(
                qwen3.read_text().replace("torch==2.13.0+cu126", "torch==2.13.0+cu130")
            )
            with (
                patch.object(environment, "ROOT", root),
                patch.object(environment, "collect", return_value=observed()) as collect,
                redirect_stdout(io.StringIO()) as stdout,
            ):
                self.assertEqual(environment.main(["--target", "l20", "--model", "qwen3"]), 1)
                changed = json.loads(stdout.getvalue())
                self.assertEqual(changed["requirements"]["torch"], "2.13.0+cu130")
                self.assertEqual(len(changed["issues"]), 2)
                self.assertTrue(any("Torch runtime" in issue for issue in changed["issues"]))
                self.assertTrue(any("Torch CUDA build" in issue for issue in changed["issues"]))
                stdout.seek(0)
                stdout.truncate(0)
                self.assertEqual(environment.main(["--target", "l20", "--model", "qwen35"]), 0)
                unchanged = json.loads(stdout.getvalue())
                self.assertEqual(unchanged["requirements"]["torch"], "2.13.0+cu126")
                collect.return_value["torch"].update(runtime_version="2.13.0+cu130", cuda="13.0")
                stdout.seek(0)
                stdout.truncate(0)
                self.assertEqual(environment.main(["--target", "l20", "--model", "qwen3"]), 0)

    def test_failed_cli_still_writes_machine_readable_report(self):
        actual = observed()
        actual["torch"] = dict(error="ModuleNotFoundError: No module named torch")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_requirements(root)
            output = root / "nested/report.json"
            with (
                patch.object(environment, "ROOT", root),
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
