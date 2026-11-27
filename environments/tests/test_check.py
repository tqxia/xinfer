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


PROFILE = dict(system="Linux", machine="x86_64", python="3.12", sm=[8, 9])
NATIVE_REQUIREMENTS = dict(torch="2.13.0+cu126", triton="3.7.1", cmake="4.3.4", ninja="1.13.2")
MODEL_REQUIREMENTS = {"transformers": "5.20.2", "kernels": "0.17.2"}
REQUIREMENTS = MODEL_REQUIREMENTS | NATIVE_REQUIREMENTS


def observed():
    return dict(
        platform=dict(system="Linux", machine="x86_64"),
        python=dict(version="3.12.13", executable="/env/bin/python"),
        native_tools={
            "c++": dict(path="/usr/bin/c++", version="g++ 11.2", exit_code=0),
            "nvcc": dict(path="/usr/local/cuda/bin/nvcc", version="nvcc 12.6", exit_code=0),
            "cargo": dict(path="/opt/rust/cargo/bin/cargo", version="cargo 1.88.0", exit_code=0),
            "rustc": dict(path="/opt/rust/cargo/bin/rustc", version="rustc 1.88.0", exit_code=0),
        },
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
    def test_requirement_reader_resolves_includes_and_rejects_cycles_and_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = root / "requirements.txt"
            shared.write_text(
                "# Shared pins\n"
                "triton==3.7.1 # Compiler\n"
                "cmake==4.3.4\n"
                "ninja==1.13.2\n"
                "transformers==5.20.2\n"
                "kernels==0.17.2\n"
            )
            target = root / "gb10" / "requirements.txt"
            target.parent.mkdir()
            target.write_text(
                "--extra-index-url https://download.pytorch.org/whl/cu130\n"
                "torch==2.13.0+cu130\n"
                "-r ../requirements.txt\n"
            )
            self.assertEqual(
                environment.read_requirements(target),
                REQUIREMENTS | {"torch": "2.13.0+cu130"},
            )
            # Conflict detection.
            conflict = root / "gb10" / "conflict.txt"
            conflict.write_text("triton==3.7.0\n-r ../requirements.txt\n")
            with self.assertRaisesRegex(ValueError, "Conflicting pin"):
                environment.read_requirements(conflict)
            # Cycle detection.
            cycle_a = root / "a.txt"
            cycle_b = root / "b.txt"
            cycle_a.write_text("-r b.txt\n")
            cycle_b.write_text("-r a.txt\n")
            with self.assertRaisesRegex(ValueError, "Circular requirement"):
                environment.read_requirements(cycle_a)

    def test_each_target_file_contains_complete_aligned_dependencies(self):
        expected = {
            "torch",
            "triton",
            "cmake",
            "ninja",
            "transformers",
            "tokenizers",
            "safetensors",
            "huggingface-hub",
            "kernels",
        }
        targets = {
            name: environment.target_requirements(name) for name in ("l20", "pro5000", "gb10")
        }
        for target, requirements in targets.items():
            with self.subTest(target=target):
                self.assertTrue(expected <= requirements.keys())
        shared_l20 = {k: v for k, v in targets["l20"].items() if k != "torch"}
        for target in ("pro5000", "gb10"):
            self.assertEqual(
                shared_l20,
                {k: v for k, v in targets[target].items() if k != "torch"},
            )
            self.assertEqual(targets[target]["torch"], "2.13.0+cu130")
        dev = environment.target_requirements("l20", include_dev=True)
        self.assertTrue({"ruff", "clang-format"} <= dev.keys())

    def test_collection_records_all_requested_native_and_model_packages(self):
        with (
            patch.object(environment, "package_version", side_effect=REQUIREMENTS.get) as versions,
            patch.object(environment, "native_tool", return_value={}),
        ):
            actual = environment.collect(REQUIREMENTS)
        self.assertEqual(actual["packages"], REQUIREMENTS)
        self.assertTrue(set(REQUIREMENTS) <= {call.args[0] for call in versions.call_args_list})

    def test_matching_stack_and_distribution_version_without_cuda_suffix(self):
        self.assertEqual(environment.validate(observed(), PROFILE, REQUIREMENTS), [])

    def test_python_version_and_native_tools_are_validated(self):
        actual = observed()
        actual["python"]["version"] = "3.11.9"
        actual["native_tools"]["c++"] = dict(path=None)
        actual["native_tools"]["nvcc"] = dict(path="/usr/local/cuda/bin/nvcc", exit_code=127)
        issues = environment.validate(actual, PROFILE, REQUIREMENTS)
        self.assertEqual(len(issues), 3)
        self.assertTrue(any("Python version" in s for s in issues))
        self.assertTrue(any("'c++' is missing" in s for s in issues))
        self.assertTrue(any("'nvcc' exited with 127" in s for s in issues))

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

    def test_normalized_cpu_names_and_blackwell_targets(self):
        actual = observed()
        actual["platform"]["machine"] = "AMD64"
        self.assertEqual(environment.validate(actual, PROFILE, REQUIREMENTS), [])
        cu130 = REQUIREMENTS | {"torch": "2.13.0+cu130"}
        pro5000_profile = dict(PROFILE, machine="x86_64", sm=[12, 0])
        actual["platform"]["machine"] = "x86_64"
        actual["torch"].update(runtime_version="2.13.0+cu130", cuda="13.0")
        actual["gpu"] = dict(name="NVIDIA RTX PRO 5000 72GB Blackwell", capability=[12, 0])
        self.assertEqual(environment.validate(actual, pro5000_profile, cu130), [])
        gb10_profile = dict(PROFILE, machine="aarch64", sm=[12, 1])
        actual["platform"]["machine"] = "arm64"
        actual["gpu"] = dict(name="GB10", capability=[12, 1])
        self.assertEqual(environment.validate(actual, gb10_profile, cu130), [])
        self.assertEqual(environment.default_target("AMD64", sm=[8, 9]), "l20")
        self.assertEqual(environment.default_target("x86_64", sm=[12, 0]), "pro5000")
        with patch.object(environment, "detect_gpu_sm", return_value=None):
            self.assertIsNone(environment.default_target("x86_64"))
        self.assertEqual(environment.default_target("arm64"), "gb10")
        self.assertIsNone(environment.default_target("riscv64"))

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

    def write_requirements(self, root, target="l20"):
        device = root / "environments" / target
        device.mkdir(parents=True, exist_ok=True)
        shared = root / "environments/requirements.txt"
        shared.write_text(
            "\n".join(
                f"{name}=={version}" for name, version in REQUIREMENTS.items() if name != "torch"
            )
            + "\n"
        )
        requirements = device / "requirements.txt"
        requirements.write_text(f"torch=={REQUIREMENTS['torch']}\n-r ../requirements.txt\n")
        (root / "requirements-dev.txt").write_text("ruff==0.17.3\nclang-format==23.1.3\n")
        return shared

    def test_cli_reads_shared_pins_auto_detects_target_and_supports_dev_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = self.write_requirements(root, "l20")
            with (
                patch.object(environment, "ROOT", root),
                patch.object(environment, "default_target", return_value="l20"),
                patch.object(environment, "collect", return_value=observed()) as collect,
                redirect_stdout(io.StringIO()) as stdout,
            ):
                self.assertEqual(environment.main([]), 0)
                first = json.loads(stdout.getvalue())
                self.assertEqual(first["target"], "l20")
                self.assertEqual(first["requirements"], REQUIREMENTS)
                self.assertEqual(first["expected"], PROFILE)
                collect.assert_called_once_with(REQUIREMENTS, None)
                shared.write_text(shared.read_text().replace("cmake==4.3.4", "cmake==4.4.0"))
                stdout.seek(0)
                stdout.truncate(0)
                self.assertEqual(environment.main(["--target", "l20", "--dev"]), 1)
                second = json.loads(stdout.getvalue())
                self.assertEqual(second["requirements"]["cmake"], "4.4.0")
                self.assertEqual(second["requirements"]["ruff"], "0.17.3")
                self.assertTrue(any(issue.startswith("cmake:") for issue in second["issues"]))
                self.assertTrue(any(issue.startswith("ruff:") for issue in second["issues"]))

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
                code = environment.main(["--target", "l20", "--output", str(output)])
            report = json.loads(output.read_text())
            self.assertEqual(code, 1)
            self.assertFalse(report["passed"])
            self.assertTrue(any("Torch cannot be imported" in s for s in report["issues"]))
            self.assertEqual(json.loads(stdout.getvalue()), report)


if __name__ == "__main__":
    unittest.main()
