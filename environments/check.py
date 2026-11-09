"""Check a tested native stack and model environment before build or export."""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from environments import ROOT


def read_requirements(path):
    """Read exact package pins, leaving pip index options to the installer."""
    result = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name, separator, version = line.partition("==")
        if not separator or not name.strip() or not version.strip():
            raise ValueError(f"Expected an exact package pin in {path}: {line}")
        result[name.strip()] = version.strip()
    return result


def model_requirements(target, model):
    return read_requirements(ROOT / "environments" / target / f"{model}_requirements.txt")


def torch_cuda_version(version):
    """The native Torch pin owns the expected CUDA build, e.g. +cu126."""
    suffix = version.rpartition("+cu")[2]
    if len(suffix) < 2 or not suffix.isdecimal():
        raise ValueError(f"Native Torch pin requires a CUDA build suffix: {version}")
    return f"{int(suffix[:-1])}.{int(suffix[-1])}"


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def native_tool(name):
    path = shutil.which(name)
    if name == "nvcc" and not path:
        candidate = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc"
        path = str(candidate) if candidate.is_file() else None
    result = dict(path=path)
    if path:
        try:
            process = subprocess.run(
                [path, "--version"], capture_output=True, text=True, timeout=10
            )
            result.update(
                version=(process.stdout + process.stderr).strip(), exit_code=process.returncode
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            result["error"] = str(error)
    return result


def configured_build(directory):
    cache = directory.resolve() / "CMakeCache.txt"
    result = dict(directory=str(directory.resolve()), cache=str(cache))
    if not cache.is_file():
        return dict(result, error="Missing CMakeCache.txt; configure this build directory first")
    values = {}
    for line in cache.read_text().splitlines():
        if not line.startswith(("#", "//")) and ":" in line and "=" in line:
            key, value = line.split("=", 1)
            values[key.split(":", 1)[0]] = value
    result.update(
        torch_dir=values.get("Torch_DIR"),
        cxx_compiler=values.get("CMAKE_CXX_COMPILER"),
        cuda_compiler=values.get("CMAKE_CUDA_COMPILER"),
    )
    return result


def collect(model, requirements, build=None):
    actual = dict(
        python=dict(version=platform.python_version(), executable=sys.executable),
        platform=dict(system=platform.system(), machine=platform.machine()),
        packages={name: package_version(name) for name in requirements},
        native_tools={name: native_tool(name) for name in ("c++", "nvcc")},
    )
    if model in ("qwen35", "qwen36"):
        actual["optional_backends"] = {
            name: importlib.util.find_spec(name) is not None for name in ("fla", "causal_conv1d")
        }
    if build is not None:
        actual["build"] = configured_build(build)
    try:
        import torch

        torch_dir = Path(torch.__file__).resolve().parent / "share/cmake/Torch"
        actual["torch"] = dict(
            runtime_version=torch.__version__,
            distribution_version=package_version("torch"),
            git_version=torch.version.git_version,
            cuda=torch.version.cuda,
            cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            path=str(Path(torch.__file__).resolve()),
            torch_dir=str(torch_dir),
            cmake_prefix_path=torch.utils.cmake_prefix_path,
        )
        if torch.cuda.is_available():
            actual["gpu"] = dict(
                name=torch.cuda.get_device_name(0),
                capability=list(torch.cuda.get_device_capability(0)),
            )
        else:
            actual["gpu"] = dict(error="CUDA device 0 is unavailable")
    except Exception as error:
        actual.setdefault("torch", dict(error=f"{type(error).__name__}: {error}"))
        actual.setdefault("gpu", dict(error=f"Cannot query CUDA: {error}"))
    return actual


def normalized_machine(value):
    return {"amd64": "x86_64", "arm64": "aarch64"}.get(value.lower(), value.lower())


def validate(actual, profile, requirements):
    """Return actionable issues without importing Torch or initializing CUDA."""
    issues = []

    def expect(label, observed, expected):
        if observed != expected:
            issues.append(f"{label}: expected {expected!r}, found {observed!r}")

    expect("Operating system", actual["platform"]["system"], profile["system"])
    expect(
        "CPU architecture", normalized_machine(actual["platform"]["machine"]), profile["machine"]
    )
    for name, version in requirements.items():
        if name != "torch":
            expect(name, actual["packages"].get(name), version)
    for name, available in actual.get("optional_backends", {}).items():
        if available:
            issues.append(f"This model requires an environment without {name}")
    torch = actual["torch"]
    if "error" in torch:
        issues.append("Torch cannot be imported: " + torch["error"])
    else:
        expect("Torch runtime", torch["runtime_version"], requirements["torch"])
        expect("Torch CUDA build", torch["cuda"], torch_cuda_version(requirements["torch"]))
        if "error" in actual["gpu"]:
            issues.append(actual["gpu"]["error"])
        else:
            expect("GPU capability", actual["gpu"]["capability"], profile["sm"])
    if "build" in actual:
        build = actual["build"]
        if "error" in build:
            issues.append(build["error"] + ": " + build["directory"])
        elif not build["torch_dir"]:
            issues.append(
                "Build has no Torch_DIR; configure with this Python's torch.utils.cmake_prefix_path"
            )
        elif "error" not in torch:
            expect(
                "Build Torch_DIR",
                str(Path(build["torch_dir"]).resolve()),
                str(Path(torch["torch_dir"]).resolve()),
            )
    return issues


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("l20", "gb10"), required=True)
    parser.add_argument("--model", choices=("qwen3", "qwen35", "qwen36"), required=True)
    parser.add_argument(
        "--build", type=Path, help="Also check this configured CMake build's Torch_DIR"
    )
    parser.add_argument(
        "--output", type=Path, help="Save the same JSON printed to stdout, including failures"
    )
    args = parser.parse_args(argv)
    if args.model == "qwen36" and args.target != "gb10":
        parser.error("Qwen3.6 environment checking currently supports only GB10")
    profile = json.loads((ROOT / "environments" / args.target / "profile.json").read_text())
    requirements = model_requirements(args.target, args.model)
    actual = collect(args.model, requirements, args.build)
    issues = validate(actual, profile, requirements)
    report = dict(
        target=args.target,
        model=args.model,
        expected=profile,
        requirements=requirements,
        actual=actual,
        issues=issues,
        passed=not issues,
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return int(bool(issues))


if __name__ == "__main__":
    sys.exit(main())
