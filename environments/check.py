"""Check a tested native stack and target environment before build or export."""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from environments import ROOT

TARGETS = {
    "l20": dict(system="Linux", machine="x86_64", python="3.12", sm=[8, 9]),
    "pro5000": dict(system="Linux", machine="x86_64", python="3.12", sm=[12, 0]),
    "gb10": dict(system="Linux", machine="aarch64", python="3.12", sm=[12, 1]),
}


def read_requirements(path, *, _seen=None):
    """Read exact package pins recursively, leaving pip index options to the installer."""
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError(f"Circular requirement include: {path}")
    seen.add(path)
    result = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(("-r ", "--requirement ")):
            included = line.split(None, 1)[1].strip()
            for name, version in read_requirements(path.parent / included, _seen=seen).items():
                if name in result and result[name] != version:
                    raise ValueError(f"Conflicting pin for {name} in {path}")
                result[name] = version
            continue
        if line.startswith("-"):
            continue
        name, separator, version = line.partition("==")
        if not separator or not name.strip() or not version.strip():
            raise ValueError(f"Expected an exact package pin in {path}: {line}")
        name, version = name.strip(), version.strip()
        if name in result and result[name] != version:
            raise ValueError(f"Conflicting pin for {name} in {path}")
        result[name] = version
    return result


def target_requirements(target, *, include_dev=False):
    requirements = read_requirements(ROOT / "environments" / target / "requirements.txt")
    if include_dev:
        requirements.update(read_requirements(ROOT / "requirements-dev.txt"))
    return requirements


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


def collect(requirements, build=None):
    actual = dict(
        python=dict(version=platform.python_version(), executable=sys.executable),
        platform=dict(system=platform.system(), machine=platform.machine()),
        packages={name: package_version(name) for name in requirements},
        native_tools={name: native_tool(name) for name in ("c++", "nvcc")},
    )
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


def detect_gpu_sm():
    """Return the unique [major, minor] GPU compute capability when detectable, else None."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        process = subprocess.run(
            [nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if process.returncode != 0:
        return None
    caps = {line.strip() for line in process.stdout.splitlines() if line.strip()}
    if len(caps) != 1:
        return None
    major, sep, minor = next(iter(caps)).partition(".")
    if not sep or not major.isdecimal() or not minor.isdecimal():
        return None
    return [int(major), int(minor)]


def default_target(machine=None, sm=None):
    arch = normalized_machine(machine or platform.machine())
    candidates = [name for name, profile in TARGETS.items() if profile["machine"] == arch]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    observed_sm = detect_gpu_sm() if sm is None else list(sm)
    if observed_sm is None:
        return None
    matches = [name for name in candidates if TARGETS[name]["sm"] == observed_sm]
    return matches[0] if len(matches) == 1 else None


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
    expected_python = profile.get("python")
    if expected_python:
        observed_python = ".".join(actual["python"]["version"].split(".")[:2])
        expect("Python version", observed_python, expected_python)
    for name, info in actual.get("native_tools", {}).items():
        if not info.get("path"):
            issues.append(f"Native tool {name!r} is missing from PATH")
        elif "error" in info:
            issues.append(f"Native tool {name!r} failed: {info['error']}")
        elif info.get("exit_code") != 0:
            issues.append(f"Native tool {name!r} exited with {info.get('exit_code')}")
    for name, version in requirements.items():
        if name != "torch":
            expect(name, actual["packages"].get(name), version)
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


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=tuple(TARGETS),
        help="Target profile (default: auto-detected from CPU/GPU)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Also verify formatting tools pinned in requirements-dev.txt",
    )
    parser.add_argument(
        "--build", type=Path, help="Also check this configured CMake build's Torch_DIR"
    )
    parser.add_argument(
        "--output", type=Path, help="Save the same JSON printed to stdout, including failures"
    )
    return parser


def check_environment(target, build=None, *, include_dev=False):
    """Collect and check the selected environment, returning its complete report."""
    profile = TARGETS[target]
    requirements = target_requirements(target, include_dev=include_dev)
    actual = collect(requirements, build)
    issues = validate(actual, profile, requirements)
    return dict(
        target=target,
        expected=profile,
        requirements=requirements,
        actual=actual,
        issues=issues,
        passed=not issues,
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    target = args.target or default_target()
    if target is None:
        choices = "|".join(TARGETS)
        parser.error(
            f"Cannot auto-detect target for {platform.machine()!r}; pass --target {choices}"
        )
    report = check_environment(target, args.build, include_dev=args.dev)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return int(bool(report["issues"]))


if __name__ == "__main__":
    sys.exit(main())
