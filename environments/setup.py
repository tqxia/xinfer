"""Create an isolated model venv from an existing, checked native installation."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from environments import ROOT
from environments.check import validate


# Run dependency-marker evaluation and RECORD lookup in the supplied interpreter.
# Never expose its complete site-packages directory to the new environment.
PROBE = r"""
import importlib.metadata as metadata
import json
from pathlib import Path
try:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
except ImportError:
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name
from environments.check import collect

actual = collect()
pending, seen, versions, links = [("torch", set()), ("triton", set())], {}, {}, {}
while pending:
    name, extras = pending.pop()
    name = canonicalize_name(name)
    requested = seen.get(name, set()) | extras
    if name in seen and requested == seen[name]:
        continue
    seen[name] = requested
    distribution = metadata.distribution(name)
    versions[name] = distribution.version
    for text in distribution.requires or []:
        requirement = Requirement(text)
        if requirement.marker and not any(
                requirement.marker.evaluate({"extra": extra}) for extra in requested | {""}):
            continue
        version = metadata.version(requirement.name)
        if version not in requirement.specifier:
            raise RuntimeError(f"Native dependency {requirement} has incompatible version {version}")
        pending.append((requirement.name, set(requirement.extras)))
    if distribution.files is None:
        raise RuntimeError(f"Native distribution has no installed file list: {name}")
    for file in distribution.files:
        first = file.parts[0]
        if first in ("..", ".") or file.is_absolute() or first.endswith((".data", ".pth")):
            continue
        source = Path(distribution.locate_file(first)).resolve()
        if not source.exists():
            raise RuntimeError(f"Missing native package component: {source}")
        if first in links and links[first] != str(source):
            raise RuntimeError(f"Conflicting native namespace: {first}")
        links[first] = str(source)
print(json.dumps(dict(actual=actual, versions=versions, links=links)))
"""


def link_packages(site, links):
    """Only populate absent top-level venv entries; never overwrite a seed package."""
    for name, source in links.items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Invalid native package component: {name}")
        destination = site / name
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Venv already contains {destination}")
        if not Path(source).exists():
            raise FileNotFoundError(source)
    for name, source in sorted(links.items()):
        (site / name).symlink_to(source, target_is_directory=Path(source).is_dir())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("l20", "gb10"), required=True)
    parser.add_argument("--model", choices=("qwen3", "qwen35"), required=True)
    parser.add_argument(
        "--native-python",
        type=Path,
        required=True,
        help="Existing Python with this target's Torch, Triton and CUDA packages",
    )
    parser.add_argument(
        "--output", type=Path, help="Fresh venv; defaults to work/<target>/<model>-venv"
    )
    args = parser.parse_args(argv)
    output = (args.output or ROOT / "work" / args.target / f"{args.model}-venv").absolute()
    if output.exists() or output.is_symlink():
        parser.error(f"Refusing to overwrite {output}")
    native = args.native_python.absolute()
    if not native.is_file():
        parser.error(f"Native Python does not exist: {native}")
    try:
        probe = subprocess.run(
            [str(native), "-c", PROBE],
            cwd=ROOT,
            check=True,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            capture_output=True,
            text=True,
            timeout=60,
        )
        data = json.loads(probe.stdout)
    except (subprocess.SubprocessError, ValueError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        parser.error(f"Cannot inspect native environment: {detail}")
    profile = json.loads((ROOT / "environments" / args.target / "profile.json").read_text())
    issues = validate(data["actual"], profile, {})
    if issues:
        parser.error("Native environment does not match profile:\n" + "\n".join(issues))
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(native), "-m", "venv", str(output)], check=True)
    python = output / "bin/python"
    query = "import sysconfig; print(sysconfig.get_path('purelib'))"
    site = Path(subprocess.check_output([str(python), "-c", query], text=True).strip())
    link_packages(site, data["links"])
    constraints = output / "native-constraints.txt"
    constraints.write_text(
        "".join(f"{name}=={version}\n" for name, version in sorted(data["versions"].items()))
    )
    data.update(
        target=args.target,
        model=args.model,
        native_python=str(native),
        profile=profile,
        venv=str(output),
    )
    (output / "native-provenance.json").write_text(json.dumps(data, indent=2) + "\n")
    print(f"Created {output}")
    print(
        f"Install model dependencies with {python} -m pip install "
        f"-c {constraints} -r {ROOT / 'compiler' / args.model / 'requirements.txt'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
