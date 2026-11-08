"""Package compiled wrappers with their native operator dependencies."""

from contextlib import contextmanager
import os
from pathlib import Path

from compiler.common.artifacts import require


@contextmanager
def _linking_environment(libraries):
    """Let GCC find native libraries and ld emit a relative runtime search path.

    Export runs in its own process: these restored environment variables are
    process-wide, while Inductor's config.patch is scoped to the current thread.
    """
    if not libraries:
        yield
        return
    previous = {name: os.environ.get(name) for name in ("LIBRARY_PATH", "LD_RUN_PATH")}
    directories = list(dict.fromkeys(str(library.parent) for library in libraries))
    if previous["LIBRARY_PATH"]:
        directories.append(previous["LIBRARY_PATH"])
    os.environ["LIBRARY_PATH"] = os.pathsep.join(directories)
    os.environ["LD_RUN_PATH"] = "$ORIGIN"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def compile_package(
    exported, destination, *, inductor_configs, native_libraries=(), custom_op_shims=None
):
    import torch

    if not native_libraries and not custom_op_shims:
        return torch._inductor.aoti_compile_and_package(
            exported, package_path=str(destination), inductor_configs=inductor_configs
        )
    libraries = [Path(path).resolve() for path in native_libraries]
    require(all(path.is_file() for path in libraries), "Native operator library is missing")
    require(
        len({path.name for path in libraries}) == len(libraries), "Duplicate native library name"
    )
    require(
        all(path.name.startswith("lib") and path.name.endswith(".so") for path in libraries),
        "Expected a native library named lib<name>.so",
    )

    from torch._inductor import config
    from torch._inductor.package import package_aoti

    shims = {
        **config.aot_inductor.custom_ops_to_c_shims,
        **(custom_op_shims or {}),
    }
    linked = list(
        dict.fromkeys(
            [*(config.aot_inductor.custom_op_libs or []), *(path.name[3:-3] for path in libraries)]
        )
    )
    args, kwargs = exported.example_inputs
    # Keep OpOverload keys out of the exporter's JSON compilation diagnostics.
    # Direct C calls retain the dependency under the linker's --as-needed rule.
    with (
        config.patch(
            {
                "aot_inductor.custom_ops_to_c_shims": shims,
                "aot_inductor.custom_op_libs": linked,
            }
        ),
        _linking_environment(libraries),
    ):
        files = torch._inductor.aot_compile(
            exported.module(),
            args,
            kwargs,
            options={**inductor_configs, "aot_inductor.package": True},
        )
    package_aoti(str(destination), [*files, *map(str, libraries)])
    return str(destination)
