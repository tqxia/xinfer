"""AOTInductor compilation, native operator bundling and shared-weight publication."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import torch

from compiler.common.artifacts import require, sha256
from compiler.common.metadata import (
    ENTRY_KINDS,
    METADATA_FILE,
    TOKENIZER_FILES,
    TOKENIZER_REQUIRED_FILES,
    finalize_metadata,
)
from compiler.common.weights import SharedWeightPool
from compiler.patches.torch_2_13.aoti_constant_handles import owned_constant_handles


SHARED_WEIGHT_OPTIONS = {
    "aot_inductor.package_constants_in_so": False,
    "always_keep_tensor_constants": True,
}


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


@contextmanager
def _capture_constants():
    from torch._inductor.graph import GraphLowering

    original = GraphLowering.allocate_non_dup_const_name
    by_runtime_name, by_name = {}, {}

    def allocate(graph, name, tensor):
        runtime_name = original(graph, name, tensor)
        if isinstance(tensor, torch.Tensor):
            by_runtime_name[runtime_name] = tensor
            if name is not None:
                by_name.setdefault(str(name), tensor)
        return runtime_name

    GraphLowering.allocate_non_dup_const_name = allocate
    try:
        yield by_runtime_name, by_name
    finally:
        GraphLowering.allocate_non_dup_const_name = original


def _resolve_constants(exported, names, captured_runtime, captured_name):
    require(len(names) == len(set(names)), "Shared weights package has duplicate constant FQNs")
    values = dict(exported.state_dict)
    values.update(
        {
            name: value
            for name, value in exported.constants.items()
            if isinstance(value, torch.Tensor)
        }
    )
    constants = {}
    for name in names:
        tensor = values.get(name)
        if tensor is None:
            tensor = captured_runtime.get(name, captured_name.get(name))
        require(
            isinstance(tensor, torch.Tensor),
            f"Shared weights cannot resolve exact constant FQN: {name}",
        )
        constants[name] = tensor
    return constants


def _compile_entry(
    kind,
    directory,
    export_entry,
    pool,
    inductor_configs,
    native_libraries,
    custom_op_shims,
):
    """Finish one graph before exporting the next, retaining only CPU tensors."""
    export_start = time.perf_counter()
    exported = export_entry(kind, directory)
    export_seconds = time.perf_counter() - export_start
    compiled_path = directory / f"{kind}.pt2"
    options = {**inductor_configs, **SHARED_WEIGHT_OPTIONS}
    print(
        json.dumps(dict(event="compile_options", entry_point=kind, inductor_configs=options)),
        flush=True,
    )
    compile_start = time.perf_counter()
    with _capture_constants() as (captured_runtime, captured_name), owned_constant_handles():
        compile_package(
            exported,
            compiled_path,
            inductor_configs=options,
            native_libraries=native_libraries,
            custom_op_shims=custom_op_shims,
        )
    compile_seconds = time.perf_counter() - compile_start
    compiled = torch._inductor.aoti_load_package(str(compiled_path))
    constants = _resolve_constants(
        exported, compiled.get_constant_fqns(), captured_runtime, captured_name
    )
    digest = sha256(compiled_path)
    pool.record_entry(
        kind, state_dict=exported.state_dict, constants=constants, package_sha256=digest
    )
    package_bytes = compiled_path.stat().st_size
    print(
        json.dumps(
            dict(
                event="compiled",
                entry_point=kind,
                package_bytes=package_bytes,
                graph_nodes=len(list(exported.graph.nodes)),
                export_seconds=round(export_seconds, 3),
                compile_seconds=round(compile_seconds, 3),
            )
        ),
        flush=True,
    )
    return dict(bytes=package_bytes, sha256=digest)


def compile_shared(
    directory,
    export_entry,
    *,
    identity,
    inductor_configs,
    metadata=None,
    native_libraries=(),
    custom_op_shims=None,
    tokenizer_dir=None,
):
    """Compile both entries and publish one complete shared-weight bundle.

    export_entry(kind, staging_directory) returns an ExportedProgram. Graph dumps
    and other auxiliary files can be written to staging_directory. Each graph is
    compiled and its constants copied to the CPU pool before the next graph is
    exported. The archive is saved once, then entry and archive checksums are
    recorded in metadata.json. Only the completed directory is exposed to readers.
    """
    directory = Path(directory)

    def check_destination():
        require(
            not directory.is_symlink()
            and (not directory.exists() or directory.is_dir() and not any(directory.iterdir())),
            f"Refusing to overwrite output directory: {directory}",
        )

    check_destination()
    directory.parent.mkdir(parents=True, exist_ok=True)
    pool = SharedWeightPool(identity)
    with tempfile.TemporaryDirectory(
        prefix=f".{directory.name}-", dir=directory.parent
    ) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        entries = {
            kind: _compile_entry(
                kind,
                staging,
                export_entry,
                pool,
                inductor_configs,
                native_libraries,
                custom_op_shims,
            )
            for kind in ENTRY_KINDS
        }
        bundle_metadata = metadata() if callable(metadata) else dict(metadata or {})
        require(isinstance(bundle_metadata, dict), "Package metadata must be an object")
        contract = pool.save(staging)
        if "model_type" in bundle_metadata:
            require(tokenizer_dir is not None, "Full-model export requires tokenizer_dir")
            source_dir = Path(tokenizer_dir)
            for name in TOKENIZER_FILES:
                source = source_dir / name
                if source.is_file():
                    shutil.copyfile(source, staging / name)
                elif name in TOKENIZER_REQUIRED_FILES:
                    raise ValueError(f"Tokenizer file is missing: {source}")
            bundle_metadata = finalize_metadata(
                bundle_metadata,
                directory=staging,
                shared_weights=contract,
            )
        else:
            # Isolated operator/CPU fixtures do not describe a full model.
            bundle_metadata = dict(bundle_metadata, entries=entries, shared_weights=contract)
        (staging / METADATA_FILE).write_text(json.dumps(bundle_metadata, indent=2) + "\n")
        check_destination()
        os.replace(staging, directory)
    return contract
