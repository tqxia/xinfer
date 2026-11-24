"""Complete prefill/decode bundles backed by one retained, verified tensor pool."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import weakref

from safetensors.torch import load_file, save_file
import torch

from compiler.common.artifacts import (
    canonical_json,
    json_sha256,
    require,
    sha256,
)
from compiler.common.metadata import finalize_metadata, validate_metadata_pair
from compiler.common.package import compile_package


SHARED_WEIGHTS_FILE = "weights.safetensors"
SHARED_WEIGHTS_MANIFEST_KEY = "xinfer_manifest"
SHARED_WEIGHTS_MANIFEST_FIELDS = frozenset(
    {"identity", "identity_sha256", "entries", "sources", "storages", "tensors"}
)
MAX_HEADER_BYTES = 100 * 1024 * 1024
SHARED_WEIGHT_OPTIONS = {
    "aot_inductor.package_constants_in_so": False,
    "always_keep_tensor_constants": True,
}


def _read_safetensors_header(path):
    size = path.stat().st_size
    require(size >= 8, "Truncated safetensors header")
    with path.open("rb") as handle:
        (header_length,) = struct.unpack("<Q", handle.read(8))
        require(
            0 < header_length <= size - 8 and header_length <= MAX_HEADER_BYTES,
            "Invalid safetensors header length",
        )
        header = json.loads(handle.read(header_length).decode("utf-8"))
    require(isinstance(header, dict), "Safetensors header must be an object")
    return header


def read_shared_weight_manifest(directory, contract):
    """Verify archive integrity and read its mappings without allocating tensor storage."""
    directory = Path(directory)
    require(
        isinstance(contract, dict)
        and set(contract) == {"file", "identity", "bytes", "sha256"}
        and contract.get("file") == SHARED_WEIGHTS_FILE
        and isinstance(contract.get("identity"), str)
        and type(contract.get("bytes")) is int
        and contract["bytes"] >= 0
        and isinstance(contract.get("sha256"), str),
        "Shared weights metadata is required; re-export both packages",
    )
    path = directory / SHARED_WEIGHTS_FILE
    require(path.is_file(), "Shared weights archive is missing")
    require(
        path.stat().st_size == contract["bytes"] and sha256(path) == contract["sha256"],
        "Shared weights archive checksum mismatch",
    )
    try:
        header = _read_safetensors_header(path)
        meta = header.get("__metadata__")
        require(
            isinstance(meta, dict) and isinstance(meta.get(SHARED_WEIGHTS_MANIFEST_KEY), str),
            "Shared weights archive metadata is missing",
        )
        data = json.loads(meta[SHARED_WEIGHTS_MANIFEST_KEY])
        require(
            isinstance(data, dict) and set(data) == SHARED_WEIGHTS_MANIFEST_FIELDS,
            "Shared weights archive metadata has invalid fields",
        )
    except (OSError, ValueError, UnicodeDecodeError, struct.error) as error:
        raise ValueError(f"Shared weights archive metadata is invalid: {error}") from error
    require(
        data.get("identity_sha256") == contract["identity"]
        and json_sha256(data.get("identity")) == contract["identity"],
        "Shared weights identity mismatch",
    )
    entries = data.get("entries", {})
    require(
        set(entries) == {"prefill", "decode"},
        "Shared weights manifest requires both package entries",
    )
    for kind, entry in entries.items():
        require(
            kind in ("prefill", "decode") and isinstance(entry.get("constants"), dict),
            "Shared weights package mapping is invalid",
        )
        package = directory / f"{kind}.pt2"
        require(
            package.is_file() and sha256(package) == entry.get("package_sha256"),
            f"Shared weights package checksum mismatch: {kind}",
        )
        for view in entry["constants"].values():
            require(
                view in data.get("tensors", {}), "Shared weights constant mapping is incomplete"
            )
    return data


def _check_storage(tensor, descriptor):
    require(
        tensor.device.type == "cpu"
        and tensor.ndim == 1
        and tensor.is_contiguous()
        and str(tensor.dtype).removeprefix("torch.") == descriptor["dtype"]
        and tensor.numel() * tensor.element_size() == descriptor["bytes"],
        "Shared weights archive tensor metadata mismatch",
    )


def _archive_storages(directory, data):
    result = load_file(str(Path(directory) / SHARED_WEIGHTS_FILE), device="cpu")
    require(set(result) == set(data["storages"]), "Shared weights archive buffer set mismatch")
    for name, tensor in result.items():
        _check_storage(tensor, data["storages"][name])
        require(
            data["storages"][name]["device"] in ("cpu", "cuda"),
            "Shared weights constant has unsupported device",
        )
    for descriptor in data["tensors"].values():
        storage = descriptor["storage"]
        require(
            storage in result
            and all(
                descriptor[key] == data["storages"][storage][key] for key in ("dtype", "device")
            ),
            "Shared weights tensor storage metadata mismatch",
        )
        _view(result[storage], descriptor)
    return result


def _view(storage, descriptor):
    shape, stride, offset = descriptor["shape"], descriptor["stride"], descriptor["offset"]
    require(
        len(shape) == len(stride)
        and offset >= 0
        and all(size >= 0 for size in shape)
        and all(step >= 0 for step in stride),
        "Shared weights tensor layout is invalid",
    )
    return storage.as_strided(shape, stride, offset)


class SharedWeights:
    """Own the pool for longer than every compiled model borrowing its pointers."""

    def __init__(self, directory, contract):
        self.directory = Path(directory)
        self.manifest = read_shared_weight_manifest(self.directory, contract)
        cpu = _archive_storages(self.directory, self.manifest)
        self.storages = {}
        for name, descriptor in self.manifest["storages"].items():
            # Pop each CPU buffer as it moves to its target device to avoid retaining both copies.
            self.storages[name] = cpu.pop(name).to(device=descriptor["device"])
        self.tensors = {
            name: _view(self.storages[descriptor["storage"]], descriptor)
            for name, descriptor in self.manifest["tensors"].items()
        }
        self.models = weakref.WeakValueDictionary()

    def load(self, kind):
        if kind not in self.models:
            entry = self.manifest["entries"][kind]
            model = torch._inductor.aoti_load_package(str(self.directory / f"{kind}.pt2"))
            names = model.get_constant_fqns()
            require(
                len(names) == len(set(names)) and set(names) == set(entry["constants"]),
                f"Shared weights constant FQNs differ: {kind}",
            )
            constants = {name: self.tensors[entry["constants"][name]] for name in names}
            model.load_constants(constants, check_full_update=True, user_managed=True)
            model._shared_weights_owner = self
            self.models[kind] = model
        return self.models[kind]


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


def _register(tensor, data, storages, cache):
    require(
        isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided,
        "Shared weights support only strided tensor constants",
    )
    device = tensor.device.type
    require(device in ("cpu", "cuda"), "Shared weights constant has unsupported device")
    dtype = str(tensor.dtype).removeprefix("torch.")
    key = (
        tensor.device,
        tensor.dtype,
        tensor.untyped_storage().data_ptr(),
        tensor.untyped_storage().nbytes(),
    )
    if key not in cache:
        elements, remainder = divmod(tensor.untyped_storage().nbytes(), tensor.element_size())
        require(not remainder, "Shared weights storage dtype alignment mismatch")
        flat = tensor.detach().as_strided((elements,), (1,), 0).to("cpu", copy=True)
        header = {"dtype": dtype, "device": device, "bytes": flat.numel() * flat.element_size()}
        digest = hashlib.sha256(canonical_json(header).encode())
        digest.update(memoryview(flat.view(torch.uint8).numpy()))
        name = "w_" + digest.hexdigest()
        if name not in storages:
            storages[name] = flat
            data["storages"][name] = header
        cache[key] = name
    descriptor = dict(
        storage=cache[key],
        dtype=dtype,
        device=device,
        shape=list(tensor.shape),
        stride=list(tensor.stride()),
        offset=tensor.storage_offset(),
    )
    view_name = "v_" + json_sha256(descriptor)
    data["tensors"][view_name] = descriptor
    return view_name


def _compile_entry(
    kind,
    directory,
    export_entry,
    data,
    storages,
    inductor_configs,
    native_libraries,
    custom_op_shims,
):
    """Finish one graph before exporting the next, retaining only CPU tensors."""
    exported, metadata_factory = export_entry(kind, directory)
    compiled_path = directory / f"{kind}.pt2"
    options = {**inductor_configs, **SHARED_WEIGHT_OPTIONS}
    print(
        json.dumps(dict(event="compile_options", entry_point=kind, inductor_configs=options)),
        flush=True,
    )
    with _capture_constants() as (captured_runtime, captured_name):
        compile_package(
            exported,
            compiled_path,
            inductor_configs=options,
            native_libraries=native_libraries,
            custom_op_shims=custom_op_shims,
        )
    compiled = torch._inductor.aoti_load_package(str(compiled_path))
    names = compiled.get_constant_fqns()
    require(len(names) == len(set(names)), "Shared weights package has duplicate constant FQNs")
    values = dict(exported.state_dict)
    values.update(
        {
            name: value
            for name, value in exported.constants.items()
            if isinstance(value, torch.Tensor)
        }
    )
    cache = {}
    for name, tensor in exported.state_dict.items():
        view = _register(tensor, data, storages, cache)
        require(
            name not in data["sources"] or data["sources"][name] == view,
            f"Shared weights source tensor differs between entries: {name}",
        )
        data["sources"][name] = view
    constants = {}
    for name in names:
        tensor = values.get(name)
        if tensor is None:
            tensor = captured_runtime.get(name, captured_name.get(name))
        require(
            isinstance(tensor, torch.Tensor),
            f"Shared weights cannot resolve exact constant FQN: {name}",
        )
        constants[name] = _register(tensor, data, storages, cache)
    data["entries"][kind] = dict(constants=constants, package_sha256=sha256(compiled_path))
    metadata = metadata_factory(compiled_path) if metadata_factory else dict(entry_point=kind)
    require(isinstance(metadata, dict), "Package metadata factory must return an object")
    return dict(metadata, inductor_configs=options)


def compile_shared(
    directory,
    export_entry,
    *,
    identity,
    inductor_configs,
    native_libraries=(),
    custom_op_shims=None,
):
    """Compile both entries and publish one complete shared-weight bundle.

    export_entry(kind, staging_directory) returns an ExportedProgram and an
    optional metadata_factory(compiled_path). Graph dumps and other auxiliary
    files can be written to staging_directory. Each graph is compiled and its
    constants copied to the CPU pool before the next graph is exported.
    The archive is saved once, then its final checksum is recorded in both
    metadata files. Only the completed directory is exposed to readers.
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
    identity_sha256 = json_sha256(identity)
    data = dict(
        identity=identity,
        identity_sha256=identity_sha256,
        entries={},
        sources={},
        storages={},
        tensors={},
    )
    storages = {}
    with tempfile.TemporaryDirectory(
        prefix=f".{directory.name}-", dir=directory.parent
    ) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        metadata = {
            kind: _compile_entry(
                kind,
                staging,
                export_entry,
                data,
                storages,
                inductor_configs,
                native_libraries,
                custom_op_shims,
            )
            for kind in ("prefill", "decode")
        }
        needed_views = {
            name for entry in data["entries"].values() for name in entry["constants"].values()
        }
        data["tensors"] = {name: data["tensors"][name] for name in needed_views}
        needed_storages = {descriptor["storage"] for descriptor in data["tensors"].values()}
        data["storages"] = {name: data["storages"][name] for name in needed_storages}
        archive_path = staging / SHARED_WEIGHTS_FILE
        ordered = {name: storages[name] for name in sorted(needed_storages)}
        save_file(
            ordered,
            str(archive_path),
            metadata={SHARED_WEIGHTS_MANIFEST_KEY: canonical_json(data)},
        )
        contract = dict(
            file=SHARED_WEIGHTS_FILE,
            identity=identity_sha256,
            bytes=archive_path.stat().st_size,
            sha256=sha256(archive_path),
        )
        for kind, entry_metadata in metadata.items():
            package = staging / f"{kind}.pt2"
            if "model_type" in entry_metadata:
                metadata[kind] = finalize_metadata(
                    entry_metadata,
                    package_path=package,
                    inductor_configs=entry_metadata["inductor_configs"],
                    shared_weights=contract,
                )
            else:
                # Isolated operator/CPU fixtures do not describe a full model.
                metadata[kind] = dict(
                    entry_metadata,
                    shared_weights=contract,
                    package_bytes=package.stat().st_size,
                    package_sha256=sha256(package),
                )
        if any("model_type" in item for item in metadata.values()):
            validate_metadata_pair(metadata["prefill"], metadata["decode"])
        for kind, entry_metadata in metadata.items():
            (staging / f"{kind}.json").write_text(json.dumps(entry_metadata, indent=2) + "\n")
        check_destination()
        os.replace(staging, directory)
    return contract
