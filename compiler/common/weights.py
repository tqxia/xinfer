"""Weightless AOTI packages backed by one retained, verified tensor pool."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import weakref
import zipfile

import torch

from compiler.common.artifacts import (
    canonical_json, json_sha256, read_json, require, sha256,
)


SHARED_WEIGHT_OPTIONS = {"aot_inductor.package_constants_in_so": False,
                         "always_keep_tensor_constants": True}


class _Archive(torch.nn.Module):
    def __init__(self, identity):
        super().__init__()
        self.identity = identity

    def forward(self):
        return self.identity


def _manifest(directory, contract, *, pair=True):
    directory = Path(directory)
    require(isinstance(contract, dict) and contract.get("format") == 2 and
             contract.get("file") == "weights.pt" and isinstance(contract.get("identity"), str) and
             type(contract.get("bytes")) is int and contract["bytes"] >= 0 and
             isinstance(contract.get("sha256"), str),
             "Shared weights metadata is required; re-export both packages")
    path = directory / "weights.pt"
    require(path.is_file(), "Shared weights archive is missing")
    require(path.stat().st_size == contract["bytes"] and sha256(path) == contract["sha256"],
             "Shared weights archive checksum mismatch")
    try:
        with zipfile.ZipFile(path) as zipped:
            records = [name for name in zipped.namelist() if name.endswith("/extra/weights.json")]
            require(len(records) == 1, "Shared weights archive metadata is missing")
            data = json.loads(zipped.read(records[0]))
        require(isinstance(data, dict) and data.get("format") == 2,
                 "Shared weights archive metadata has unsupported format")
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError(f"Shared weights archive metadata is invalid: {error}") from error
    require(data.get("identity_sha256") == contract["identity"] and
             json_sha256(data.get("identity")) == contract["identity"], "Shared weights identity mismatch")
    entries = data.get("entries", {})
    if pair:
        require(set(entries) == {"prefill", "decode"}, "Shared weights manifest requires both package entries")
    for kind, entry in entries.items():
        require(kind in ("prefill", "decode") and isinstance(entry.get("constants"), dict),
                 "Shared weights package mapping is invalid")
        package = directory / f"{kind}.pt2"
        require(package.is_file() and sha256(package) == entry.get("package_sha256"),
                 f"Shared weights package checksum mismatch: {kind}")
        for view in entry["constants"].values():
            require(view in data.get("tensors", {}), "Shared weights constant mapping is incomplete")
    return data


def validate_shared_weights(directory, prefill, decode):
    """Validate the complete bundle without allocating CUDA storage."""
    require("shared_weights" in prefill and "shared_weights" in decode,
             "Shared weights metadata is required; re-export both packages")
    require(prefill["shared_weights"] == decode["shared_weights"], "Shared weights contracts differ")
    for metadata in (prefill, decode):
        options = metadata.get("inductor_configs", {})
        require(all(key in options and options[key] == value for key, value in SHARED_WEIGHT_OPTIONS.items()),
                 "Shared weights require weightless package compiler options")
    return _manifest(directory, decode["shared_weights"])


def _check_storage(tensor, descriptor):
    require(tensor.device.type == "cpu" and tensor.ndim == 1 and tensor.is_contiguous() and
             str(tensor.dtype).removeprefix("torch.") == descriptor["dtype"] and
             tensor.numel() * tensor.element_size() == descriptor["bytes"],
             "Shared weights archive tensor metadata mismatch")


def _archive_storages(directory, data):
    archive = torch.jit.load(str(Path(directory) / "weights.pt"), map_location="cpu")
    require(archive.identity == data["identity_sha256"], "Shared weights archive identity mismatch")
    result = dict(archive.named_buffers())
    require(set(result) == set(data["storages"]), "Shared weights archive buffer set mismatch")
    for name, tensor in result.items():
        _check_storage(tensor, data["storages"][name])
        require(data["storages"][name]["device"] in ("cpu", "cuda"),
                 "Shared weights constant has unsupported device")
    for descriptor in data["tensors"].values():
        storage = descriptor["storage"]
        require(storage in result and all(descriptor[key] == data["storages"][storage][key]
                                          for key in ("dtype", "device")),
                 "Shared weights tensor storage metadata mismatch")
        _view(result[storage], descriptor)
    return result


def _view(storage, descriptor):
    shape, stride, offset = descriptor["shape"], descriptor["stride"], descriptor["offset"]
    require(len(shape) == len(stride) and offset >= 0 and
             all(size >= 0 for size in shape) and all(step >= 0 for step in stride),
             "Shared weights tensor layout is invalid")
    return storage.as_strided(shape, stride, offset)


class SharedWeights:
    """Own the pool for longer than every compiled model borrowing its pointers."""
    def __init__(self, directory, contract):
        self.directory = Path(directory)
        self.manifest = _manifest(self.directory, contract)
        cpu = _archive_storages(self.directory, self.manifest)
        self.storages = {name: value.to(device=descriptor["device"])
                         for name, descriptor in self.manifest["storages"].items()
                         for value in (cpu[name],)}
        self.tensors = {name: _view(self.storages[descriptor["storage"]], descriptor)
                        for name, descriptor in self.manifest["tensors"].items()}
        self.models = weakref.WeakValueDictionary()

    def load(self, kind):
        if kind not in self.models:
            entry = self.manifest["entries"][kind]
            model = torch._inductor.aoti_load_package(str(self.directory / f"{kind}.pt2"))
            names = model.get_constant_fqns()
            require(len(names) == len(set(names)) and set(names) == set(entry["constants"]),
                     f"Shared weights constant FQNs differ: {kind}")
            constants = {name: self.tensors[entry["constants"][name]] for name in names}
            model.load_constants(constants, check_full_update=True, user_managed=True)
            model._shared_weights_owner = self
            self.models[kind] = model
        return self.models[kind]


@contextmanager
def _capture_constants():
    from torch._inductor.graph import GraphLowering
    original = GraphLowering.allocate_non_dup_const_name
    captured = {}

    def allocate(graph, name, tensor):
        runtime_name = original(graph, name, tensor)
        if isinstance(tensor, torch.Tensor):
            captured[runtime_name] = tensor
            if name is not None:
                captured[str(name)] = tensor
        return runtime_name

    GraphLowering.allocate_non_dup_const_name = allocate
    try:
        yield captured
    finally:
        GraphLowering.allocate_non_dup_const_name = original


def _register(tensor, data, storages, cache):
    require(isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided,
             "Shared weights support only strided tensor constants")
    device = tensor.device.type
    require(device in ("cpu", "cuda"), "Shared weights constant has unsupported device")
    dtype = str(tensor.dtype).removeprefix("torch.")
    key = (tensor.device, tensor.dtype, tensor.untyped_storage().data_ptr(),
           tensor.untyped_storage().nbytes())
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
    descriptor = dict(storage=cache[key], dtype=dtype, device=device, shape=list(tensor.shape),
                      stride=list(tensor.stride()), offset=tensor.storage_offset())
    view_name = "v_" + json_sha256(descriptor)
    data["tensors"][view_name] = descriptor
    return view_name


def compile_shared(exported, package_path, *, identity, inductor_configs, metadata_factory=None):
    """Compile an entry and publish its archive and complete package metadata.

    metadata_factory receives the temporary compiled package path for its size
    and checksum fields. The helper supplies the effective compiler options and
    shared_weights record. When the second entry grows the archive, it also
    refreshes the already-exported entry's checksum.
    """
    package_path = Path(package_path)
    kind = package_path.stem
    require(kind in ("prefill", "decode"), "Shared weights entry must be prefill or decode")
    require(not package_path.exists(), f"Refusing to overwrite {package_path}")
    directory = package_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    identity_sha256 = json_sha256(identity)
    other_kind = "decode" if kind == "prefill" else "prefill"
    other_metadata = None
    options = {**inductor_configs, **SHARED_WEIGHT_OPTIONS}
    with (directory / ".weights.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / "weights.pt").exists():
            require((directory / f"{other_kind}.json").is_file(),
                    f"Shared weights metadata for {other_kind} is missing")
            other_metadata = read_json(directory / f"{other_kind}.json")
            other_contract = other_metadata.get("shared_weights", {})
            require(other_contract.get("identity") == identity_sha256, "Shared weights identity mismatch")
            data = _manifest(directory, other_contract, pair=False)
            require(kind not in data["entries"], f"Shared weights entry already exists: {kind}")
            storages = _archive_storages(directory, data)
        else:
            require(not (directory / f"{other_kind}.json").exists(),
                    f"Shared weights metadata for {other_kind} exists without archive")
            data = dict(format=2, identity=identity, identity_sha256=identity_sha256,
                        entries={}, sources={}, storages={}, tensors={})
            storages = {}
        with tempfile.TemporaryDirectory(prefix=".weights-", dir=directory) as temporary:
            temporary = Path(temporary)
            compiled_path = temporary / f"{kind}.pt2"
            with _capture_constants() as captured:
                torch._inductor.aoti_compile_and_package(
                    exported, package_path=str(compiled_path), inductor_configs=options)
            compiled = torch._inductor.aoti_load_package(str(compiled_path))
            names = compiled.get_constant_fqns()
            require(len(names) == len(set(names)), "Shared weights package has duplicate constant FQNs")
            values = dict(exported.state_dict)
            values.update({name: value for name, value in exported.constants.items()
                           if isinstance(value, torch.Tensor)})
            cache = {}
            for name, tensor in exported.state_dict.items():
                view = _register(tensor, data, storages, cache)
                require(name not in data["sources"] or data["sources"][name] == view,
                         f"Shared weights source tensor differs between entries: {name}")
                data["sources"][name] = view
            constants = {}
            for name in names:
                tensor = values.get(name, captured.get(name))
                require(isinstance(tensor, torch.Tensor),
                         f"Shared weights cannot resolve exact constant FQN: {name}")
                constants[name] = _register(tensor, data, storages, cache)
            del compiled
            data["entries"][kind] = dict(constants=constants, package_sha256=sha256(compiled_path))
            needed_views = {name for entry in data["entries"].values()
                            for name in entry["constants"].values()}
            data["tensors"] = {name: data["tensors"][name] for name in needed_views}
            needed_storages = {descriptor["storage"] for descriptor in data["tensors"].values()}
            data["storages"] = {name: data["storages"][name] for name in needed_storages}
            storages = {name: storages[name] for name in needed_storages}
            archive = _Archive(identity_sha256)
            for name, tensor in sorted(storages.items()):
                archive.register_buffer(name, tensor)
            archive_path = temporary / "weights.pt"
            torch.jit.save(torch.jit.script(archive), str(archive_path),
                           _extra_files={"weights.json": canonical_json(data)})
            contract = dict(format=2, file="weights.pt", identity=identity_sha256,
                            bytes=archive_path.stat().st_size, sha256=sha256(archive_path))
            metadata = metadata_factory(compiled_path) if metadata_factory else dict(entry_point=kind)
            require(isinstance(metadata, dict), "Package metadata factory must return an object")
            metadata = dict(metadata, shared_weights=contract, inductor_configs=options)
            metadata_path = temporary / f"{kind}.json"
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            if other_metadata is not None:
                other_metadata["shared_weights"] = contract
                (temporary / f"{other_kind}.json").write_text(json.dumps(other_metadata, indent=2) + "\n")
            # A reader racing publication rejects mixed archive checksums.
            # All metadata is prepared before replacing any bundle files.
            os.replace(archive_path, directory / "weights.pt")
            os.replace(compiled_path, package_path)
            if other_metadata is not None:
                os.replace(temporary / f"{other_kind}.json", directory / f"{other_kind}.json")
            os.replace(metadata_path, directory / f"{kind}.json")
    return contract
