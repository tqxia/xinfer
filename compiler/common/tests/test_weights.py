"""CPU roundtrip and rejection coverage for externally owned AOTI weights."""

import gc
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch
import weakref

from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

from compiler.common.artifacts import json_sha256, sha256
from compiler.common.metadata import METADATA_FILE
from compiler.common.compile import (
    SHARED_WEIGHT_OPTIONS,
    compile_package,
    compile_shared,
)
from compiler.common.weights import (
    SHARED_WEIGHTS_FILE,
    SHARED_WEIGHTS_MANIFEST_KEY,
    SharedWeights,
    read_shared_weight_manifest,
)


class TiedViews(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base = torch.arange(24, dtype=torch.bfloat16).reshape(4, 6) / 16
        self.weight = torch.nn.Parameter(base, requires_grad=False)
        self.tied = self.weight
        self.sliced = torch.nn.Parameter(base[:, ::2], requires_grad=False)
        self.bias = torch.nn.Parameter(base[:, 1], requires_grad=False)
        self.register_buffer("scale", torch.tensor([1.0, 2.0, 4.0, 8.0]))

    def forward(self, x):
        result = F.linear(x, self.weight).float() + F.linear(x, self.tied).float()
        result = (
            result + F.linear(x[:, ::2], self.sliced).float() + self.bias.float()
        ) * self.scale
        return result, self.weight, self.tied, self.sliced, self.bias, self.scale


def inputs(rows):
    return torch.arange(rows * 6, dtype=torch.bfloat16).reshape(rows, 6) / 8


def export_entry(model, kind, directory):
    exported = torch.export.export(model, (inputs(1 if kind == "decode" else 3),))
    (directory / f"{kind}.graph.txt").write_text(exported.graph_module.code)
    return exported, None


def compile_bundle(model, directory, identity):
    with torch.inference_mode():
        return compile_shared(
            directory,
            lambda kind, stage: export_entry(model, kind, stage),
            identity=identity,
            inductor_configs={"emulate_precision_casts": True},
            metadata=dict(marker="preserved metadata"),
        )


def metadata(directory):
    return json.loads((directory / METADATA_FILE).read_text())


def mapping(directory):
    path = directory / SHARED_WEIGHTS_FILE
    with path.open("rb") as handle:
        (header_length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_length).decode("utf-8"))
    return json.loads(header["__metadata__"][SHARED_WEIGHTS_MANIFEST_KEY])


class SharedWeightBundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.bundle = cls.root / "complete"
        cls.identity = dict(model="cpu-tied-view-fixture", revision="one")
        cls.model = TiedViews().eval()
        with patch("compiler.common.weights.save_file", wraps=save_file) as save_archive:
            cls.contract = compile_bundle(cls.model, cls.bundle, cls.identity)
        cls.archive_writes = save_archive.call_count

    def copy_bundle(self, source=None):
        temporary = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(temporary.cleanup)
        destination = Path(temporary.name) / "bundle"
        shutil.copytree(source or self.bundle, destination)
        return destination

    def rewrite_archive(
        self, directory, mutate=None, *, update_checksums=True, include_mapping=True, identity=None
    ):
        payload = mapping(directory)
        if identity is not None:
            payload.update(identity=identity, identity_sha256=json_sha256(identity))
        if mutate:
            mutate(payload)
        path = directory / SHARED_WEIGHTS_FILE
        tensors = load_file(str(path), device="cpu")
        rewritten = directory / "rewritten.safetensors"
        extra = {SHARED_WEIGHTS_MANIFEST_KEY: json.dumps(payload)} if include_mapping else None
        save_file(tensors, str(rewritten), metadata=extra)
        rewritten.replace(path)
        contract = dict(
            metadata(directory)["shared_weights"], bytes=path.stat().st_size, sha256=sha256(path)
        )
        if identity is not None:
            contract["identity"] = payload["identity_sha256"]
        if update_checksums:
            package = metadata(directory)
            package["shared_weights"] = contract
            (directory / METADATA_FILE).write_text(json.dumps(package))
        return contract

    def test_one_export_publishes_complete_pair_with_one_archive(self):
        self.assertEqual(self.archive_writes, 1)
        self.assertEqual(set(mapping(self.bundle)["entries"]), {"prefill", "decode"})
        bundle_meta = metadata(self.bundle)
        self.assertEqual(self.contract, bundle_meta["shared_weights"])
        self.assertEqual(self.contract["bytes"], (self.bundle / SHARED_WEIGHTS_FILE).stat().st_size)
        self.assertEqual(self.contract["sha256"], sha256(self.bundle / SHARED_WEIGHTS_FILE))
        self.assertEqual(bundle_meta["marker"], "preserved metadata")
        self.assertFalse((self.bundle / "weights.pt").exists())
        self.assertFalse((self.bundle / "weights.json").exists())
        self.assertFalse((self.bundle / "prefill.json").exists())
        self.assertFalse((self.bundle / "decode.json").exists())
        for kind in ("prefill", "decode"):
            self.assertEqual(
                bundle_meta["entries"][kind]["sha256"], sha256(self.bundle / (kind + ".pt2"))
            )
            self.assertTrue((self.bundle / (kind + ".graph.txt")).is_file())
        read_shared_weight_manifest(self.bundle, self.contract)
        directory = self.copy_bundle()
        (directory / "weights.json").write_text("unused old sidecar")
        read_shared_weight_manifest(directory, metadata(directory)["shared_weights"])

    def test_model_identity_does_not_select_operator_payload_checks(self):
        for model in ("qwen3", "qwen3.5"):
            with self.subTest(model=model):
                directory = self.copy_bundle()
                contract = self.rewrite_archive(
                    directory, identity=dict(model=model, revision="cpu-fixture")
                )
                manifest = read_shared_weight_manifest(directory, contract)
                self.assertEqual(manifest["identity"]["model"], model)
                pool = SharedWeights(directory, contract)
                with torch.inference_mode():
                    for kind, rows in (("decode", 1), ("prefill", 3)):
                        compiled = pool.load(kind)
                        torch.testing.assert_close(
                            compiled(inputs(rows)), self.model(inputs(rows)), rtol=0, atol=0
                        )

    def test_real_cpu_packages_share_tied_strided_storage(self):
        manifest = read_shared_weight_manifest(self.bundle, self.contract)
        pool = SharedWeights(self.bundle, self.contract)
        source = {name: pool.tensors[key] for name, key in manifest["sources"].items()}
        names = ("weight", "tied", "sliced", "bias")
        self.assertEqual(len({source[name].untyped_storage().data_ptr() for name in names}), 1)
        for name, original in self.model.state_dict().items():
            value = source[name]
            self.assertEqual(value.dtype, original.dtype)
            self.assertEqual(value.device.type, "cpu")
            self.assertEqual(value.stride(), original.stride())
            self.assertEqual(value.storage_offset(), original.storage_offset())
            self.assertTrue(torch.equal(value, original), name)
        self.assertEqual(source["bias"].storage_offset(), 1)
        self.assertEqual(source["sliced"].stride(), (6, 2))
        with torch.inference_mode():
            for kind, rows in (("decode", 1), ("prefill", 3)):
                compiled = pool.load(kind)
                self.assertIs(compiled, pool.load(kind))
                torch.testing.assert_close(
                    compiled(inputs(rows)), self.model(inputs(rows)), rtol=0, atol=0
                )

    def test_compile_options_have_one_owner_and_do_not_mutate_the_caller(self):
        from torch._inductor import config

        directory = self.root / "compile-options"
        requested = {"emulate_precision_casts": True, "aot_inductor.package_constants_in_so": True}
        original = dict(requested)
        # An unused operator exercises shim forwarding without needing a native DSO.
        # OpOverload keys also catch accidental inclusion in JSON metadata/options.
        operator = torch.ops.aten.sin.default
        shims = {
            operator: [
                "AOTITorchError aoti_torch_cpu_sin("
                "AtenTensorHandle input, AtenTensorHandle* output)"
            ]
        }
        original_shims = {operator: list(shims[operator])}
        previous_shims = dict(config.aot_inductor.custom_ops_to_c_shims)
        real_compile = torch._inductor.aot_compile

        def compile_with_shims(*args, **kwargs):
            self.assertEqual(
                config.aot_inductor.custom_ops_to_c_shims,
                {**previous_shims, **shims},
            )
            return real_compile(*args, **kwargs)

        def prepare(kind, stage):
            return export_entry(self.model, kind, stage)

        with (
            torch.inference_mode(),
            patch(
                "compiler.common.compile.compile_package",
                wraps=compile_package,
            ) as compile_graph,
            patch("torch._inductor.aot_compile", side_effect=compile_with_shims) as compile_native,
        ):
            compile_shared(
                directory,
                prepare,
                identity=self.identity,
                inductor_configs=requested,
                custom_op_shims=shims,
            )
        self.assertEqual(requested, original)
        self.assertEqual(shims, original_shims)
        self.assertEqual(config.aot_inductor.custom_ops_to_c_shims, previous_shims)
        self.assertEqual(compile_graph.call_count, 2)
        self.assertEqual(compile_native.call_count, 2)
        first, second = (call.kwargs["inductor_configs"] for call in compile_graph.call_args_list)
        self.assertIsNot(first, second)
        self.assertNotIn("inductor_configs", metadata(directory))
        for call in compile_graph.call_args_list:
            self.assertIs(call.kwargs["custom_op_shims"], shims)
            self.assertEqual(
                call.kwargs["inductor_configs"], {**requested, **SHARED_WEIGHT_OPTIONS}
            )

    def test_populated_and_invalid_destinations_are_rejected_before_export(self):
        temporary = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        populated = self.copy_bundle()
        file = root / "file"
        file.write_text("preserve")
        target = root / "empty"
        target.mkdir()
        link = root / "link"
        link.symlink_to(target, target_is_directory=True)
        before = {path.name: sha256(path) for path in populated.iterdir() if path.is_file()}
        for directory in (populated, file, link):
            with self.subTest(directory=directory):
                prepare = Mock(side_effect=AssertionError("Export must not run"))
                with self.assertRaises(ValueError):
                    compile_shared(directory, prepare, identity=self.identity, inductor_configs={})
                prepare.assert_not_called()
        self.assertEqual(
            before, {path.name: sha256(path) for path in populated.iterdir() if path.is_file()}
        )
        self.assertEqual(file.read_text(), "preserve")
        self.assertTrue(link.is_symlink())
        self.assertEqual(list(target.iterdir()), [])

    def test_empty_output_is_published_only_after_both_entries_succeed(self):
        directory = self.root / "initially-empty"
        directory.mkdir()
        kinds = []

        def prepare(kind, stage):
            kinds.append(kind)
            self.assertNotEqual(stage, directory)
            self.assertEqual(list(directory.iterdir()), [])
            return export_entry(self.model, kind, stage)

        with torch.inference_mode():
            compile_shared(directory, prepare, identity=self.identity, inductor_configs={})
        self.assertEqual(kinds, ["prefill", "decode"])
        read_shared_weight_manifest(directory, metadata(directory)["shared_weights"])

    def test_second_compile_or_metadata_failure_leaves_no_published_bundle(self):
        real_compile = torch._inductor.aoti_compile_and_package
        for failure in ("compile", "metadata"):
            for existing_empty in (False, True):
                with self.subTest(failure=failure, existing_empty=existing_empty):
                    temporary = tempfile.TemporaryDirectory(dir=self.root)
                    self.addCleanup(temporary.cleanup)
                    root = Path(temporary.name)
                    directory = root / "bundle"
                    if existing_empty:
                        directory.mkdir()
                    before = set(root.iterdir())
                    kinds = []

                    def prepare(kind, stage):
                        kinds.append(kind)
                        return export_entry(self.model, kind, stage)

                    def fail_metadata():
                        raise RuntimeError("decode metadata failed")

                    def compile_graph(exported, **options):
                        if failure == "compile" and Path(options["package_path"]).stem == "decode":
                            raise RuntimeError("decode compile failed")
                        return real_compile(exported, **options)

                    with (
                        torch.inference_mode(),
                        patch(
                            "torch._inductor.aoti_compile_and_package", side_effect=compile_graph
                        ),
                        patch("compiler.common.weights.save_file", wraps=save_file) as save_archive,
                    ):
                        with self.assertRaisesRegex(RuntimeError, f"decode {failure} failed"):
                            compile_shared(
                                directory,
                                prepare,
                                identity=self.identity,
                                inductor_configs={},
                                metadata=fail_metadata if failure == "metadata" else None,
                            )
                    self.assertEqual(kinds, ["prefill", "decode"])
                    save_archive.assert_not_called()
                    self.assertEqual(
                        set(root.iterdir()), before, "Staging files must be cleaned up"
                    )
                    if existing_empty:
                        self.assertEqual(list(directory.iterdir()), [])
                    else:
                        self.assertFalse(directory.exists())

    def test_changed_parameter_between_entries_is_rejected_before_publication(self):
        directory = self.root / "changed-source"
        changed = TiedViews().eval()
        with torch.inference_mode():
            changed.weight.add_(1)
        with (
            torch.inference_mode(),
            self.assertRaisesRegex(ValueError, "source tensor differs between entries"),
        ):
            compile_shared(
                directory,
                lambda kind, stage: export_entry(
                    changed if kind == "decode" else self.model, kind, stage
                ),
                identity=self.identity,
                inductor_configs={},
            )
        self.assertFalse(directory.exists())

    def test_missing_and_corrupt_bundle_files_are_rejected_without_cuda(self):
        for filename, corrupt, message in (
            (SHARED_WEIGHTS_FILE, False, "archive is missing"),
            (SHARED_WEIGHTS_FILE, True, "archive checksum mismatch"),
            ("prefill.pt2", True, "package checksum mismatch"),
            ("prefill.pt2", False, "package checksum mismatch"),
        ):
            with self.subTest(filename=filename, corrupt=corrupt):
                directory = self.copy_bundle()
                path = directory / filename
                path.write_bytes(b"corrupt") if corrupt else path.unlink()
                with patch(
                    "torch.cuda.is_available",
                    side_effect=AssertionError("CUDA must not be queried"),
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        read_shared_weight_manifest(directory, self.contract)

    def test_missing_embedded_mapping_is_rejected_without_cuda(self):
        directory = self.copy_bundle()
        self.rewrite_archive(directory, include_mapping=False)
        with patch(
            "torch.cuda.is_available", side_effect=AssertionError("CUDA must not be queried")
        ):
            with self.assertRaisesRegex(ValueError, "archive metadata is invalid"):
                read_shared_weight_manifest(directory, metadata(directory)["shared_weights"])

    def test_archive_contract_rejects_retired_file_and_format_keys(self):
        for altered in (
            dict(self.contract, file="weights.safetensors"),
            dict(self.contract, file="../weights.safetensors"),
            dict(self.contract, file="weights.pt"),
            dict(self.contract, format=2, file="weights.pt"),
            dict(self.contract, format=3),
        ):
            with (
                self.subTest(altered=altered),
                self.assertRaisesRegex(ValueError, "metadata is required"),
            ):
                read_shared_weight_manifest(self.bundle, altered)

    def test_archive_dtype_device_and_view_layout_are_checked(self):
        for field, value, message in (
            ("dtype", "int8", "archive tensor metadata mismatch"),
            ("device", "meta", "unsupported device"),
            ("stride", -1, "tensor layout is invalid"),
        ):
            with self.subTest(field=field):
                directory = self.copy_bundle()

                def mutate(manifest):
                    if field == "stride":
                        next(iter(manifest["tensors"].values()))["stride"][0] = value
                    else:
                        next(iter(manifest["storages"].values()))[field] = value

                contract = self.rewrite_archive(directory, mutate)
                with self.assertRaisesRegex(ValueError, message):
                    SharedWeights(directory, contract)

    def test_embedded_mapping_changes_invalidate_package_checksums(self):
        for damage in ("mapping", "layout"):
            with self.subTest(damage=damage):
                directory = self.copy_bundle()

                def mutate(manifest):
                    if damage == "mapping":
                        constants = manifest["entries"]["decode"]["constants"]
                        name = next(iter(constants))
                        constants[name] = next(
                            view for view in manifest["tensors"] if view != constants[name]
                        )
                    else:
                        next(iter(manifest["tensors"].values()))["offset"] += 1

                self.rewrite_archive(directory, mutate, update_checksums=False)
                with patch(
                    "torch.cuda.is_available",
                    side_effect=AssertionError("CUDA must not be queried"),
                ):
                    with self.assertRaisesRegex(ValueError, "archive checksum mismatch"):
                        read_shared_weight_manifest(directory, self.contract)

    def test_owner_retains_borrowed_tensors_and_checks_exact_runtime_names(self):
        calls, pointers = [], []

        class BorrowingLoader:
            def __init__(self, names):
                self.names = names

            def get_constant_fqns(self):
                return self.names

            def load_constants(self, tensors, **options):
                # A borrowing runtime need not own Python Tensor references.
                calls.append(options)
                pointers.append(
                    {
                        name: (weakref.ref(value), value.data_ptr())
                        for name, value in tensors.items()
                    }
                )

        pool = SharedWeights(self.bundle, self.contract)
        names = list(pool.manifest["entries"]["decode"]["constants"])
        loader = BorrowingLoader(names)
        with patch("torch._inductor.aoti_load_package", return_value=loader) as load:
            self.assertIs(pool.load("decode"), loader)
            self.assertIs(pool.load("decode"), loader)
            self.assertEqual(load.call_count, 1)
        owner = weakref.ref(pool)
        del pool
        gc.collect()
        self.assertIsNotNone(owner(), "A retained borrowing model must keep its pool alive")
        self.assertEqual(calls, [dict(check_full_update=True, user_managed=True)])
        self.assertTrue(
            all(
                reference() is not None and reference().data_ptr() == pointer
                for reference, pointer in pointers[0].values()
            )
        )
        del loader, load
        gc.collect()
        self.assertIsNone(owner(), "The weak model cache must not create an ownership cycle")
        bad_pool = SharedWeights(self.bundle, self.contract)
        with patch(
            "torch._inductor.aoti_load_package", return_value=BorrowingLoader(["unrelated.weight"])
        ):
            with self.assertRaisesRegex(ValueError, "constant FQNs differ"):
                bad_pool.load("decode")

    def test_shared_weights_releases_cpu_buffers_as_storages_move(self):
        from compiler.common.weights import _archive_storages

        remaining = []

        class TrackingDict(dict):
            def pop(self, key, *args):
                value = super().pop(key, *args)
                remaining.append(len(self))
                return value

        def load_tracked(directory, manifest):
            return TrackingDict(_archive_storages(directory, manifest))

        with patch("compiler.common.weights._archive_storages", side_effect=load_tracked):
            pool = SharedWeights(self.bundle, self.contract)
        self.assertEqual(remaining, list(range(len(pool.manifest["storages"]) - 1, -1, -1)))


if __name__ == "__main__":
    unittest.main()
