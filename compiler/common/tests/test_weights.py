"""CPU roundtrip and rejection coverage for externally owned AOTI weights."""

import copy
import gc
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch
import weakref
import zipfile

import torch
import torch.nn.functional as F

from compiler.common.artifacts import sha256
from compiler.common.weights import SHARED_WEIGHT_OPTIONS, SharedWeights, compile_shared, validate_shared_weights
from compiler.common.tests.test_metadata import metadata_pair


class TiedViews(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base = torch.arange(24, dtype=torch.bfloat16).reshape(4, 6) / 16
        self.weight = torch.nn.Parameter(base, requires_grad=False)
        self.tied = self.weight
        self.sliced = torch.nn.Parameter(base[:, ::2], requires_grad=False)
        self.bias = torch.nn.Parameter(base[:, 1], requires_grad=False)
        self.register_buffer("scale", torch.tensor([1., 2., 4., 8.]))

    def forward(self, x):
        result = F.linear(x, self.weight).float() + F.linear(x, self.tied).float()
        result = (result + F.linear(x[:, ::2], self.sliced).float() + self.bias.float()) * self.scale
        return result, self.weight, self.tied, self.sliced, self.bias, self.scale


def inputs(rows):
    return torch.arange(rows * 6, dtype=torch.bfloat16).reshape(rows, 6) / 8


def export_entry(model, kind, directory):
    exported = torch.export.export(model, (inputs(1 if kind == "decode" else 3),))
    (directory / f"{kind}.graph.txt").write_text(exported.graph_module.code)
    return exported, lambda compiled: dict(
        entry_point=kind, marker="preserved metadata",
        package_bytes=compiled.stat().st_size, package_sha256=sha256(compiled))


def compile_bundle(model, directory, identity):
    with torch.inference_mode():
        return compile_shared(
            directory, lambda kind, stage: export_entry(model, kind, stage), identity=identity,
            inductor_configs={"emulate_precision_casts": True})


def metadata(directory, kind="decode"):
    return json.loads((directory / (kind + ".json")).read_text())


def mapping(directory):
    with zipfile.ZipFile(directory / "weights.pt") as archive:
        record = next(name for name in archive.namelist() if name.endswith("/extra/weights.json"))
        return json.loads(archive.read(record))


class SharedWeightBundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.bundle = cls.root / "complete"
        cls.identity = dict(model="cpu-tied-view-fixture", revision="one")
        cls.model = TiedViews().eval()
        with patch("torch.jit.save", wraps=torch.jit.save) as save_archive:
            cls.contract = compile_bundle(cls.model, cls.bundle, cls.identity)
        cls.archive_writes = save_archive.call_count
        cls.metadata = metadata(cls.bundle)

    def copy_bundle(self, source=None):
        temporary = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(temporary.cleanup)
        destination = Path(temporary.name) / "bundle"
        shutil.copytree(source or self.bundle, destination)
        return destination

    def rewrite_archive(self, directory, mutate=None, *, update_checksums=True, include_mapping=True):
        payload = mapping(directory)
        if mutate:
            mutate(payload)
        path = directory / "weights.pt"
        archive = torch.jit.load(str(path), map_location="cpu")
        rewritten = directory / "rewritten.pt"
        extra = {"weights.json": json.dumps(payload)} if include_mapping else {}
        torch.jit.save(archive, str(rewritten), _extra_files=extra)
        rewritten.replace(path)
        contract = dict(metadata(directory)["shared_weights"], bytes=path.stat().st_size, sha256=sha256(path))
        if update_checksums:
            for kind in ("prefill", "decode"):
                package = metadata(directory, kind)
                package["shared_weights"] = contract
                (directory / (kind + ".json")).write_text(json.dumps(package))
        return contract

    def test_one_export_publishes_complete_pair_with_one_archive(self):
        self.assertEqual(self.archive_writes, 1)
        self.assertEqual(set(mapping(self.bundle)["entries"]), {"prefill", "decode"})
        prefill, decode = (metadata(self.bundle, kind) for kind in ("prefill", "decode"))
        self.assertEqual(prefill["shared_weights"], decode["shared_weights"])
        self.assertEqual(self.contract, decode["shared_weights"])
        self.assertEqual(self.contract["bytes"], (self.bundle / "weights.pt").stat().st_size)
        self.assertEqual(self.contract["sha256"], sha256(self.bundle / "weights.pt"))
        self.assertFalse((self.bundle / "weights.json").exists())
        for kind, package in (("prefill", prefill), ("decode", decode)):
            self.assertEqual(package["entry_point"], kind)
            self.assertEqual(package["marker"], "preserved metadata")
            self.assertEqual(package["package_sha256"], sha256(self.bundle / (kind + ".pt2")))
            self.assertTrue((self.bundle / (kind + ".graph.txt")).is_file())
        validate_shared_weights(self.bundle, prefill, decode)
        directory = self.copy_bundle()
        (directory / "weights.json").write_text("unused old sidecar")
        validate_shared_weights(directory, metadata(directory, "prefill"), metadata(directory))

    def versioned_metadata(self):
        pair = metadata_pair()
        for kind, value in pair.items():
            package = self.bundle / f"{kind}.pt2"
            value.update(package_bytes=package.stat().st_size, package_sha256=sha256(package),
                         shared_weights=self.contract)
        return pair

    def test_versioned_file_facts_fail_before_archive_access(self):
        for kind in ("prefill", "decode"):
            for field, value, error in (("package_bytes", 1, "Package file size mismatch"),
                                        ("package_sha256", "0" * 64, "Package checksum mismatch")):
                with self.subTest(kind=kind, field=field):
                    pair = self.versioned_metadata()
                    pair[kind][field] = value
                    with patch("compiler.common.weights._manifest") as archive:
                        with self.assertRaisesRegex(ValueError, error):
                            validate_shared_weights(self.bundle, pair["prefill"], pair["decode"])
                        archive.assert_not_called()

    def test_versioned_metadata_rejects_a_different_archive_model(self):
        pair = self.versioned_metadata()
        with self.assertRaisesRegex(ValueError, "Package model identity mismatch"):
            validate_shared_weights(self.bundle, pair["prefill"], pair["decode"])

    def test_real_cpu_packages_share_tied_strided_storage(self):
        manifest = validate_shared_weights(self.bundle, self.metadata, self.metadata)
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
                torch.testing.assert_close(compiled(inputs(rows)), self.model(inputs(rows)), rtol=0, atol=0)

    def test_compile_options_have_one_owner_and_do_not_mutate_the_caller(self):
        directory = self.root / "compile-options"
        requested = {"emulate_precision_casts": True, "aot_inductor.package_constants_in_so": True}
        original = dict(requested)

        def prepare(kind, stage):
            exported, factory = export_entry(self.model, kind, stage)
            return exported, lambda path: dict(
                factory(path), inductor_configs={"always_keep_tensor_constants": False})

        with torch.inference_mode(), patch(
                "torch._inductor.aoti_compile_and_package",
                wraps=torch._inductor.aoti_compile_and_package) as compile_graph:
            compile_shared(directory, prepare, identity=self.identity, inductor_configs=requested)
        self.assertEqual(requested, original)
        self.assertEqual(compile_graph.call_count, 2)
        first, second = (call.kwargs["inductor_configs"] for call in compile_graph.call_args_list)
        self.assertIsNot(first, second)
        for kind, call in zip(("prefill", "decode"), compile_graph.call_args_list):
            recorded = metadata(directory, kind)["inductor_configs"]
            self.assertEqual(recorded, call.kwargs["inductor_configs"])
            for key, value in {**requested, **SHARED_WEIGHT_OPTIONS}.items():
                self.assertEqual(recorded[key], value)

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
        self.assertEqual(before, {path.name: sha256(path) for path in populated.iterdir() if path.is_file()})
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
        validate_shared_weights(directory, metadata(directory, "prefill"), metadata(directory))

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
                        exported, factory = export_entry(self.model, kind, stage)
                        if failure == "metadata" and kind == "decode":
                            def factory(_):
                                raise RuntimeError("decode metadata failed")
                        return exported, factory

                    def compile_graph(exported, **options):
                        if failure == "compile" and Path(options["package_path"]).stem == "decode":
                            raise RuntimeError("decode compile failed")
                        return real_compile(exported, **options)

                    with torch.inference_mode(), patch(
                            "torch._inductor.aoti_compile_and_package", side_effect=compile_graph), patch(
                            "torch.jit.save", wraps=torch.jit.save) as save_archive:
                        with self.assertRaisesRegex(RuntimeError, f"decode {failure} failed"):
                            compile_shared(directory, prepare, identity=self.identity, inductor_configs={})
                    self.assertEqual(kinds, ["prefill", "decode"])
                    save_archive.assert_not_called()
                    self.assertEqual(set(root.iterdir()), before, "Staging files must be cleaned up")
                    if existing_empty:
                        self.assertEqual(list(directory.iterdir()), [])
                    else:
                        self.assertFalse(directory.exists())

    def test_changed_parameter_between_entries_is_rejected_before_publication(self):
        directory = self.root / "changed-source"
        changed = TiedViews().eval()
        with torch.inference_mode():
            changed.weight.add_(1)
        with torch.inference_mode(), self.assertRaisesRegex(
                ValueError, "source tensor differs between entries"):
            compile_shared(
                directory,
                lambda kind, stage: export_entry(changed if kind == "decode" else self.model, kind, stage),
                identity=self.identity, inductor_configs={})
        self.assertFalse(directory.exists())

    def test_missing_and_corrupt_bundle_files_are_rejected_without_cuda(self):
        for filename, corrupt, message in (
            ("weights.pt", False, "archive is missing"),
            ("weights.pt", True, "archive checksum mismatch"),
            ("prefill.pt2", True, "package checksum mismatch"),
            ("prefill.pt2", False, "package checksum mismatch"),
        ):
            with self.subTest(filename=filename, corrupt=corrupt):
                directory = self.copy_bundle()
                path = directory / filename
                path.write_bytes(b"corrupt") if corrupt else path.unlink()
                with patch("torch.cuda.is_available", side_effect=AssertionError("CUDA must not be queried")):
                    with self.assertRaisesRegex(ValueError, message):
                        validate_shared_weights(directory, self.metadata, self.metadata)

    def test_missing_embedded_mapping_is_rejected_without_cuda(self):
        directory = self.copy_bundle()
        self.rewrite_archive(directory, include_mapping=False)
        with patch("torch.cuda.is_available", side_effect=AssertionError("CUDA must not be queried")):
            with self.assertRaisesRegex(ValueError, "archive metadata is invalid"):
                validate_shared_weights(directory, metadata(directory, "prefill"), metadata(directory))

    def test_contract_and_compiler_options_cannot_select_another_pool(self):
        altered = copy.deepcopy(self.metadata)
        altered["shared_weights"]["identity"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "contracts differ"):
            validate_shared_weights(self.bundle, altered, self.metadata)
        altered = copy.deepcopy(self.metadata)
        altered["shared_weights"]["file"] = "../weights.pt"
        with self.assertRaisesRegex(ValueError, "metadata is required"):
            validate_shared_weights(self.bundle, altered, altered)
        with self.assertRaisesRegex(ValueError, "metadata is required"):
            validate_shared_weights(self.bundle, {}, self.metadata)
        altered = copy.deepcopy(self.metadata)
        altered["inductor_configs"]["aot_inductor.package_constants_in_so"] = True
        with self.assertRaisesRegex(ValueError, "weightless package compiler options"):
            validate_shared_weights(self.bundle, altered, altered)

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
                        constants[name] = next(view for view in manifest["tensors"] if view != constants[name])
                    else:
                        next(iter(manifest["tensors"].values()))["offset"] += 1
                self.rewrite_archive(directory, mutate, update_checksums=False)
                with patch("torch.cuda.is_available", side_effect=AssertionError("CUDA must not be queried")):
                    with self.assertRaisesRegex(ValueError, "archive checksum mismatch"):
                        validate_shared_weights(directory, self.metadata, self.metadata)

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
                pointers.append({name: (weakref.ref(value), value.data_ptr()) for name, value in tensors.items()})

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
        self.assertTrue(all(reference() is not None and reference().data_ptr() == pointer
                            for reference, pointer in pointers[0].values()))
        del loader, load
        gc.collect()
        self.assertIsNone(owner(), "The weak model cache must not create an ownership cycle")
        bad_pool = SharedWeights(self.bundle, self.contract)
        with patch("torch._inductor.aoti_load_package", return_value=BorrowingLoader(["unrelated.weight"])):
            with self.assertRaisesRegex(ValueError, "constant FQNs differ"):
                bad_pool.load("decode")


if __name__ == "__main__":
    unittest.main()
