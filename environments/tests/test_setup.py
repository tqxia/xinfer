"""Native package linking must preserve the source and a fresh venv's seed files."""

from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest

from environments.setup import link_packages, main


class NativePackageLinks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "native"
        self.site = self.root / "venv/site-packages"
        self.source.mkdir()
        self.site.mkdir(parents=True)
        self.module = self.source / "typing_extensions.py"
        self.module.write_text("native_version = 1\n")

    def test_files_directories_metadata_and_shared_cuda_namespace(self):
        for name in ("torch", "torchgen", "functorch", "torch-2.7.0.dist-info"):
            directory = self.source / name
            directory.mkdir()
            (directory / "native.txt").write_text(name)
        # Several NVIDIA distributions share this one top-level package.
        for name in ("cublas", "cudnn"):
            directory = self.source / "nvidia" / name
            directory.mkdir(parents=True)
            (directory / "library.so").write_bytes(name.encode())
        links = {p.name: str(p) for p in self.source.iterdir()}
        before = {str(p.relative_to(self.source)): p.read_bytes()
                  for p in self.source.rglob("*") if p.is_file()}

        link_packages(self.site, links)

        self.assertEqual({p.name for p in self.site.iterdir()}, set(links))
        for name, source in links.items():
            destination = self.site / name
            self.assertTrue(destination.is_symlink())
            self.assertEqual(destination.resolve(), Path(source).resolve())
        self.assertEqual((self.site / "typing_extensions.py").read_text(), "native_version = 1\n")
        for name in ("cublas", "cudnn"):
            self.assertEqual((self.site / "nvidia" / name / "library.so").read_bytes(), name.encode())
        after = {str(p.relative_to(self.source)): p.read_bytes()
                 for p in self.source.rglob("*") if p.is_file()}
        self.assertEqual(after, before)

    def test_existing_seed_file_prevents_every_link(self):
        seed = self.site / "setuptools"
        seed.write_text("venv seed\n")
        links = {"typing_extensions.py": str(self.module), "setuptools": str(self.module)}

        with self.assertRaises(FileExistsError):
            link_packages(self.site, links)

        self.assertEqual(list(self.site.iterdir()), [seed])
        self.assertFalse(seed.is_symlink())
        self.assertEqual(seed.read_text(), "venv seed\n")

    def test_broken_destination_symlink_is_preserved(self):
        seed = self.site / "setuptools"
        absent = self.root / "absent"
        seed.symlink_to(absent)
        links = {"typing_extensions.py": str(self.module), "setuptools": str(self.module)}

        with self.assertRaises(FileExistsError):
            link_packages(self.site, links)

        self.assertEqual(list(self.site.iterdir()), [seed])
        self.assertTrue(seed.is_symlink())
        self.assertEqual(seed.readlink(), absent)

    def test_invalid_component_cannot_escape_site_packages(self):
        for name in ("../outside", "nested/package", str(self.root / "outside"), ".", ".."):
            with self.subTest(name=name), self.assertRaises(ValueError):
                link_packages(self.site, {"typing_extensions.py": str(self.module), name: str(self.module)})
            self.assertEqual(list(self.site.iterdir()), [])
        self.assertFalse((self.root / "outside").exists())

    def test_missing_or_broken_source_prevents_every_link(self):
        broken = self.source / "broken"
        broken.symlink_to(self.root / "absent")
        for source in (self.root / "missing", broken):
            with self.subTest(source=source), self.assertRaises(FileNotFoundError):
                link_packages(self.site, {"typing_extensions.py": str(self.module), "other": str(source)})
            self.assertEqual(list(self.site.iterdir()), [])

    def test_cli_refuses_existing_output_before_inspecting_native_python(self):
        output = self.root / "existing-venv"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("existing environment\n")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["--target", "l20", "--model", "qwen3", "--native-python", "/missing/python",
                  "--output", str(output)])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(marker.read_text(), "existing environment\n")


if __name__ == "__main__":
    unittest.main()
